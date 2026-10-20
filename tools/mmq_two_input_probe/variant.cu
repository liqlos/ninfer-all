#include "probe_api.h"
#include "mmq_two_input_native.cuh"

#include <climits>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>

#ifndef NINFER_LAB_MMQ_TWO_INPUT
#define NINFER_LAB_MMQ_TWO_INPUT 0
#endif

#if NINFER_LAB_MMQ_TWO_INPUT != 0 && NINFER_LAB_MMQ_TWO_INPUT != 1
#error NINFER_LAB_MMQ_TWO_INPUT_must_be_zero_or_one
#endif

#if NINFER_LAB_MMQ_TWO_INPUT
#define MMQ_TWO_INPUT_ENTRY mmq_two_input_candidate
#else
#define MMQ_TWO_INPUT_ENTRY mmq_two_input_control
#endif

namespace {

constexpr int kColumns = 1280;
constexpr int kI = 128;
constexpr int kJ = 128;
constexpr int kThreads = 256;
constexpr int kSharedBytes = 57856;

static_assert(sizeof(block_q8_1_mmq) == 144);
static_assert(sizeof(block_q4_K) == 144);
static_assert(sizeof(block_q6_K) == 210);

struct Range {
    std::uintptr_t first;
    std::uintptr_t last;
};

bool range_for(const void* pointer, std::size_t bytes, Range& range) {
    const auto first = reinterpret_cast<std::uintptr_t>(pointer);
    if (first == 0 || (first & 15U) != 0 || bytes == 0 ||
        bytes > std::numeric_limits<std::uintptr_t>::max() - first) {
        return false;
    }
    range = {first, first + bytes};
    return true;
}

bool overlaps(const Range& first, const Range& second) {
    return first.first < second.last && second.first < first.last;
}

void check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

template <ggml_type Type, bool Store, bool Accumulate>
bool prepare_device() {
    struct State {
        std::once_flag once;
        bool admitted = false;
    };
    static State states[16];
    int device = -1;
    check(cudaGetDevice(&device), "two-input current device");
    if (device < 0 || device >= 16) { return false; }
    auto& state = states[device];
    std::call_once(state.once, [&state] {
        int current = -1;
        cudaDeviceProp properties{};
        check(cudaGetDevice(&current), "two-input device initialization");
        check(cudaGetDeviceProperties(&properties, current), "two-input properties");
        if (properties.major != 8 || properties.minor != 6 ||
            properties.sharedMemPerBlockOptin < kSharedBytes) {
            return;
        }
        const auto config = ggml_cuda_mmq_get_config(Type, kJ, false, 860);
        if (config.I != kI || config.nthreads != kThreads || !config.stream_k ||
            mmq_get_nbytes_shared(config, 860) != kSharedBytes) {
            return;
        }
        check(cudaFuncSetAttribute(
                  mul_mat_q<Type, kJ, false, GGML_PREC_Q8, Store, Accumulate, false, false>,
                  cudaFuncAttributeMaxDynamicSharedMemorySize, kSharedBytes),
              "two-input shared memory");
        state.admitted = true;
    });
    return state.admitted;
}

template <ggml_type Type, bool Store, bool Accumulate>
bool run(const MmqTwoInputArguments& a, cudaStream_t stream) {
    static_assert(!Accumulate || Store);
    constexpr int rows = Type == GGML_TYPE_Q4_K ? 17408 : 5120;
    constexpr int k = Type == GGML_TYPE_Q4_K ? 5120 : 17408;
    constexpr int layout = Type == GGML_TYPE_Q4_K ? MMQ_Q8_1_DS_LAYOUT_DS4
                                                 : MMQ_Q8_1_DS_LAYOUT_D4;
    constexpr int weight_block_bytes = Type == GGML_TYPE_Q4_K ? 144 : 210;
    constexpr std::int64_t row_bytes = std::int64_t(k / 256) * weight_block_bytes;
    constexpr std::size_t weight_bytes = std::size_t(rows) * row_bytes;
    constexpr std::int64_t packed_panel_bytes = std::int64_t(k / 128) * kColumns * 144;
    constexpr std::int64_t element_bytes = Store ? 2 : 4;
    if (a.rows != rows || a.k != k || a.columns != kColumns ||
        a.activation_layout != layout || a.weight_row_stride_bytes != row_bytes ||
        a.weight_capacity_bytes < weight_bytes ||
        a.packed_sample_stride_bytes < packed_panel_bytes ||
        a.packed_sample_stride_bytes % 16 != 0 ||
        a.packed_sample_stride_bytes / 4 > INT_MAX - packed_panel_bytes / 4 ||
        a.output_column_stride_elements < rows ||
        a.output_column_stride_elements > INT_MAX / kColumns ||
        a.output_sample_stride_elements < 0 ||
        a.output_sample_stride_elements > INT_MAX) {
        return false;
    }
    const std::int64_t output_panel_elements =
        (kColumns - 1) * a.output_column_stride_elements + rows;
    if (a.output_sample_stride_elements < output_panel_elements ||
        a.output_sample_stride_elements > INT_MAX - output_panel_elements ||
        (a.output_sample_stride_elements * element_bytes) % 16 != 0) {
        return false;
    }
    const auto packed_bytes = std::size_t(a.packed_sample_stride_bytes + packed_panel_bytes);
    const auto output_bytes =
        std::size_t(a.output_sample_stride_elements + output_panel_elements) * element_bytes;
    if (a.packed_capacity_bytes < packed_bytes || a.output_capacity_bytes < output_bytes) {
        return false;
    }
    Range weights{}, inputs{}, output{};
    if (!range_for(a.weight, weight_bytes, weights) ||
        !range_for(a.packed_inputs, packed_bytes, inputs) ||
        !range_for(a.output, output_bytes, output) ||
        overlaps(output, weights) || overlaps(output, inputs)) {
        return false;
    }
    if (!prepare_device<Type, Store, Accumulate>()) { return false; }
    const auto one = init_fastdiv_values(1);
    const auto two = init_fastdiv_values(2);
    const auto columns = init_fastdiv_values(kColumns / kJ);
    const auto k_blocks = init_fastdiv_values(k / 256);
    constexpr int single_blocks = (rows / kI) * (kColumns / kJ);
    constexpr int samples = NINFER_LAB_MMQ_TWO_INPUT ? 2 : 1;
    constexpr int launches = NINFER_LAB_MMQ_TWO_INPUT ? 1 : 2;
    const auto samples_fd = NINFER_LAB_MMQ_TWO_INPUT ? two : one;
    const int input_sample_words = static_cast<int>(a.packed_sample_stride_bytes / 4);
    const int output_sample_elements = static_cast<int>(a.output_sample_stride_elements);
    const int output_column_elements = static_cast<int>(a.output_column_stride_elements);
    for (int panel = 0; panel < launches; ++panel) {
        const auto* input = static_cast<const int*>(a.packed_inputs) + panel * input_sample_words;
        float* output_fp32 = nullptr;
        __nv_bfloat16* output_bf16 = nullptr;
        if constexpr (Store) {
            output_bf16 = static_cast<__nv_bfloat16*>(a.output) + panel * output_sample_elements;
        } else {
            output_fp32 = static_cast<float*>(a.output) + panel * output_sample_elements;
        }
        mul_mat_q<Type, kJ, false, GGML_PREC_Q8, Store, Accumulate, false, false>
            <<<dim3(samples * single_blocks, 1, 1), dim3(32, 8, 1), kSharedBytes, stream>>>(
                static_cast<const char*>(a.weight), input, nullptr, nullptr,
                output_fp32, output_bf16, nullptr, nullptr,
                k_blocks, rows, kColumns, k / 256, kColumns, output_column_elements,
                one, one, 0, 0, 0,
                samples_fd, samples_fd, 0,
                NINFER_LAB_MMQ_TWO_INPUT ? input_sample_words : 0,
                NINFER_LAB_MMQ_TWO_INPUT ? output_sample_elements : 0, columns);
        check(cudaGetLastError(), "two-input native MMQ launch");
    }
    return true;
}

template <ggml_type Type>
bool dispatch(const MmqTwoInputArguments& arguments, cudaStream_t stream) {
    switch (arguments.epilogue) {
        case MmqTwoInputEpilogue::PlainFp32: return run<Type, false, false>(arguments, stream);
        case MmqTwoInputEpilogue::StoreBf16: return run<Type, true, false>(arguments, stream);
        case MmqTwoInputEpilogue::AccumulateBf16: return run<Type, true, true>(arguments, stream);
    }
    return false;
}

}

bool MMQ_TWO_INPUT_ENTRY(const MmqTwoInputArguments& arguments, cudaStream_t stream) {
    switch (arguments.profile) {
        case MmqTwoInputProfile::Q4GateOrUp: return dispatch<GGML_TYPE_Q4_K>(arguments, stream);
        case MmqTwoInputProfile::Q6Down: return dispatch<GGML_TYPE_Q6_K>(arguments, stream);
    }
    return false;
}
