#include "ops/linear/gguf/ggml_bridge_internal.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::ops::gguf::detail {
static DeviceFacts q6_probe_device_facts{860, 82, 101376};

const DeviceFacts& device_facts() { return q6_probe_device_facts; }

static void initialize_device_facts() {
    int device = 0;
    check(cudaGetDevice(&device), "get device");
    cudaDeviceProp properties{};
    check(cudaGetDeviceProperties(&properties, device), "get device properties");
    q6_probe_device_facts = {100 * properties.major + 10 * properties.minor,
                             properties.multiProcessorCount, properties.sharedMemPerBlockOptin};
}
}

extern "C" void ggml_abort(const char* file, int line, const char* fmt, ...) {
    std::fprintf(stderr, "ggml abort at %s:%d: ", file, line);
    va_list args;
    va_start(args, fmt);
    std::vfprintf(stderr, fmt, args);
    va_end(args);
    std::fputc('\n', stderr);
    std::abort();
}

#include "ops/linear/gguf/ggml_mmq_q6_k.cu"

using namespace ninfer::ops::gguf::detail;

static constexpr int kHeadRows = 248320;
static constexpr int kHeadK = 5120;
static constexpr int kBlockBytes = 210;
static constexpr std::uint64_t kScratchCap = 96ull * 1024ull * 1024ull;
static constexpr std::array<int, 7> kColumns{9, 12, 16, 32, 64, 128, 1024};

struct ChunkPlan {
    int rows = 0;
    std::int64_t scratch_bytes = 0;
    MatrixLaunch launch{};
};

static std::vector<ChunkPlan> make_plan(int rows, int k, int columns, int max_rows) {
    const auto full = select_matrix_launch<GGML_TYPE_Q6_K>(rows, columns);
    if (full.fixup || rows % full.I != 0) { throw std::runtime_error("full MMQ selection needs fixup or partial row tile"); }
    const auto partition = ninfer::ops::gguf::q6_matrix_partition(rows, columns, max_rows);
    if (partition.row_tile != full.I || partition.parts <= 0) { throw std::runtime_error("partition row tile mismatch"); }
    const int total_tiles = rows / full.I;
    const int low_tiles = total_tiles / partition.parts;
    const int high_count = total_tiles % partition.parts;
    std::vector<ChunkPlan> chunks;
    int row = 0;
    for (int i = 0; i < partition.parts; ++i) {
        const int tiles = low_tiles + (i < high_count ? 1 : 0);
        const int chunk_rows = tiles * full.I;
        const auto launch = select_matrix_launch<GGML_TYPE_Q6_K>(chunk_rows, columns);
        if (launch.fixup || launch.J != full.J || launch.I != full.I || launch.threads != full.threads ||
            launch.shared != full.shared || launch.stream_k != full.stream_k || launch.blocks != tiles * full.column_tiles) {
            throw std::runtime_error("chunk MMQ launch differs from full J/I/K schedule");
        }
        chunks.push_back({chunk_rows, std::int64_t(chunk_rows) * (k / 256) * kBlockBytes, launch});
        row += chunk_rows;
    }
    if (row != rows || chunks.size() != size_t(partition.parts)) { throw std::runtime_error("chunk coverage mismatch"); }
    return chunks;
}

static void print_launch(const char* name, const MatrixLaunch& l) {
    if (name[0] != '\0') { std::cout << "\"" << name << "\":"; }
    std::cout << '{';
    std::cout << "\"J\":" << l.J << ",\"I\":" << l.I << ",\"threads\":" << l.threads
              << ",\"shared_bytes\":" << l.shared << ",\"row_tiles\":" << l.row_tiles
              << ",\"column_tiles\":" << l.column_tiles << ",\"stream_k\":" << (l.stream_k ? "true" : "false")
              << ",\"blocks\":" << l.blocks << ",\"fixup\":" << (l.fixup ? "true" : "false") << '}';
}

static void print_plan(int rows, int k, int columns) {
    const int row_bytes = (k / 256) * kBlockBytes;
    const int full_scratch_rows = int(kScratchCap / row_bytes);
    const auto full = select_matrix_launch<GGML_TYPE_Q6_K>(rows, columns);
    const int max_rows = full_scratch_rows / full.I * full.I;
    const auto partition = ninfer::ops::gguf::q6_matrix_partition(rows, columns, max_rows);
    const auto chunks = make_plan(rows, k, columns, max_rows);
    int low_rows = rows, high_rows = 0;
    std::uint64_t max_chunk_bytes = 0;
    for (const auto& chunk : chunks) {
        low_rows = std::min(low_rows, chunk.rows);
        high_rows = std::max(high_rows, chunk.rows);
        max_chunk_bytes = std::max(max_chunk_bytes, std::uint64_t(chunk.scratch_bytes));
    }
    std::cout << "{\"N\":" << rows << ",\"K\":" << k << ",\"columns\":" << columns
              << ",\"device_facts\":{\"cc\":" << device_facts().cc << ",\"sm_count\":" << device_facts().sm_count
              << ",\"shared_per_block_optin\":" << device_facts().shared_per_block_optin << '}'
              << ",\"row_bytes\":" << row_bytes << ",\"full_raw_bytes\":" << std::uint64_t(rows) * row_bytes
              << ",\"scratch_cap_bytes\":" << kScratchCap << ",\"max_rows_from_cap_before_tile_rounding\":" << full_scratch_rows
              << ",\"max_rows_per_chunk\":" << max_rows << ",\"parts\":" << partition.parts
              << ",\"row_tile\":" << partition.row_tile << ",\"low_chunk_rows\":" << low_rows
              << ",\"high_chunk_rows\":" << high_rows << ",\"max_chunk_raw_bytes\":" << max_chunk_bytes
              << ",\"max_chunk_under_cap\":" << (max_chunk_bytes <= kScratchCap ? "true" : "false") << ',';
    print_launch("full_launch", full);
    std::cout << ',';
    print_launch("low_launch", chunks.front().launch);
    std::cout << ",\"high_launch\":";
    auto high = chunks.front().launch;
    for (const auto& chunk : chunks) { if (chunk.rows == high_rows) { high = chunk.launch; break; } }
    print_launch("", high);
    std::cout << ",\"same_original_JI_and_K_reduction\":true,\"all_chunk_launches_nofixup\":true}\n";
}

static void fill_activation(std::vector<block_q8_1_mmq>& activation, int k, int columns) {
    const int blocks_per_column = k / QK8_1_MMQ;
    for (int column = 0; column < columns; ++column) {
        for (int block = 0; block < blocks_per_column; ++block) {
            auto& value = activation[size_t(column) * blocks_per_column + block];
            for (int d = 0; d < 4; ++d) { value.d4[d] = 0.0078125f * float(1 + ((column + block + d) % 16)); }
            for (int i = 0; i < QK8_1_MMQ; ++i) { value.qs[i] = std::int8_t(((column * 17 + block * 13 + i * 7) % 31) - 15); }
        }
    }
}

__global__ static void convert_bf16(const float* source, __nv_bfloat16* destination, std::uint64_t count) {
    const auto i = std::uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) { destination[i] = __float2bfloat16(source[i]); }
}

static void check_equal(const std::vector<float>& whole, const std::vector<float>& chunked) {
    if (whole.size() != chunked.size() || std::memcmp(whole.data(), chunked.data(), whole.size() * sizeof(float)) != 0) {
        throw std::runtime_error("whole versus chunked FP32 output mismatch");
    }
}

static void check_equal(const std::vector<std::uint16_t>& whole, const std::vector<std::uint16_t>& chunked) {
    if (whole != chunked) { throw std::runtime_error("whole versus chunked BF16 output mismatch"); }
}

static void run_bench(const std::string& model, std::uint64_t offset, int rows, int k, int columns) {
    initialize_device_facts();
    const auto row_bytes = std::int64_t(k / 256) * kBlockBytes;
    const int max_rows = int(kScratchCap / row_bytes / 128) * 128;
    const auto chunks = make_plan(rows, k, columns, max_rows);
    std::vector<std::uint8_t> host_weights(size_t(rows) * row_bytes);
    std::ifstream file(model, std::ios::binary);
    if (!file) { throw std::runtime_error("cannot open model"); }
    file.seekg(std::streamoff(offset));
    file.read(reinterpret_cast<char*>(host_weights.data()), std::streamsize(host_weights.size()));
    if (size_t(file.gcount()) != host_weights.size()) { throw std::runtime_error("truncated Q6_K matrix"); }
    std::vector<block_q8_1_mmq> host_activation(size_t(columns) * (k / QK8_1_MMQ) + 128);
    fill_activation(host_activation, k, columns);
    std::uint8_t* weights = nullptr;
    block_q8_1_mmq* activation = nullptr;
    float* whole_out = nullptr;
    float* chunked_out = nullptr;
    __nv_bfloat16* whole_bf16 = nullptr;
    __nv_bfloat16* chunked_bf16 = nullptr;
    auto alloc = [](auto** ptr, size_t bytes) { check(cudaMalloc(ptr, bytes), "probe allocation"); };
    alloc(&weights, host_weights.size());
    check(cudaMemcpy(weights, host_weights.data(), host_weights.size(), cudaMemcpyHostToDevice), "weight upload");
    alloc(&activation, host_activation.size() * sizeof(block_q8_1_mmq));
    check(cudaMemcpy(activation, host_activation.data(), host_activation.size() * sizeof(block_q8_1_mmq), cudaMemcpyHostToDevice), "activation upload");
    const size_t output_count = size_t(rows) * columns;
    alloc(&whole_out, output_count * sizeof(float));
    alloc(&chunked_out, output_count * sizeof(float));
    alloc(&whole_bf16, output_count * sizeof(__nv_bfloat16));
    alloc(&chunked_bf16, output_count * sizeof(__nv_bfloat16));
    cudaStream_t stream;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "probe stream");
    auto whole = [&] {
        matrix_product_impl<GGML_TYPE_Q6_K>(weights, row_bytes, rows, k, activation, columns,
                                                    whole_out, rows, nullptr, stream);
    };
    auto row_chunks = [&] {
        int row = 0;
        for (const auto& chunk : chunks) {
            matrix_product_impl<GGML_TYPE_Q6_K>(weights + std::int64_t(row) * row_bytes,
                row_bytes, chunk.rows, k, activation, columns, chunked_out + row, rows, nullptr, stream);
            row += chunk.rows;
        }
        if (row != rows) { throw std::runtime_error("GPU row chunk coverage mismatch"); }
    };
    whole();
    row_chunks();
    check(cudaStreamSynchronize(stream), "warmup sync");
    cudaGraphExec_t graphs[2]{};
    for (int mode = 0; mode < 2; ++mode) {
        cudaGraph_t graph;
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "graph capture begin");
        if (mode == 0) { whole(); } else { row_chunks(); }
        check(cudaStreamEndCapture(stream, &graph), "graph capture end");
        check(cudaGraphInstantiate(&graphs[mode], graph, 0), "graph instantiate");
        check(cudaGraphDestroy(graph), "graph destroy");
    }
    for (int repeat = 0; repeat < 3; ++repeat) {
        check(cudaGraphLaunch(graphs[0], stream), "whole graph replay");
        check(cudaGraphLaunch(graphs[1], stream), "chunk graph replay");
    }
    check(cudaStreamSynchronize(stream), "graph replay sync");
    convert_bf16<<<(output_count + 255) / 256, 256, 0, stream>>>(whole_out, whole_bf16, output_count);
    convert_bf16<<<(output_count + 255) / 256, 256, 0, stream>>>(chunked_out, chunked_bf16, output_count);
    check(cudaGetLastError(), "BF16 convert launch");
    check(cudaStreamSynchronize(stream), "output sync");
    std::vector<float> host_whole(output_count), host_chunked(output_count);
    std::vector<std::uint16_t> host_whole_bf16(output_count), host_chunked_bf16(output_count);
    check(cudaMemcpy(host_whole.data(), whole_out, output_count * sizeof(float), cudaMemcpyDeviceToHost), "whole FP32 copy");
    check(cudaMemcpy(host_chunked.data(), chunked_out, output_count * sizeof(float), cudaMemcpyDeviceToHost), "chunked FP32 copy");
    check(cudaMemcpy(host_whole_bf16.data(), whole_bf16, output_count * sizeof(std::uint16_t), cudaMemcpyDeviceToHost), "whole BF16 copy");
    check(cudaMemcpy(host_chunked_bf16.data(), chunked_bf16, output_count * sizeof(std::uint16_t), cudaMemcpyDeviceToHost), "chunked BF16 copy");
    check_equal(host_whole, host_chunked);
    check_equal(host_whole_bf16, host_chunked_bf16);
    std::cout << "{\"mode\":\"gpu_bench\",\"N\":" << rows << ",\"K\":" << k << ",\"columns\":" << columns
              << ",\"whole_fp32_bit_exact\":true,\"whole_bf16_bit_exact\":true,\"whole_and_chunk_graph_replay_exact\":true"
              << ",\"activation\":\"synthetic_q8_1_mmq\",\"parts\":" << chunks.size()
              << ",\"max_chunk_raw_bytes\":" << std::max_element(chunks.begin(), chunks.end(),
                    [](const ChunkPlan& a, const ChunkPlan& b) { return a.scratch_bytes < b.scratch_bytes; })->scratch_bytes
              << ",\"max_chunk_under_96mib\":true}\n";
    for (auto graph : graphs) { check(cudaGraphExecDestroy(graph), "graph exec destroy"); }
    check(cudaStreamDestroy(stream), "stream destroy");
    check(cudaFree(chunked_bf16), "free");
    check(cudaFree(whole_bf16), "free");
    check(cudaFree(chunked_out), "free");
    check(cudaFree(whole_out), "free");
    check(cudaFree(activation), "free");
    check(cudaFree(weights), "free");
}

int main(int argc, char** argv) {
    if (argc == 5 && std::string(argv[1]) == "plan") {
        const int rows = std::stoi(argv[2]);
        const int k = std::stoi(argv[3]);
        const int columns = std::stoi(argv[4]);
        print_plan(rows, k, columns);
        return 0;
    }
    if (argc == 7 && std::string(argv[1]) == "bench") {
        run_bench(argv[2], std::stoull(argv[3]), std::stoi(argv[4]), std::stoi(argv[5]), std::stoi(argv[6]));
        return 0;
    }
    std::cerr << "q6_mmq_probe plan N K columns | bench model offset N K columns\n";
    return 2;
}
