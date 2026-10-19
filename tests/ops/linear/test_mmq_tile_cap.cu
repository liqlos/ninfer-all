#include "ops/linear/gguf/ggml_bridge.h"

#include <cuda_runtime.h>

#include <array>
#include <iostream>
#include <stdexcept>
#include <string>

#if !defined(NINFER_GGUF_SM86_MMQ_MAX_J)
#error NINFER_GGUF_SM86_MMQ_MAX_J must be provided by the test target
#endif

namespace {

using ninfer::ops::gguf::GgmlType;
using ninfer::ops::gguf::MatrixBf16StorePlan;

void check(cudaError_t status, const char* label) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(label) + ": " + cudaGetErrorString(status));
    }
}

int expected_unlimited_j(int columns) {
    for (int j : {16, 32, 64, 128}) {
        if (columns <= j) { return j; }
    }
    return 128;
}

void check_type(GgmlType type, const char* name, int rows,
                const std::array<int, 8>& widths) {
    for (int columns : widths) {
        MatrixBf16StorePlan plan;
        if (!ninfer::ops::gguf::matrix_bf16_store_plan(type, rows, columns, plan)) {
            throw std::runtime_error(std::string(name) + " did not admit the expected MMQ plan at T=" +
                                     std::to_string(columns));
        }
        if (plan.J != 16 && plan.J != 32 && plan.J != 64 && plan.J != 128) {
            throw std::runtime_error(std::string(name) + " selected an unknown MMQ tile");
        }
        if (plan.I != 64 && plan.I != 128) {
            throw std::runtime_error(std::string(name) + " selected an unexpected MMQ row tile");
        }
        if (columns < 64) {
            const int expected = expected_unlimited_j(columns);
            if (plan.J != expected) {
                throw std::runtime_error(std::string(name) + " changed its uncapped tile at T=" +
                                         std::to_string(columns));
            }
        } else if (plan.J > NINFER_GGUF_SM86_MMQ_MAX_J) {
            throw std::runtime_error(std::string(name) + " exceeded the configured MMQ tile cap at T=" +
                                     std::to_string(columns));
        }
        std::cout << name << " T=" << columns << " J=" << plan.J << " I=" << plan.I << '\n';
    }
}

}

int main() {
    try {
        int device = 0;
        check(cudaGetDevice(&device), "get CUDA device");
        cudaDeviceProp properties{};
        check(cudaGetDeviceProperties(&properties, device), "get CUDA device properties");
        if (properties.major != 8 || properties.minor != 6) { return 77; }
        if (NINFER_GGUF_SM86_MMQ_MAX_J != 32 && NINFER_GGUF_SM86_MMQ_MAX_J != 64 &&
            NINFER_GGUF_SM86_MMQ_MAX_J != 128) {
            throw std::runtime_error("configured MMQ tile cap is outside the supported set");
        }
        constexpr std::array<int, 8> widths{17, 63, 64, 65, 127, 128, 129, 1024};
        const int rows = 128 * properties.multiProcessorCount;
        check_type(GgmlType::Q4_K, "Q4_K", rows, widths);
        check_type(GgmlType::Q6_K, "Q6_K", rows, widths);
        std::cout << "SM86 MMQ tile cap selector checks passed; cap="
                  << NINFER_GGUF_SM86_MMQ_MAX_J << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "SM86 MMQ tile cap selector: " << error.what() << '\n';
        return 1;
    }
}
