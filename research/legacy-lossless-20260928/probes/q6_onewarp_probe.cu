#include "artifact/q6_fields.h"
#include "core/weight_view.h"
#include "ops/linear/gguf/ggml_bridge_vec.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::ops::gguf::detail {
const DeviceFacts& device_facts() {
    static DeviceFacts facts = [] {
        cudaDeviceProp p{};
        check(cudaGetDeviceProperties(&p, 0), "device facts");
        return DeviceFacts{100 * p.major + 10 * p.minor, p.multiProcessorCount, p.sharedMemPerBlockOptin};
    }();
    return facts;
}
}

using namespace ninfer::ops::gguf::detail;
using namespace ninfer::ops::gguf::detail::vec;

struct Activation {
    int k = 0;
    int t = 0;
    std::vector<std::int8_t> qs;
    std::vector<half2> ds;
};

struct Planes {
    std::vector<std::uint8_t> raw_planar;
    std::uint64_t raw_row_bytes = 0;
};

void check_runtime(cudaError_t e, const char* name) {
    if (e != cudaSuccess) { throw std::runtime_error(std::string(name) + ": " + cudaGetErrorString(e)); }
}

std::uint32_t get_u32(const std::uint8_t* p) {
    std::uint32_t value;
    std::memcpy(&value, p, 4);
    return value;
}

void put_u32(std::uint8_t* p, std::uint32_t value) { std::memcpy(p, &value, 4); }

std::uint32_t bits_host(const std::uint8_t* data, std::uint64_t bit, unsigned width) {
    std::uint32_t value = 0;
    for (unsigned i = 0; i < width; ++i) { value |= ((data[(bit + i) >> 3] >> ((bit + i) & 7)) & 1u) << i; }
    return value;
}

std::vector<std::byte> read_bytes(const std::string& path, std::uint64_t offset, std::size_t bytes) {
    std::ifstream file(path, std::ios::binary);
    if (!file) { throw std::runtime_error("cannot open model"); }
    file.seekg(std::streamoff(offset));
    std::vector<std::byte> data(bytes);
    file.read(reinterpret_cast<char*>(data.data()), std::streamsize(bytes));
    if (std::size_t(file.gcount()) != bytes) { throw std::runtime_error("truncated model tensor"); }
    return data;
}

Activation load_activation(const std::string& path, int k, int t) {
    std::ifstream file(path, std::ios::binary);
    if (!file) { throw std::runtime_error("cannot open activation snapshot"); }
    std::array<std::uint32_t, 8> header{};
    file.read(reinterpret_cast<char*>(header.data()), sizeof(header));
    const std::uint32_t bytes = std::uint32_t(k * t + (k / 32) * t * 4);
    if (!file || header[0] != 0x31534156u || header[1] != 1 || header[2] != GGML_TYPE_Q6_K ||
        header[3] != std::uint32_t(k) || header[4] != std::uint32_t(t) || header[5] != 0 ||
        header[6] != bytes || header[7] != 0) { throw std::runtime_error("activation snapshot header mismatch"); }
    Activation a{k, t, std::vector<std::int8_t>(size_t(k) * t), std::vector<half2>(size_t(k / 32) * t)};
    file.read(reinterpret_cast<char*>(a.qs.data()), std::streamsize(a.qs.size()));
    file.read(reinterpret_cast<char*>(a.ds.data()), std::streamsize(a.ds.size() * sizeof(half2)));
    if (!file || file.peek() != std::char_traits<char>::eof()) { throw std::runtime_error("activation payload size mismatch"); }
    return a;
}

Planes make_raw_planar(const std::vector<std::byte>& raw, int n, int k) {
    const int blocks_per_row = k / 256;
    const int groups = blocks_per_row / 4;
    if (k % 1024 || raw.size() != size_t(n) * blocks_per_row * 210) { throw std::runtime_error("unsupported Q6 geometry"); }
    Planes p;
    p.raw_row_bytes = std::uint64_t(groups) * 840;
    p.raw_planar.resize(std::size_t(n) * p.raw_row_bytes);
    for (int row = 0; row < n; ++row) {
        for (int tile = 0; tile < groups; ++tile) {
            const auto src_tile = (size_t(row) * blocks_per_row + tile * 4) * 210;
            const auto dst_tile = size_t(row) * p.raw_row_bytes + size_t(tile) * 840;
            for (int slot = 0; slot < 4; ++slot) {
                const auto* block = raw.data() + src_tile + size_t(slot) * 210;
                auto* out = p.raw_planar.data() + dst_tile;
                for (int h = 0; h < 2; ++h) {
                    for (int pair = 0; pair < 2; ++pair) {
                        for (int word = 0; word < 8; ++word) {
                            const auto value = get_u32(reinterpret_cast<const std::uint8_t*>(block + 64 * h + 32 * pair + 4 * word));
                            put_u32(out + size_t(slot) * 128 + size_t(h * 2 + pair) * 32 + size_t(word) * 4, value);
                        }
                    }
                    for (int word = 0; word < 8; ++word) {
                        const auto value = get_u32(reinterpret_cast<const std::uint8_t*>(block + 128 + 32 * h + 4 * word));
                        put_u32(out + 512 + size_t(slot) * 64 + size_t(h) * 32 + size_t(word) * 4, value);
                    }
                }
                std::memcpy(out + 768 + size_t(slot) * 16, block + 192, 16);
                std::memcpy(out + 832 + size_t(slot) * 2, block + 208, 2);
            }
        }
    }
    return p;
}

__device__ __forceinline__ std::uint32_t load_word(const std::uint8_t* p) {
    return *reinterpret_cast<const std::uint32_t*>(p);
}

__device__ __forceinline__ std::uint32_t read_d_code(const std::uint8_t* p, std::uint32_t block, std::uint32_t width) {
    const std::uint32_t bit = block * width;
    const std::uint32_t byte = bit >> 3;
    const std::uint32_t shift = bit & 7u;
    std::uint32_t value = std::uint32_t(p[byte]) | (std::uint32_t(p[byte + 1]) << 8);
    if (shift + width > 16) { value |= std::uint32_t(p[byte + 2]) << 16; }
    return (value >> shift) & ((1u << width) - 1u);
}

template <int T, int RowsPerWarp, int Threads, bool CompressedD, bool LateActivation = false>
__launch_bounds__(Threads) __global__ void onewarp_kernel(VecArgs p, const std::uint8_t* d_bits,
    std::uint32_t d_width, std::uint32_t d_exp_base, std::uint32_t d_exp_bits,
    std::uint32_t d_sign_variable, std::uint32_t d_sign_value) {
    constexpr int warps = Threads / 32;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int slices = p.k / 32;
    const int blocks_per_row = p.k / 256;
    const int tile_bytes = CompressedD ? 832 : 840;
    const int row_bytes = blocks_per_row / 4 * tile_bytes;
    const int rows_per_block = warps * RowsPerWarp;
    for (int row0 = (blockIdx.x * warps + warp) * RowsPerWarp; row0 < p.rows;
         row0 += gridDim.x * rows_per_block) {
        float acc[RowsPerWarp][T];
#pragma unroll
        for (int r = 0; r < RowsPerWarp; ++r) {
#pragma unroll
            for (int j = 0; j < T; ++j) { acc[r][j] = 0.0f; }
        }
        for (int it = 0; lane + 32 * it < slices; ++it) {
            int a[T][8];
            float d[T], sum[T];
            if constexpr (!LateActivation) {
#pragma unroll
                for (int j = 0; j < T; ++j) {
                    const float2 ds = __half22float2(p.ds[int64_t(j) * slices + lane + 32 * it]);
                    d[j] = ds.x;
                    sum[j] = ds.y;
                    const int4* q = reinterpret_cast<const int4*>(p.qs + int64_t(j) * p.k + 32 * (lane + 32 * it));
                    const int4 v0 = q[0], v1 = q[1];
                    a[j][0] = v0.x; a[j][1] = v0.y; a[j][2] = v0.z; a[j][3] = v0.w;
                    a[j][4] = v1.x; a[j][5] = v1.y; a[j][6] = v1.z; a[j][7] = v1.w;
                }
            }
#pragma unroll
            for (int r = 0; r < RowsPerWarp; ++r) {
                const int row = row0 + r;
                if (row < p.rows) {
                    const int slot = lane / 8;
                    const int h = (lane % 8) / 4;
                    const int q = lane % 4;
                    const auto* tile = p.weight + int64_t(row) * row_bytes + int64_t(it) * tile_bytes;
                    Slice decoded;
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        const auto lo = load_word(tile + size_t(slot * 128 + (h * 2 + (q & 1)) * 32 + i * 4));
                        const auto hi = load_word(tile + size_t(512 + slot * 64 + h * 32 + i * 4));
                        const auto code = ((lo >> (4 * (q >> 1))) & 0x0F0F0F0Fu) |
                                          (((hi >> (2 * q)) << 4) & 0x30303030u);
                        decoded.w[i] = int(code ^ 0xE0E0E0E0u ^ ((code & 0x20202020u) * 6u));
                    }
                    const auto scale_offset = size_t(768 + slot * 16 + h * 8 + 2 * q);
                    const int8_t s0 = *reinterpret_cast<const int8_t*>(tile + scale_offset);
                    const int8_t s1 = *reinterpret_cast<const int8_t*>(tile + scale_offset + 1);
                    float block_d;
                    if constexpr (CompressedD) {
                        const auto block = std::uint32_t(row * blocks_per_row + it * 4 + slot);
                        const auto code = read_d_code(d_bits, block, d_width);
                        const auto exponent = d_exp_base + ((code >> 10) & ((1u << d_exp_bits) - 1u));
                        const auto sign = d_sign_variable ? ((code >> (10 + d_exp_bits)) & 1u) : d_sign_value;
                        block_d = __half2float(__ushort_as_half(std::uint16_t((code & 0x3ffu) | (exponent << 10) | (sign << 15))));
                    } else {
                        block_d = half_at(tile + 832 + slot * 2);
                    }
                    decoded.f[0] = block_d * float(s0);
                    decoded.f[1] = block_d * float(s1);
                    if constexpr (LateActivation) {
#pragma unroll
                        for (int j = 0; j < T; ++j) {
                            int lane_activation[8];
                            const float2 ds = __half22float2(p.ds[int64_t(j) * slices + lane + 32 * it]);
                            const int4* input = reinterpret_cast<const int4*>(p.qs + int64_t(j) * p.k + 32 * (lane + 32 * it));
                            const int4 v0 = input[0], v1 = input[1];
                            lane_activation[0] = v0.x; lane_activation[1] = v0.y; lane_activation[2] = v0.z; lane_activation[3] = v0.w;
                            lane_activation[4] = v1.x; lane_activation[5] = v1.y; lane_activation[6] = v1.z; lane_activation[7] = v1.w;
                            acc[r][j] += Decoder<GGML_TYPE_Q6_K>::dot(decoded, lane_activation, ds.x, ds.y);
                        }
                    } else {
#pragma unroll
                        for (int j = 0; j < T; ++j) { acc[r][j] += Decoder<GGML_TYPE_Q6_K>::dot(decoded, a[j], d[j], sum[j]); }
                    }
                }
            }
        }
#pragma unroll
        for (int r = 0; r < RowsPerWarp; ++r) {
#pragma unroll
            for (int j = 0; j < T; ++j) {
#pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1) { acc[r][j] += __shfl_xor_sync(0xffffffffu, acc[r][j], offset); }
                if (lane == 0 && row0 + r < p.rows) { vec_store(p.out, row0 + r, j, acc[r][j]); }
            }
        }
    }
}

struct Resource {
    int regs;
    std::size_t local;
    int shared;
    int blocks;
    int warps;
    float occupancy;
};

template <typename Kernel>
Resource resource(Kernel kernel, int threads) {
    cudaFuncAttributes attributes{};
    check_runtime(cudaFuncGetAttributes(&attributes, kernel), "kernel attributes");
    int active = 0;
    check_runtime(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active, kernel, threads, 0), "kernel occupancy");
    cudaDeviceProp p{};
    check_runtime(cudaGetDeviceProperties(&p, 0), "device properties");
    return {attributes.numRegs, attributes.localSizeBytes, int(attributes.sharedSizeBytes), active,
            active * (threads / 32), float(active * threads) / float(p.maxThreadsPerMultiProcessor)};
}

void print_resource(const Resource& r) {
    std::cout << "{\"registers_per_thread\":" << r.regs
              << ",\"local_bytes_per_thread\":" << r.local << ",\"static_shared_bytes\":" << r.shared
              << ",\"active_blocks_per_sm\":" << r.blocks << ",\"active_warps_per_sm\":" << r.warps
              << ",\"warp_occupancy\":" << r.occupancy << '}';
}

template <int T, int RowsPerWarp, int Threads, bool CompressedD, bool LateActivation = false>
void launch_candidate(VecArgs args, const std::uint8_t* d_bits, std::uint32_t width,
                      std::uint32_t exp_base, std::uint32_t exp_bits,
                      std::uint32_t sign_variable, std::uint32_t sign_value, cudaStream_t stream) {
    constexpr int warps = Threads / 32;
    int active = 0;
    check_runtime(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,
        onewarp_kernel<T, RowsPerWarp, Threads, CompressedD, LateActivation>, Threads, 0), "candidate occupancy");
    const int groups = (args.rows + warps * RowsPerWarp - 1) / (warps * RowsPerWarp);
    const int grid = std::min(groups, std::max(active, 1) * device_facts().sm_count);
    onewarp_kernel<T, RowsPerWarp, Threads, CompressedD, LateActivation><<<grid, Threads, 0, stream>>>(
        args, d_bits, width, exp_base, exp_bits, sign_variable, sign_value);
    check_runtime(cudaGetLastError(), "candidate launch");
}

template <int T, int RowsPerWarp, int Threads, bool CompressedD, bool LateActivation = false>
int candidate_grid(int rows) {
    constexpr int warps = Threads / 32;
    int active = 0;
    check_runtime(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,
        onewarp_kernel<T, RowsPerWarp, Threads, CompressedD, LateActivation>, Threads, 0), "candidate graph occupancy");
    const int groups = (rows + warps * RowsPerWarp - 1) / (warps * RowsPerWarp);
    return std::min(groups, std::max(active, 1) * device_facts().sm_count);
}

template <int T, int RowsPerWarp, int Threads, bool CompressedD, bool LateActivation = false>
void launch_candidate_grid(VecArgs args, const std::uint8_t* d_bits, std::uint32_t width,
                           std::uint32_t exp_base, std::uint32_t exp_bits,
                           std::uint32_t sign_variable, std::uint32_t sign_value,
                           int grid, cudaStream_t stream) {
    onewarp_kernel<T, RowsPerWarp, Threads, CompressedD, LateActivation><<<grid, Threads, 0, stream>>>(
        args, d_bits, width, exp_base, exp_bits, sign_variable, sign_value);
    check_runtime(cudaGetLastError(), "candidate graph launch");
}

template <int T, int RowsPerWarp, int Threads, bool CompressedD, bool LateActivation = false>
void measure_case(const char* mode, VecArgs args, const std::uint8_t* d_bits,
                  std::uint32_t width, std::uint32_t exp_base, std::uint32_t exp_bits,
                  std::uint32_t sign_variable, std::uint32_t sign_value, cudaStream_t stream,
                  const std::vector<float>& reference, std::vector<float>& output,
                  const std::vector<std::uint16_t>& bf_reference, std::vector<std::uint16_t>& bf_output,
                  __nv_bfloat16* bf_device_output) {
    auto launch = [&] { launch_candidate<T, RowsPerWarp, Threads, CompressedD, LateActivation>(args, d_bits, width,
        exp_base, exp_bits, sign_variable, sign_value, stream); };
    launch();
    check_runtime(cudaStreamSynchronize(stream), "candidate sync");
    check_runtime(cudaMemcpy(output.data(), args.out.f32, output.size() * sizeof(float), cudaMemcpyDeviceToHost), "candidate output copy");
    if (std::memcmp(reference.data(), output.data(), output.size() * sizeof(float)) != 0) {
        throw std::runtime_error(std::string(mode) + " differs from original vector kernel");
    }
    args.out.f32 = nullptr;
    args.out.bf16 = bf_device_output;
    launch();
    check_runtime(cudaStreamSynchronize(stream), "candidate BF16 sync");
    check_runtime(cudaMemcpy(bf_output.data(), bf_device_output, bf_output.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost), "candidate BF16 copy");
    if (bf_reference != bf_output) { throw std::runtime_error(std::string(mode) + " BF16 differs from original vector kernel"); }
    const auto r = resource(onewarp_kernel<T, RowsPerWarp, Threads, CompressedD, LateActivation>, Threads);
    std::cout << "{\"mode\":\"" << mode << "\",\"rows_per_warp\":" << RowsPerWarp
              << ",\"threads\":" << Threads << ",\"fp32_bit_exact\":true,\"bf16_bit_exact\":true,\"resources\":";
    print_resource(r);
    std::cout << "}\n";
}

template <int T>
void run(const std::string& model, std::uint64_t offset, int n, int k,
         const std::string& identity, const std::string& activation_file) {
    const auto row_bytes = std::uint64_t(k / 256) * 210;
    auto raw_bytes = read_bytes(model, offset, size_t(n) * row_bytes);
    auto planes = make_raw_planar(raw_bytes, n, k);
    const std::array<std::uint64_t, 2> shape{std::uint64_t(n), std::uint64_t(k)};
    auto original_geometry = ninfer::weight_geometry(ninfer::QType::GGUF_Q6_K,
        ninfer::QuantLayout::GgufBlocks, shape);
    auto encoded_geometry = ninfer::artifact::describe_q6_fields(original_geometry, raw_bytes);
    if (!encoded_geometry) { throw std::runtime_error("Q6 field encoding ineligible"); }
    auto encoded = ninfer::artifact::encode_q6_fields(*encoded_geometry, raw_bytes);
    auto activation = load_activation(activation_file, k, T);
    const auto code_bytes = size_t(encoded_geometry->code_bytes);
    const auto d_bytes = size_t(encoded_geometry->scale_bytes);
    const auto* header = reinterpret_cast<const std::uint32_t*>(encoded.data());
    const auto d_width = header[6];
    const auto flags = header[7];
    const auto exp_base = flags & 31u;
    const auto exp_bits = (flags >> 5) & 31u;
    const auto sign_variable = (flags >> 10) & 1u;
    const auto sign_value = (flags >> 11) & 1u;
    if (code_bytes != std::size_t(n) * std::size_t(k / 1024) * 832) { throw std::runtime_error("encoded plane geometry mismatch"); }
    std::vector<std::uint8_t> d_padded(d_bytes + 4, 0);
    std::memcpy(d_padded.data(), encoded.data() + encoded_geometry->scale_offset, d_bytes);
    std::uint8_t* device_raw = nullptr;
    std::uint8_t* device_planar = nullptr;
    std::uint8_t* device_compressed = nullptr;
    std::uint8_t* device_d = nullptr;
    std::int8_t* device_qs = nullptr;
    half2* device_ds = nullptr;
    float* device_output = nullptr;
    __nv_bfloat16* device_bf16_output = nullptr;
    auto alloc = [](auto** ptr, size_t bytes) { check_runtime(cudaMalloc(reinterpret_cast<void**>(ptr), bytes), "device allocation"); };
    alloc(&device_raw, raw_bytes.size());
    check_runtime(cudaMemcpy(device_raw, raw_bytes.data(), raw_bytes.size(), cudaMemcpyHostToDevice), "raw upload");
    alloc(&device_planar, planes.raw_planar.size());
    check_runtime(cudaMemcpy(device_planar, planes.raw_planar.data(), planes.raw_planar.size(), cudaMemcpyHostToDevice), "planar upload");
    alloc(&device_compressed, code_bytes);
    check_runtime(cudaMemcpy(device_compressed, encoded.data() + encoded_geometry->code_offset, code_bytes, cudaMemcpyHostToDevice), "compressed upload");
    alloc(&device_d, d_padded.size());
    check_runtime(cudaMemcpy(device_d, d_padded.data(), d_padded.size(), cudaMemcpyHostToDevice), "d bits upload");
    alloc(&device_qs, activation.qs.size());
    check_runtime(cudaMemcpy(device_qs, activation.qs.data(), activation.qs.size(), cudaMemcpyHostToDevice), "activation qs upload");
    alloc(&device_ds, activation.ds.size() * sizeof(half2));
    check_runtime(cudaMemcpy(device_ds, activation.ds.data(), activation.ds.size() * sizeof(half2), cudaMemcpyHostToDevice), "activation ds upload");
    const auto output_count = size_t(n) * T;
    alloc(&device_output, output_count * sizeof(float));
    alloc(&device_bf16_output, output_count * sizeof(__nv_bfloat16));
    cudaStream_t stream;
    check_runtime(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream creation");
    VecArgs args{device_raw, std::int64_t(row_bytes), n, k, device_qs, device_ds,
                 {nullptr, device_output, n, ninfer::ops::gguf::Epilogue::Store, nullptr, 0}};
    vec::launch<GGML_TYPE_Q6_K, T, false, 0>(args, stream);
    check_runtime(cudaStreamSynchronize(stream), "baseline sync");
    std::vector<float> reference(output_count), output(output_count);
    check_runtime(cudaMemcpy(reference.data(), device_output, output_count * sizeof(float), cudaMemcpyDeviceToHost), "baseline copy");
    std::vector<std::uint16_t> bf_reference(output_count), bf_output(output_count);
    args.out.f32 = nullptr;
    args.out.bf16 = device_bf16_output;
    vec::launch<GGML_TYPE_Q6_K, T, false, 0>(args, stream);
    check_runtime(cudaStreamSynchronize(stream), "baseline BF16 sync");
    check_runtime(cudaMemcpy(bf_reference.data(), device_bf16_output, output_count * sizeof(std::uint16_t), cudaMemcpyDeviceToHost), "baseline BF16 copy");
    args.out.f32 = device_output;
    args.out.bf16 = nullptr;
    args.weight = device_planar;
    measure_case<T, 1, 128, false>("raw_planar_1row_128", args, nullptr, 0, 0, 0, 0, 0, stream, reference, output, bf_reference, bf_output, device_bf16_output);
    measure_case<T, 1, 256, false>("raw_planar_1row_256", args, nullptr, 0, 0, 0, 0, 0, stream, reference, output, bf_reference, bf_output, device_bf16_output);
    measure_case<T, 2, 128, false>("raw_planar_2row_128", args, nullptr, 0, 0, 0, 0, 0, stream, reference, output, bf_reference, bf_output, device_bf16_output);
    measure_case<T, 2, 256, false>("raw_planar_2row_256", args, nullptr, 0, 0, 0, 0, 0, stream, reference, output, bf_reference, bf_output, device_bf16_output);
    args.weight = device_compressed;
    measure_case<T, 1, 128, true>("compressed_1row_128", args, device_d, d_width, exp_base, exp_bits, sign_variable, sign_value, stream, reference, output, bf_reference, bf_output, device_bf16_output);
    measure_case<T, 1, 256, true>("compressed_1row_256", args, device_d, d_width, exp_base, exp_bits, sign_variable, sign_value, stream, reference, output, bf_reference, bf_output, device_bf16_output);
    measure_case<T, 2, 128, true>("compressed_2row_128", args, device_d, d_width, exp_base, exp_bits, sign_variable, sign_value, stream, reference, output, bf_reference, bf_output, device_bf16_output);
    measure_case<T, 2, 256, true>("compressed_2row_256", args, device_d, d_width, exp_base, exp_bits, sign_variable, sign_value, stream, reference, output, bf_reference, bf_output, device_bf16_output);
    measure_case<T, 1, 128, true, true>("compressed_1row_lateload_128", args, device_d, d_width, exp_base, exp_bits, sign_variable, sign_value, stream, reference, output, bf_reference, bf_output, device_bf16_output);
    measure_case<T, 1, 256, true, true>("compressed_1row_lateload_256", args, device_d, d_width, exp_base, exp_bits, sign_variable, sign_value, stream, reference, output, bf_reference, bf_output, device_bf16_output);
    const std::array<const char*, 11> mode_names{"original_vector", "raw_planar_1row_128", "raw_planar_1row_256",
        "raw_planar_2row_128", "raw_planar_2row_256", "compressed_1row_128", "compressed_1row_256",
        "compressed_2row_128", "compressed_2row_256", "compressed_1row_lateload_128", "compressed_1row_lateload_256"};
    const std::array<int, 10> grids{
        candidate_grid<T, 1, 128, false>(n), candidate_grid<T, 1, 256, false>(n),
        candidate_grid<T, 2, 128, false>(n), candidate_grid<T, 2, 256, false>(n),
        candidate_grid<T, 1, 128, true>(n), candidate_grid<T, 1, 256, true>(n),
        candidate_grid<T, 2, 128, true>(n), candidate_grid<T, 2, 256, true>(n),
        candidate_grid<T, 1, 128, true, true>(n), candidate_grid<T, 1, 256, true, true>(n)};
    const int rotations = raw_bytes.size() <= 256ull * 1024ull * 1024ull ? 3 : 1;
    std::vector<std::uint8_t*> raw_copies(rotations), planar_copies(rotations), compressed_copies(rotations), d_copies(rotations);
    raw_copies[0] = device_raw;
    planar_copies[0] = device_planar;
    compressed_copies[0] = device_compressed;
    d_copies[0] = device_d;
    for (int rotation = 1; rotation < rotations; ++rotation) {
        alloc(&raw_copies[rotation], raw_bytes.size());
        check_runtime(cudaMemcpy(raw_copies[rotation], raw_bytes.data(), raw_bytes.size(), cudaMemcpyHostToDevice), "rotate raw upload");
        alloc(&planar_copies[rotation], planes.raw_planar.size());
        check_runtime(cudaMemcpy(planar_copies[rotation], planes.raw_planar.data(), planes.raw_planar.size(), cudaMemcpyHostToDevice), "rotate planar upload");
        alloc(&compressed_copies[rotation], code_bytes);
        check_runtime(cudaMemcpy(compressed_copies[rotation], encoded.data() + encoded_geometry->code_offset, code_bytes, cudaMemcpyHostToDevice), "rotate compressed upload");
        alloc(&d_copies[rotation], d_padded.size());
        check_runtime(cudaMemcpy(d_copies[rotation], d_padded.data(), d_padded.size(), cudaMemcpyHostToDevice), "rotate d upload");
    }
    auto launch_mode = [&](int mode, int rotation) {
        VecArgs graph_args = args;
        graph_args.out = {nullptr, device_output, n, ninfer::ops::gguf::Epilogue::Store, nullptr, 0};
        if (mode == 0) {
            graph_args.weight = raw_copies[rotation];
            graph_args.row_bytes = std::int64_t(row_bytes);
            vec::launch<GGML_TYPE_Q6_K, T, false, 0>(graph_args, stream);
        } else if (mode <= 4) {
            graph_args.weight = planar_copies[rotation];
            graph_args.row_bytes = std::int64_t(planes.raw_row_bytes);
            if (mode == 1) { launch_candidate_grid<T, 1, 128, false>(graph_args, nullptr, 0, 0, 0, 0, 0, grids[mode - 1], stream); }
            else if (mode == 2) { launch_candidate_grid<T, 1, 256, false>(graph_args, nullptr, 0, 0, 0, 0, 0, grids[mode - 1], stream); }
            else if (mode == 3) { launch_candidate_grid<T, 2, 128, false>(graph_args, nullptr, 0, 0, 0, 0, 0, grids[mode - 1], stream); }
            else { launch_candidate_grid<T, 2, 256, false>(graph_args, nullptr, 0, 0, 0, 0, 0, grids[mode - 1], stream); }
        } else {
            graph_args.weight = compressed_copies[rotation];
            graph_args.row_bytes = std::int64_t(encoded_geometry->code_bytes_per_row);
            const auto* dplane = d_copies[rotation];
            if (mode == 5) { launch_candidate_grid<T, 1, 128, true>(graph_args, dplane, d_width, exp_base, exp_bits, sign_variable, sign_value, grids[mode - 1], stream); }
            else if (mode == 6) { launch_candidate_grid<T, 1, 256, true>(graph_args, dplane, d_width, exp_base, exp_bits, sign_variable, sign_value, grids[mode - 1], stream); }
            else if (mode == 7) { launch_candidate_grid<T, 2, 128, true>(graph_args, dplane, d_width, exp_base, exp_bits, sign_variable, sign_value, grids[mode - 1], stream); }
            else if (mode == 8) { launch_candidate_grid<T, 2, 256, true>(graph_args, dplane, d_width, exp_base, exp_bits, sign_variable, sign_value, grids[mode - 1], stream); }
            else if (mode == 9) { launch_candidate_grid<T, 1, 128, true, true>(graph_args, dplane, d_width, exp_base, exp_bits, sign_variable, sign_value, grids[mode - 1], stream); }
            else { launch_candidate_grid<T, 1, 256, true, true>(graph_args, dplane, d_width, exp_base, exp_bits, sign_variable, sign_value, grids[mode - 1], stream); }
        }
    };
    std::vector<std::vector<cudaGraphExec_t>> graphs(11, std::vector<cudaGraphExec_t>(rotations));
    for (int mode = 0; mode < 11; ++mode) {
        for (int rotation = 0; rotation < rotations; ++rotation) {
            cudaGraph_t graph;
            check_runtime(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "graph capture begin");
            launch_mode(mode, rotation);
            check_runtime(cudaStreamEndCapture(stream, &graph), "graph capture end");
            check_runtime(cudaGraphInstantiate(&graphs[mode][rotation], graph, 0), "graph instantiate");
            check_runtime(cudaGraphDestroy(graph), "graph destroy");
        }
    }
    for (int mode = 0; mode < 11; ++mode) {
        for (int replay = 0; replay < 10; ++replay) {
            check_runtime(cudaGraphLaunch(graphs[mode][replay % rotations], stream), "graph warmup");
        }
    }
    check_runtime(cudaStreamSynchronize(stream), "graph warmup sync");
    cudaEvent_t start, stop;
    check_runtime(cudaEventCreate(&start), "start event");
    check_runtime(cudaEventCreate(&stop), "stop event");
    std::vector<std::vector<float>> timings(11, std::vector<float>(5));
    for (int sample = 0; sample < 5; ++sample) {
        for (int index = 0; index < 11; ++index) {
            const int mode = sample % 2 == 0 ? index : 10 - index;
            check_runtime(cudaEventRecord(start, stream), "start record");
            for (int replay = 0; replay < 10; ++replay) {
                check_runtime(cudaGraphLaunch(graphs[mode][replay % rotations], stream), "timed graph replay");
            }
            check_runtime(cudaEventRecord(stop, stream), "stop record");
            check_runtime(cudaEventSynchronize(stop), "stop sync");
            float elapsed = 0.0f;
            check_runtime(cudaEventElapsedTime(&elapsed, start, stop), "elapsed time");
            timings[mode][sample] = elapsed / 10.0f;
        }
    }
    check_runtime(cudaEventDestroy(stop), "stop event destroy");
    check_runtime(cudaEventDestroy(start), "start event destroy");
    for (auto& mode_graphs : graphs) {
        for (const auto graph : mode_graphs) { check_runtime(cudaGraphExecDestroy(graph), "graph exec destroy"); }
    }
    cudaDeviceProp device_properties{};
    check_runtime(cudaGetDeviceProperties(&device_properties, 0), "benchmark device properties");
    std::cout << "{\"summary\":true,\"matrix\":\"" << identity << "\",\"N\":" << n << ",\"K\":" << k
              << ",\"T\":" << T << ",\"raw_bytes\":" << raw_bytes.size()
              << ",\"raw_planar_bytes\":" << planes.raw_planar.size()
              << ",\"compressed_code_bytes\":" << code_bytes << ",\"compressed_d_bytes\":" << d_bytes
              << ",\"compressed_format_header_bytes\":32,\"compressed_format_bytes_including_header\":" << encoded.size()
              << ",\"rotating_copies\":" << rotations << ",\"device_l2_bytes\":" << device_properties.l2CacheSize
              << ",\"working_set_bytes\":{\"raw\":" << raw_bytes.size() * rotations
              << ",\"raw_planar\":" << planes.raw_planar.size() * rotations
              << ",\"compressed_code_and_d\":" << (code_bytes + d_bytes) * rotations << '}'
              << ",\"d_width\":" << d_width << ",\"exponent_bits\":" << exp_bits
              << ",\"sign_variable\":" << sign_variable << ",\"baseline_registers\":"
              << resource(kernel<GGML_TYPE_Q6_K, T, false, 0>, kVecWarps * 32).regs << ",\"samples_ms\":{";
    for (int mode = 0; mode < 11; ++mode) {
        if (mode) { std::cout << ','; }
        std::cout << '\"' << mode_names[mode] << "\":[";
        for (int sample = 0; sample < 5; ++sample) {
            if (sample) { std::cout << ','; }
            std::cout << timings[mode][sample];
        }
        std::cout << ']';
    }
    std::cout << "},\"timing_method\":\"CUDA graphs, 10 sequential replays/sample, 5 samples/mode, alternating mode order, rotating weight copies\"}\n";
    for (int rotation = 1; rotation < rotations; ++rotation) {
        check_runtime(cudaFree(d_copies[rotation]), "free rotated d bits");
        check_runtime(cudaFree(compressed_copies[rotation]), "free rotated compressed payload");
        check_runtime(cudaFree(planar_copies[rotation]), "free rotated planar payload");
        check_runtime(cudaFree(raw_copies[rotation]), "free rotated raw payload");
    }
    check_runtime(cudaStreamDestroy(stream), "destroy stream");
    check_runtime(cudaFree(device_output), "free output");
    check_runtime(cudaFree(device_bf16_output), "free BF16 output");
    check_runtime(cudaFree(device_ds), "free activation ds");
    check_runtime(cudaFree(device_qs), "free activation qs");
    check_runtime(cudaFree(device_d), "free d bits");
    check_runtime(cudaFree(device_compressed), "free compressed plane");
    check_runtime(cudaFree(device_planar), "free planar");
    check_runtime(cudaFree(device_raw), "free raw");
}

int main(int argc, char** argv) {
    if (argc != 7) {
        std::cerr << "model offset N K tensor activation\n";
        return 2;
    }
    try {
        run<4>(argv[1], std::stoull(argv[2]), std::stoi(argv[3]), std::stoi(argv[4]), argv[5], argv[6]);
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
