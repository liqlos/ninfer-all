#define main q4_fastfields_unused_main
#include "q4_fastfields_probe.cu"
#undef main

__device__ __forceinline__ uint32_t q4_to_s8(uint32_t x, uint32_t shift) {
    return (x >> shift) & 0x0F0F0F0Fu;
}

__device__ __forceinline__ void mma_q4_dot(uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
                                           uint32_t b0, uint32_t b1, int& c0, int& c1, int& c2, int& c3) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t load_q4_fragment_word(const uint8_t* qs, uint32_t full_row,
                                                          uint32_t block_in_row, uint32_t group,
                                                          uint32_t k_offset) {
    const uint32_t block = full_row * 20u + block_in_row;
    const uint32_t* words = reinterpret_cast<const uint32_t*>(qs + block * 128u + (group >> 1) * 32u + k_offset);
    return q4_to_s8(*words, 4u * (group & 1u));
}

__device__ __forceinline__ int half_bits_to_int(uint16_t value) {
    return int(value);
}

__global__ void reconstruct_fixed_mma_kernel(int rows, const uint8_t* dm, const uint8_t* qs,
                                              const uint8_t* metadata, const uint8_t* d_code,
                                              const uint8_t* dmin_code, uint8_t* raw) {
    const uint32_t block = uint32_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint32_t count = uint32_t(rows) * 20u;
    if (block >= count) { return; }
    const uint32_t at = block * 144u;
    const uint16_t d = decode_half_field_fixed_device<3, 0>(d_code, block);
    const uint16_t dmin = decode_half_field_fixed_device<3, 2>(dmin_code, block);
    raw[at] = uint8_t(d); raw[at + 1] = uint8_t(d >> 8);
    raw[at + 2] = uint8_t(dmin); raw[at + 3] = uint8_t(dmin >> 8);
    for (uint32_t i = 0; i < 12u; ++i) { raw[at + 4u + i] = metadata[block * 12u + i]; }
    for (uint32_t i = 0; i < 128u; ++i) { raw[at + 16u + i] = qs[block * 128u + i]; }
}

__device__ __forceinline__ uint8_t transposed_q4_code_device(const uint8_t* qs, uint32_t row,
                                                             uint32_t slice, uint32_t pos) {
    const uint32_t lane = (row & 7u) * 4u + ((pos & 15u) >> 2);
    const uint32_t byte_in_lane = (pos >> 4) * 4u + (pos & 3u);
    const uint64_t at = (uint64_t(row / 16u) * 160u + slice) * 256u + uint64_t(lane) * 8u + byte_in_lane;
    return uint8_t((qs[at] >> ((row & 8u) ? 4u : 0u)) & 0x0Fu);
}

__global__ void restore_transposed_qs_kernel(uint32_t rows, const uint8_t* transposed,
                                              uint8_t* qs) {
    const uint64_t at = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const uint64_t count = uint64_t(rows) * 20u * 128u;
    if (at >= count) { return; }
    const uint32_t block = uint32_t(at / 128u);
    const uint32_t row = block / 20u;
    const uint32_t block_in_row = block % 20u;
    const uint32_t within = uint32_t(at % 128u);
    const uint32_t pair = within / 32u;
    const uint32_t pos = within % 32u;
    const uint32_t slice = block_in_row * 8u;
    const uint8_t low = transposed_q4_code_device(transposed, row, slice + pair * 2u, pos);
    const uint8_t high = transposed_q4_code_device(transposed, row, slice + pair * 2u + 1u, pos);
    qs[at] = uint8_t(low | (high << 4));
}

static std::vector<uint8_t> transpose_qs_host(const std::vector<uint8_t>& qs, uint32_t rows) {
    std::vector<uint8_t> out(size_t(rows / 16u) * 160u * 256u);
    for (uint32_t tile = 0; tile < rows / 16u; ++tile) {
        for (uint32_t slice = 0; slice < 160u; ++slice) {
            const uint32_t block_in_row = slice / 8u;
            const uint32_t group = slice & 7u;
            for (uint32_t lane = 0; lane < 32u; ++lane) {
                const uint32_t row0 = tile * 16u + (lane >> 2);
                const uint32_t row1 = row0 + 8u;
                const uint32_t pos0 = (lane & 3u) * 4u;
                const size_t base = (size_t(tile) * 160u + slice) * 256u + size_t(lane) * 8u;
                const size_t block0 = size_t(row0) * 20u + block_in_row;
                const size_t block1 = size_t(row1) * 20u + block_in_row;
                for (uint32_t j = 0; j < 4u; ++j) {
                    const uint32_t pos = pos0 + j;
                    const size_t at0 = block0 * 128u + size_t(group >> 1) * 32u + pos;
                    const size_t at1 = block1 * 128u + size_t(group >> 1) * 32u + pos;
                    const uint8_t code0 = uint8_t((qs[at0] >> (4u * (group & 1u))) & 15u);
                    const uint8_t code1 = uint8_t((qs[at1] >> (4u * (group & 1u))) & 15u);
                    const size_t at2 = block0 * 128u + size_t(group >> 1) * 32u + pos + 16u;
                    const size_t at3 = block1 * 128u + size_t(group >> 1) * 32u + pos + 16u;
                    const uint8_t code2 = uint8_t((qs[at2] >> (4u * (group & 1u))) & 15u);
                    const uint8_t code3 = uint8_t((qs[at3] >> (4u * (group & 1u))) & 15u);
                    out[base + j] = uint8_t(code0 | (code1 << 4));
                    out[base + 4u + j] = uint8_t(code2 | (code3 << 4));
                }
            }
        }
    }
    return out;
}

static std::vector<uint8_t> restore_transposed_qs_host(const std::vector<uint8_t>& transposed, uint32_t rows) {
    std::vector<uint8_t> qs(size_t(rows) * 20u * 128u);
    for (uint32_t block = 0; block < rows * 20u; ++block) {
        const uint32_t row = block / 20u;
        const uint32_t block_in_row = block % 20u;
        for (uint32_t within = 0; within < 128u; ++within) {
            const uint32_t pair = within / 32u;
            const uint32_t pos = within & 31u;
            const uint32_t slice = block_in_row * 8u + pair * 2u;
            const uint32_t lane = (row & 7u) * 4u + ((pos & 15u) >> 2);
            const uint32_t byte_in_lane = (pos >> 4) * 4u + (pos & 3u);
            const size_t at0 = (size_t(row / 16u) * 160u + slice) * 256u + size_t(lane) * 8u + byte_in_lane;
            const size_t at1 = (size_t(row / 16u) * 160u + slice + 1u) * 256u + size_t(lane) * 8u + byte_in_lane;
            const uint8_t lo = uint8_t((transposed[at0] >> ((row & 8u) ? 4u : 0u)) & 15u);
            const uint8_t hi = uint8_t((transposed[at1] >> ((row & 8u) ? 4u : 0u)) & 15u);
            qs[size_t(block) * 128u + within] = uint8_t(lo | (hi << 4));
        }
    }
    return qs;
}

template <unsigned DExpBase, unsigned DMinExpBase, bool PadBuckets = false>
__launch_bounds__(256) __global__ void q4_mma_fused_kernel(ScaleArgs p) {
    __shared__ float buckets[2][16][4][PadBuckets ? 33 : 32];
    const uint32_t tid = threadIdx.x;
    const uint32_t lane_group = (tid & 31u) >> 2;
    const uint32_t lane_col = tid & 3u;
    const uint32_t warp = tid >> 5;
    const uint32_t out_row = blockIdx.x * 16u;
    const uint8_t* qplane = p.qs;
    const uint8_t* metadata = p.raw_metadata;
    const uint8_t* dplane = p.d_code;
    const uint8_t* dminplane = p.dmin_code;
    const uint8_t* xplane = reinterpret_cast<const uint8_t*>(p.vec.qs);
    const uint32_t base_u = warp * 4u;
    for (uint32_t uv = 0; uv < 4u; ++uv) {
        const uint32_t u = base_u + uv;
        const uint32_t group = u & 7u;
        const uint32_t qshift = 4u * (group & 1u);
        float gate0 = 0.0f, gate1 = 0.0f, gate2 = 0.0f, gate3 = 0.0f;
        float up0 = 0.0f, up1 = 0.0f, up2 = 0.0f, up3 = 0.0f;
#pragma unroll 1
        for (uint32_t it = 0; it < 5u; ++it) {
            const uint32_t block_in_row = 4u * it + (u >> 3);
            const uint32_t k0 = lane_col * 4u;
            const uint32_t k1 = k0 + 16u;
            const uint32_t row0 = out_row + lane_group;
            const uint32_t row1 = row0 + 8u;
            const uint32_t gate_row0 = row0;
            const uint32_t gate_row1 = row1;
            const uint32_t up_row0 = p.vec.rows + row0;
            const uint32_t up_row1 = p.vec.rows + row1;
            const uint32_t ga0 = load_q4_fragment_word(qplane, gate_row0, block_in_row, group, k0);
            const uint32_t ga1 = load_q4_fragment_word(qplane, gate_row1, block_in_row, group, k0);
            const uint32_t ga2 = load_q4_fragment_word(qplane, gate_row0, block_in_row, group, k1);
            const uint32_t ga3 = load_q4_fragment_word(qplane, gate_row1, block_in_row, group, k1);
            const uint32_t ua0 = load_q4_fragment_word(qplane, up_row0, block_in_row, group, k0);
            const uint32_t ua1 = load_q4_fragment_word(qplane, up_row1, block_in_row, group, k0);
            const uint32_t ua2 = load_q4_fragment_word(qplane, up_row0, block_in_row, group, k1);
            const uint32_t ua3 = load_q4_fragment_word(qplane, up_row1, block_in_row, group, k1);
            uint32_t b0 = 0u, b1 = 0u;
            if (lane_group < 4u) {
                const uint32_t col = lane_group;
                const uint32_t base = col * 5120u + 32u * (u + 32u * it);
                b0 = *reinterpret_cast<const uint32_t*>(xplane + base + k0);
                b1 = *reinterpret_cast<const uint32_t*>(xplane + base + k1);
            }
            int gc0 = 0, gc1 = 0, gc2 = 0, gc3 = 0;
            int uc0 = 0, uc1 = 0, uc2 = 0, uc3 = 0;
            mma_q4_dot(ga0, ga1, ga2, ga3, b0, b1, gc0, gc1, gc2, gc3);
            mma_q4_dot(ua0, ua1, ua2, ua3, b0, b1, uc0, uc1, uc2, uc3);
            const uint32_t output_cols[2] = {2u * lane_col, 2u * lane_col + 1u};
            const int gate_values[4] = {gc0, gc1, gc2, gc3};
            const int up_values[4] = {uc0, uc1, uc2, uc3};
            const uint32_t block0 = row0 * 20u + block_in_row;
            const uint32_t block1 = row1 * 20u + block_in_row;
            int sc0, mn0, sc1, mn1;
            scale_min_k4(metadata + block0 * 12u, group, sc0, mn0);
            scale_min_k4(metadata + block1 * 12u, group, sc1, mn1);
            const uint16_t d0 = decode_half_field_fixed_device<13u - 10u, DExpBase>(dplane, block0);
            const uint16_t dmin0 = decode_half_field_fixed_device<13u - 10u, DMinExpBase>(dminplane, block0);
            const uint16_t d1 = decode_half_field_fixed_device<13u - 10u, DExpBase>(dplane, block1);
            const uint16_t dmin1 = decode_half_field_fixed_device<13u - 10u, DMinExpBase>(dminplane, block1);
            const float f0 = __half2float(__ushort_as_half(d0)) * float(sc0);
            const float fmin0 = __half2float(__ushort_as_half(dmin0)) * float(mn0);
            const float f1 = __half2float(__ushort_as_half(d1)) * float(sc1);
            const float fmin1 = __half2float(__ushort_as_half(dmin1)) * float(mn1);
            const uint32_t up_block0 = (p.vec.rows + row0) * 20u + block_in_row;
            const uint32_t up_block1 = (p.vec.rows + row1) * 20u + block_in_row;
            int up_sc0, up_mn0, up_sc1, up_mn1;
            scale_min_k4(metadata + up_block0 * 12u, group, up_sc0, up_mn0);
            scale_min_k4(metadata + up_block1 * 12u, group, up_sc1, up_mn1);
            const uint16_t up_d0 = decode_half_field_fixed_device<13u - 10u, DExpBase>(dplane, up_block0);
            const uint16_t up_dmin0 = decode_half_field_fixed_device<13u - 10u, DMinExpBase>(dminplane, up_block0);
            const uint16_t up_d1 = decode_half_field_fixed_device<13u - 10u, DExpBase>(dplane, up_block1);
            const uint16_t up_dmin1 = decode_half_field_fixed_device<13u - 10u, DMinExpBase>(dminplane, up_block1);
            const float uf0 = __half2float(__ushort_as_half(up_d0)) * float(up_sc0);
            const float ufmin0 = __half2float(__ushort_as_half(up_dmin0)) * float(up_mn0);
            const float uf1 = __half2float(__ushort_as_half(up_d1)) * float(up_sc1);
            const float ufmin1 = __half2float(__ushort_as_half(up_dmin1)) * float(up_mn1);
            for (uint32_t r = 0; r < 2u; ++r) {
                const float f = r == 0u ? f0 : f1;
                const float fmin = r == 0u ? fmin0 : fmin1;
                const float uf = r == 0u ? uf0 : uf1;
                const float ufmin = r == 0u ? ufmin0 : ufmin1;
                for (uint32_t c = 0; c < 2u; ++c) {
                    const uint32_t col = output_cols[c];
                    if (col < 4u) {
                        const uint32_t slice = u + 32u * it;
                        const float2 ds = __half22float2(p.vec.ds[uint64_t(col) * 160u + slice]);
                        const uint32_t vi = r * 2u + c;
                        const float gv = f * ds.x * float(gate_values[vi]) - fmin * ds.y;
                        const float uvv = uf * ds.x * float(up_values[vi]) - ufmin * ds.y;
                        if (r == 0u && c == 0u) { gate0 += gv; up0 += uvv; }
                        if (r == 0u && c == 1u) { gate1 += gv; up1 += uvv; }
                        if (r == 1u && c == 0u) { gate2 += gv; up2 += uvv; }
                        if (r == 1u && c == 1u) { gate3 += gv; up3 += uvv; }
                    }
                }
            }
        }
        if (lane_col < 2u) {
            const uint32_t col0 = 2u * lane_col;
            const uint32_t col1 = col0 + 1u;
            buckets[0][lane_group][col0][u] = gate0;
            buckets[1][lane_group][col0][u] = up0;
            buckets[0][lane_group][col1][u] = gate1;
            buckets[1][lane_group][col1][u] = up1;
            buckets[0][lane_group + 8u][col0][u] = gate2;
            buckets[1][lane_group + 8u][col0][u] = up2;
            buckets[0][lane_group + 8u][col1][u] = gate3;
            buckets[1][lane_group + 8u][col1][u] = up3;
        }
    }
    __syncthreads();
    float gate = 0.0f, up = 0.0f;
    for (uint32_t item = warp; item < 64u; item += 8u) {
        const uint32_t row = item / 4u;
        const uint32_t col = item & 3u;
        gate = buckets[0][row][col][tid & 31u];
        up = buckets[1][row][col][tid & 31u];
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            gate += __shfl_xor_sync(0xffffffffu, gate, offset);
            up += __shfl_xor_sync(0xffffffffu, up, offset);
        }
        if ((tid & 31u) == 0u) {
            vec_store(p.vec.out, int(out_row + row), int(col), vec_silu(gate) * up);
        }
    }
}

template <unsigned DExpBase, unsigned DMinExpBase>
__launch_bounds__(1024) __global__ void q4_mma_transposed_kernel(ScaleArgs p) {
    __shared__ float buckets[2][16][4][33];
    const uint32_t tid = threadIdx.x;
    const uint32_t lane_group = (tid & 31u) >> 2;
    const uint32_t lane_col = tid & 3u;
    const uint32_t warp = tid >> 5;
    const uint32_t u = warp;
    const uint32_t out_row = blockIdx.x * 16u;
    const uint8_t* qplane = p.qs;
    const uint8_t* metadata = p.raw_metadata;
    const uint8_t* dplane = p.d_code;
    const uint8_t* dminplane = p.dmin_code;
    const uint8_t* xplane = reinterpret_cast<const uint8_t*>(p.vec.qs);

                        const uint32_t group = u & 7u;
            const uint32_t qshift = 4u * (group & 1u);
            float gate0 = 0.0f, gate1 = 0.0f, gate2 = 0.0f, gate3 = 0.0f;
            float up0 = 0.0f, up1 = 0.0f, up2 = 0.0f, up3 = 0.0f;
    #pragma unroll 1
            for (uint32_t it = 0; it < 5u; ++it) {
                const uint32_t block_in_row = 4u * it + (u >> 3);
                const uint32_t k0 = lane_col * 4u;
                const uint32_t k1 = k0 + 16u;
                const uint32_t row0 = out_row + lane_group;
                const uint32_t row1 = row0 + 8u;
                const uint32_t gate_row0 = row0;
                const uint32_t gate_row1 = row1;
                const uint32_t up_row0 = p.vec.rows + row0;
                const uint32_t up_row1 = p.vec.rows + row1;
                const uint32_t qslice = u + 32u * it;
            const uint32_t qlane = tid & 31u;
            const uint64_t gate_at = (uint64_t(row0 / 16u) * 160u + qslice) * 256u + uint64_t(qlane) * 8u;
            const uint64_t up_at = (uint64_t(up_row0 / 16u) * 160u + qslice) * 256u + uint64_t(qlane) * 8u;
            const uint64_t gate_packed = *reinterpret_cast<const uint64_t*>(qplane + gate_at);
            const uint64_t up_packed = *reinterpret_cast<const uint64_t*>(qplane + up_at);
            const uint32_t gp0 = uint32_t(gate_packed);
            const uint32_t gp1 = uint32_t(gate_packed >> 32);
            const uint32_t up0p = uint32_t(up_packed);
            const uint32_t up1p = uint32_t(up_packed >> 32);
            const uint32_t ga0 = gp0 & 0x0F0F0F0Fu;
            const uint32_t ga1 = (gp0 >> 4) & 0x0F0F0F0Fu;
            const uint32_t ga2 = gp1 & 0x0F0F0F0Fu;
            const uint32_t ga3 = (gp1 >> 4) & 0x0F0F0F0Fu;
            const uint32_t ua0 = up0p & 0x0F0F0F0Fu;
            const uint32_t ua1 = (up0p >> 4) & 0x0F0F0F0Fu;
            const uint32_t ua2 = up1p & 0x0F0F0F0Fu;
            const uint32_t ua3 = (up1p >> 4) & 0x0F0F0F0Fu;
                uint32_t b0 = 0u, b1 = 0u;
                if (lane_group < 4u) {
                    const uint32_t col = lane_group;
                    const uint32_t base = col * 5120u + 32u * (u + 32u * it);
                    b0 = *reinterpret_cast<const uint32_t*>(xplane + base + k0);
                    b1 = *reinterpret_cast<const uint32_t*>(xplane + base + k1);
                }
                int gc0 = 0, gc1 = 0, gc2 = 0, gc3 = 0;
                int uc0 = 0, uc1 = 0, uc2 = 0, uc3 = 0;
                mma_q4_dot(ga0, ga1, ga2, ga3, b0, b1, gc0, gc1, gc2, gc3);
                mma_q4_dot(ua0, ua1, ua2, ua3, b0, b1, uc0, uc1, uc2, uc3);
                const uint32_t output_cols[2] = {2u * lane_col, 2u * lane_col + 1u};
                const int gate_values[4] = {gc0, gc1, gc2, gc3};
                const int up_values[4] = {uc0, uc1, uc2, uc3};
                const uint32_t block0 = row0 * 20u + block_in_row;
                const uint32_t block1 = row1 * 20u + block_in_row;
                int sc0, mn0, sc1, mn1;
                scale_min_k4(metadata + block0 * 12u, group, sc0, mn0);
                scale_min_k4(metadata + block1 * 12u, group, sc1, mn1);
                const uint16_t d0 = decode_half_field_fixed_device<13u - 10u, DExpBase>(dplane, block0);
                const uint16_t dmin0 = decode_half_field_fixed_device<13u - 10u, DMinExpBase>(dminplane, block0);
                const uint16_t d1 = decode_half_field_fixed_device<13u - 10u, DExpBase>(dplane, block1);
                const uint16_t dmin1 = decode_half_field_fixed_device<13u - 10u, DMinExpBase>(dminplane, block1);
                const float f0 = __half2float(__ushort_as_half(d0)) * float(sc0);
                const float fmin0 = __half2float(__ushort_as_half(dmin0)) * float(mn0);
                const float f1 = __half2float(__ushort_as_half(d1)) * float(sc1);
                const float fmin1 = __half2float(__ushort_as_half(dmin1)) * float(mn1);
                const uint32_t up_block0 = (p.vec.rows + row0) * 20u + block_in_row;
                const uint32_t up_block1 = (p.vec.rows + row1) * 20u + block_in_row;
                int up_sc0, up_mn0, up_sc1, up_mn1;
                scale_min_k4(metadata + up_block0 * 12u, group, up_sc0, up_mn0);
                scale_min_k4(metadata + up_block1 * 12u, group, up_sc1, up_mn1);
                const uint16_t up_d0 = decode_half_field_fixed_device<13u - 10u, DExpBase>(dplane, up_block0);
                const uint16_t up_dmin0 = decode_half_field_fixed_device<13u - 10u, DMinExpBase>(dminplane, up_block0);
                const uint16_t up_d1 = decode_half_field_fixed_device<13u - 10u, DExpBase>(dplane, up_block1);
                const uint16_t up_dmin1 = decode_half_field_fixed_device<13u - 10u, DMinExpBase>(dminplane, up_block1);
                const float uf0 = __half2float(__ushort_as_half(up_d0)) * float(up_sc0);
                const float ufmin0 = __half2float(__ushort_as_half(up_dmin0)) * float(up_mn0);
                const float uf1 = __half2float(__ushort_as_half(up_d1)) * float(up_sc1);
                const float ufmin1 = __half2float(__ushort_as_half(up_dmin1)) * float(up_mn1);
                for (uint32_t r = 0; r < 2u; ++r) {
                    const float f = r == 0u ? f0 : f1;
                    const float fmin = r == 0u ? fmin0 : fmin1;
                    const float uf = r == 0u ? uf0 : uf1;
                    const float ufmin = r == 0u ? ufmin0 : ufmin1;
                    for (uint32_t c = 0; c < 2u; ++c) {
                        const uint32_t col = output_cols[c];
                        if (col < 4u) {
                            const uint32_t slice = qslice;
                            const float2 ds = __half22float2(p.vec.ds[uint64_t(col) * 160u + slice]);
                            const uint32_t vi = r * 2u + c;
                            const float gv = f * ds.x * float(gate_values[vi]) - fmin * ds.y;
                            const float uvv = uf * ds.x * float(up_values[vi]) - ufmin * ds.y;
                            if (r == 0u && c == 0u) { gate0 += gv; up0 += uvv; }
                            if (r == 0u && c == 1u) { gate1 += gv; up1 += uvv; }
                            if (r == 1u && c == 0u) { gate2 += gv; up2 += uvv; }
                            if (r == 1u && c == 1u) { gate3 += gv; up3 += uvv; }
                        }
                    }
                }
            }
            if (lane_col < 2u) {
                const uint32_t col0 = 2u * lane_col;
                const uint32_t col1 = col0 + 1u;
                buckets[0][lane_group][col0][u] = gate0;
                buckets[1][lane_group][col0][u] = up0;
                buckets[0][lane_group][col1][u] = gate1;
                buckets[1][lane_group][col1][u] = up1;
                buckets[0][lane_group + 8u][col0][u] = gate2;
                buckets[1][lane_group + 8u][col0][u] = up2;
                buckets[0][lane_group + 8u][col1][u] = gate3;
                buckets[1][lane_group + 8u][col1][u] = up3;
            }
    __syncthreads();
    float gate = 0.0f, up = 0.0f;
    for (uint32_t item = warp; item < 64u; item += 32u) {
        const uint32_t row = item / 4u;
        const uint32_t col = item & 3u;
        gate = buckets[0][row][col][tid & 31u];
        up = buckets[1][row][col][tid & 31u];
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            gate += __shfl_xor_sync(0xffffffffu, gate, offset);
            up += __shfl_xor_sync(0xffffffffu, up, offset);
        }
        if ((tid & 31u) == 0u) {
            vec_store(p.vec.out, int(out_row + row), int(col), vec_silu(gate) * up);
        }
    }
}

static void cpu_mma_fragment_checks() {
    int8_t a[16][32];
    int8_t b[32][8];
    uint8_t packed_a[16][16]{};
    int8_t decoded_a[16][32]{};
    int8_t decoded_b[32][8]{};
    bool a_coverage[16][32]{};
    bool b_coverage[32][8]{};
    bool c_coverage[16][8]{};
    for (int r = 0; r < 16; ++r) {
        for (int k = 0; k < 32; ++k) {
            a[r][k] = int8_t(k & 15);
            packed_a[r][k / 2] |= uint8_t(a[r][k] << ((k & 1) * 4));
        }
        for (int k = 0; k < 32; ++k) {
            const int decoded = (packed_a[r][k / 2] >> ((k & 1) * 4)) & 15;
            if (decoded != a[r][k]) { throw std::runtime_error("Q4 nibble fixture mismatch"); }
        }
    }
    for (int k = 0; k < 32; ++k) {
        for (int c = 0; c < 8; ++c) { b[k][c] = c < 4 ? int8_t(-128 + k + c * 32) : int8_t(0); }
    }
    for (int lane = 0; lane < 32; ++lane) {
        const int row = lane >> 2;
        const int col = lane & 3;
        const int rows[4] = {row, row + 8, row, row + 8};
        const int starts[4] = {col * 4, col * 4, col * 4 + 16, col * 4 + 16};
        uint32_t afrag[4]{};
        uint32_t bfrag[2]{};
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) { afrag[i] |= uint32_t(uint8_t(a[rows[i]][starts[i] + j])) << (8 * j); }
        }
        for (int j = 0; j < 4; ++j) {
            bfrag[0] |= uint32_t(uint8_t(b[col * 4 + j][row])) << (8 * j);
            bfrag[1] |= uint32_t(uint8_t(b[col * 4 + 16 + j][row])) << (8 * j);
        }
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                if (int8_t(afrag[i] >> (8 * j)) != a[rows[i]][starts[i] + j]) { throw std::runtime_error("A fragment mapping mismatch"); }
            }
        }
        for (int j = 0; j < 4; ++j) {
            if (int8_t(bfrag[0] >> (8 * j)) != b[col * 4 + j][row] ||
                int8_t(bfrag[1] >> (8 * j)) != b[col * 4 + 16 + j][row]) { throw std::runtime_error("B fragment mapping mismatch"); }
        }
        const int a_rows[4] = {row, row + 8, row, row + 8};
        const int a_starts[4] = {col * 4, col * 4, col * 4 + 16, col * 4 + 16};
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                const int kk = a_starts[i] + j;
                if (a_coverage[a_rows[i]][kk]) { throw std::runtime_error("A fragment mapping duplicated"); }
                a_coverage[a_rows[i]][kk] = true;
                decoded_a[a_rows[i]][kk] = int8_t(afrag[i] >> (8 * j));
            }
        }
        for (int j = 0; j < 4; ++j) {
            const int k0 = col * 4 + j;
            const int k1 = col * 4 + 16 + j;
            if (b_coverage[k0][row] || b_coverage[k1][row]) { throw std::runtime_error("B fragment mapping duplicated"); }
            b_coverage[k0][row] = true;
            b_coverage[k1][row] = true;
            decoded_b[k0][row] = int8_t(bfrag[0] >> (8 * j));
            decoded_b[k1][row] = int8_t(bfrag[1] >> (8 * j));
        }
        const int c_rows[4] = {row, row, row + 8, row + 8};
        const int c_cols[4] = {2 * col, 2 * col + 1, 2 * col, 2 * col + 1};
        for (int i = 0; i < 4; ++i) {
            if (c_coverage[c_rows[i]][c_cols[i]]) { throw std::runtime_error("C fragment mapping duplicated"); }
            c_coverage[c_rows[i]][c_cols[i]] = true;
        }
    }
    for (int row = 0; row < 16; ++row) {
        for (int kk = 0; kk < 32; ++kk) { if (!a_coverage[row][kk]) { throw std::runtime_error("A fragment mapping gap"); } }
        for (int col = 0; col < 8; ++col) {
            if (!c_coverage[row][col]) { throw std::runtime_error("C fragment mapping gap"); }
            int dot = 0;
            for (int k = 0; k < 32; ++k) { dot += int(decoded_a[row][k]) * int(decoded_b[k][col]); }
            int reference = 0;
            for (int k = 0; k < 32; ++k) { reference += int(a[row][k]) * int(b[k][col]); }
            if (dot != reference) { throw std::runtime_error("MMA integer fixture mismatch"); }
        }
    }
    for (int kk = 0; kk < 32; ++kk) {
        for (int col = 0; col < 8; ++col) { if (!b_coverage[kk][col]) { throw std::runtime_error("B fragment mapping gap"); } }
    }
    int max_abs = 0;
    for (int q = 0; q < 16; ++q) {
        for (int x = -128; x <= 127; ++x) { max_abs = std::max(max_abs, std::abs(q * x)); }
    }
    if (max_abs != 1920 || 32 * max_abs != 61440) { throw std::runtime_error("MMA integer bound mismatch"); }
}

static void cpu_shared_bank_checks() {
    unsigned old_max_conflict = 0;
    unsigned padded_max_conflict = 0;
    for (unsigned u = 0; u < 32; ++u) {
        for (unsigned row_half = 0; row_half < 2; ++row_half) {
            for (unsigned col_part = 0; col_part < 2; ++col_part) {
                unsigned old_counts[32]{};
                unsigned padded_counts[32]{};
                for (unsigned lane_group = 0; lane_group < 8; ++lane_group) {
                    for (unsigned lane_col = 0; lane_col < 2; ++lane_col) {
                        const unsigned row = lane_group + row_half * 8;
                        const unsigned col = lane_col * 2 + col_part;
                        const unsigned old_word = ((row * 4 + col) * 32 + u);
                        const unsigned padded_word = ((row * 4 + col) * 33 + u);
                        ++old_counts[old_word & 31u];
                        ++padded_counts[padded_word & 31u];
                    }
                }
                unsigned old_distinct = 0;
                unsigned padded_distinct = 0;
                for (unsigned bank = 0; bank < 32; ++bank) {
                    old_distinct += old_counts[bank] != 0;
                    padded_distinct += padded_counts[bank] != 0;
                    old_max_conflict = std::max(old_max_conflict, old_counts[bank]);
                    padded_max_conflict = std::max(padded_max_conflict, padded_counts[bank]);
                }
                if (old_distinct != 1 || padded_distinct != 16) {
                    throw std::runtime_error("shared bucket store bank mapping mismatch");
                }
            }
        }
    }
    for (unsigned plane = 0; plane < 2; ++plane) {
        for (unsigned row = 0; row < 16; ++row) {
            for (unsigned col = 0; col < 4; ++col) {
                unsigned old_seen[32]{};
                unsigned padded_seen[32]{};
                for (unsigned lane = 0; lane < 32; ++lane) {
                    const unsigned old_word = (((plane * 16 + row) * 4 + col) * 32 + lane);
                    const unsigned padded_word = (((plane * 16 + row) * 4 + col) * 33 + lane);
                    old_seen[old_word & 31u]++;
                    padded_seen[padded_word & 31u]++;
                }
                for (unsigned bank = 0; bank < 32; ++bank) {
                    if (old_seen[bank] != 1 || padded_seen[bank] != 1) {
                        throw std::runtime_error("shared bucket reduction read is not bank bijective");
                    }
                }
            }
        }
    }
    if (old_max_conflict != 16 || padded_max_conflict != 1) {
        throw std::runtime_error("shared bucket store conflict degree mismatch");
    }
}

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--cpu-check") {
        cpu_mma_fragment_checks();
        cpu_shared_bank_checks();
        std::vector<uint8_t> ql(16u * 20u * 128u);
        for (size_t i = 0; i < ql.size(); ++i) { ql[i] = uint8_t((i * 73u + (i >> 5) * 19u) & 255u); }
        if (restore_transposed_qs_host(transpose_qs_host(ql, 16u), 16u) != ql) { throw std::runtime_error("CPU transposed QL inverse mismatch"); }
        std::cout << "{\"cpu_fragment_mapping_exact\":true,\"cpu_shared_bank_mapping_exact\":true,\"transposed_cpu_ql_roundtrip_exact\":true,\"store_active_lanes\":16,\"old_store_conflict_degree\":16,\"padded_store_conflict_degree\":1,\"reduction_read_bank_bijection\":true,\"mma_bound\":61440}\n";
        return 0;
    }
    if (argc != 5) { std::cerr << "model offset N activation\n"; return 2; }
    cpu_mma_fragment_checks();
    cpu_shared_bank_checks();
    const int n = std::stoi(argv[3]);
    constexpr int k = 5120;
    if (n != 34816) { throw std::runtime_error("fused MLP root shape required"); }
    if ((n % 32) != 0) { throw std::runtime_error("row count must be a multiple of 32"); }
    const size_t bytes = size_t(n) * (k / 256) * 144;
    std::vector<uint8_t> raw(bytes);
    std::ifstream file(argv[1], std::ios::binary);
    file.seekg(std::stoull(argv[2]));
    file.read(reinterpret_cast<char*>(raw.data()), raw.size());
    if (size_t(file.gcount()) != raw.size()) { throw std::runtime_error("truncated model tensor"); }
    PackedScales packed = pack_scales(raw, n, k);
    if (!packed.coded || packed.sign_bits || packed.d_exp_bits != 3 || packed.d_exp_min != 0 ||
        packed.dmin_exp_bits != 3 || packed.dmin_exp_min != 2) { throw std::runtime_error("expected fixed13/13 exact fields"); }
    const std::vector<uint8_t> transposed_host = transpose_qs_host(packed.qs, uint32_t(n));
    const std::vector<uint8_t> restored_qs_host = restore_transposed_qs_host(transposed_host, uint32_t(n));
    if (restored_qs_host != packed.qs) { throw std::runtime_error("transposed QL CPU inverse mismatch"); }
    PackedScales transposed_roundtrip = packed;
    transposed_roundtrip.qs = restored_qs_host;
    if (unpack_cpu(transposed_roundtrip) != raw) { throw std::runtime_error("transposed QL full CPU payload mismatch"); }
    const Activation activation = load_activation(argv[4]);
    if (activation.t != 4 || activation.fused != 1) { throw std::runtime_error("fused T4 activation required"); }
    uint8_t *raw_device, *dm_device, *qs_device, *metadata_device, *d_device, *dmin_device;
    uint8_t *transposed_device, *restored_qs_device;
    int8_t* x_device;
    half2* ds_device;
    float* output[5];
    auto alloc = [](auto** p, size_t count) { check(cudaMalloc(p, count), "allocate"); };
    alloc(&raw_device, raw.size());
    alloc(&dm_device, packed.dm.size());
    alloc(&qs_device, packed.qs.size());
    alloc(&metadata_device, packed.raw_metadata.size());
    alloc(&d_device, packed.fixed_d_code.size());
    alloc(&dmin_device, packed.fixed_dmin_code.size());
    alloc(&transposed_device, transposed_host.size());
    alloc(&restored_qs_device, packed.qs.size());
    alloc(&x_device, activation.qs.size());
    alloc(&ds_device, activation.ds.size() * sizeof(half2));
    check(cudaMemcpy(raw_device, raw.data(), raw.size(), cudaMemcpyHostToDevice), "raw copy");
    check(cudaMemcpy(dm_device, packed.dm.data(), packed.dm.size(), cudaMemcpyHostToDevice), "dm copy");
    check(cudaMemcpy(qs_device, packed.qs.data(), packed.qs.size(), cudaMemcpyHostToDevice), "qs copy");
    check(cudaMemcpy(metadata_device, packed.raw_metadata.data(), packed.raw_metadata.size(), cudaMemcpyHostToDevice), "metadata copy");
    check(cudaMemcpy(d_device, packed.fixed_d_code.data(), packed.fixed_d_code.size(), cudaMemcpyHostToDevice), "d copy");
    check(cudaMemcpy(dmin_device, packed.fixed_dmin_code.data(), packed.fixed_dmin_code.size(), cudaMemcpyHostToDevice), "dmin copy");
    check(cudaMemcpy(transposed_device, transposed_host.data(), transposed_host.size(), cudaMemcpyHostToDevice), "transposed QL copy");
    check(cudaMemcpy(x_device, activation.qs.data(), activation.qs.size(), cudaMemcpyHostToDevice), "activation copy");
    check(cudaMemcpy(ds_device, activation.ds.data(), activation.ds.size() * sizeof(half2), cudaMemcpyHostToDevice), "activation scales copy");
    uint8_t* reconstructed;
    alloc(&reconstructed, raw.size());
    const uint64_t block_count = uint64_t(n) * (k / 256);
    reconstruct_fixed_mma_kernel<<<(block_count + 255) / 256, 256>>>(n, dm_device, qs_device, metadata_device,
        d_device, dmin_device, reconstructed);
    check(cudaGetLastError(), "full byte reconstruction launch");
    std::vector<uint8_t> reconstructed_host(raw.size());
    check(cudaMemcpy(reconstructed_host.data(), reconstructed, raw.size(), cudaMemcpyDeviceToHost), "full byte reconstruction readback");
    if (reconstructed_host != raw) { throw std::runtime_error("full GPU Q4_K payload reconstruction mismatch"); }
    restore_transposed_qs_kernel<<<(uint64_t(n) * 20u * 128u + 255u) / 256u, 256>>>(uint32_t(n), transposed_device, restored_qs_device);
    check(cudaGetLastError(), "transposed QL restore launch");
    std::vector<uint8_t> restored_qs_gpu(packed.qs.size());
    check(cudaMemcpy(restored_qs_gpu.data(), restored_qs_device, restored_qs_gpu.size(), cudaMemcpyDeviceToHost), "transposed QL restore readback");
    if (restored_qs_gpu != packed.qs) { throw std::runtime_error("full GPU transposed QL reconstruction mismatch"); }
    reconstruct_fixed_mma_kernel<<<(block_count + 255) / 256, 256>>>(n, dm_device, restored_qs_device, metadata_device,
        d_device, dmin_device, reconstructed);
    check(cudaGetLastError(), "transposed full byte reconstruction launch");
    check(cudaMemcpy(reconstructed_host.data(), reconstructed, raw.size(), cudaMemcpyDeviceToHost), "transposed full byte reconstruction readback");
    if (reconstructed_host != raw) { throw std::runtime_error("full transposed GPU Q4_K payload mismatch"); }
    check(cudaFree(reconstructed), "full byte reconstruction free");
    const int out_rows = n / 2;
    const size_t out_bytes = size_t(out_rows) * 4 * sizeof(float);
    for (auto& p : output) { alloc(&p, out_bytes + 64); check(cudaMemset(reinterpret_cast<uint8_t*>(p) + out_bytes, 0xA5, 64), "canary initialize"); }
    VecArgs args{raw_device, (k / 256) * 144, out_rows, k, x_device, ds_device,
                 {nullptr, output[0], out_rows, ninfer::ops::gguf::Epilogue::Store, nullptr, 0}};
    cudaStream_t stream;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream create");
    auto launch_baseline = [&](int mode, float* dst) {
        args.out.f32 = dst;
        if (mode == 0) { vec::launch<GGML_TYPE_Q4_K, 4, true, k>(args, stream); }
        else if (mode == 1) { launch_candidate<4, false, true, false, 3, 0, 3, 2>(args, packed, dm_device, qs_device, metadata_device, d_device, dmin_device, stream); }
        else if (mode == 2) { q4_mma_fused_kernel<0, 2, false><<<out_rows / 16, 256, 0, stream>>>(ScaleArgs{args, dm_device, qs_device, metadata_device, d_device, dmin_device, 0, 2, 3, 3, 0, true}); check(cudaGetLastError(), "mma control launch"); }
        else if (mode == 3) { q4_mma_fused_kernel<0, 2, true><<<out_rows / 16, 256, 0, stream>>>(ScaleArgs{args, dm_device, qs_device, metadata_device, d_device, dmin_device, 0, 2, 3, 3, 0, true}); check(cudaGetLastError(), "mma padded launch"); }
        else { q4_mma_transposed_kernel<0, 2><<<out_rows / 16, 1024, 0, stream>>>(ScaleArgs{args, dm_device, transposed_device, metadata_device, d_device, dmin_device, 0, 2, 3, 3, 0, true}); check(cudaGetLastError(), "mma transposed launch"); }
    };
    std::vector<float> reference(size_t(out_rows) * 4), actual(reference.size());
    for (int mode = 0; mode < 5; ++mode) {
        launch_baseline(mode, output[mode]);
        check(cudaStreamSynchronize(stream), "output sync");
        std::array<uint8_t, 64> guard{};
        check(cudaMemcpy(guard.data(), reinterpret_cast<uint8_t*>(output[mode]) + out_bytes, 64, cudaMemcpyDeviceToHost), "canary copy");
        if (!std::all_of(guard.begin(), guard.end(), [](uint8_t v) { return v == 0xA5; })) { throw std::runtime_error("output canary changed"); }
        check(cudaMemcpy(actual.data(), output[mode], out_bytes, cudaMemcpyDeviceToHost), "output readback");
        if (mode == 0) { reference = actual; }
        else if (std::memcmp(reference.data(), actual.data(), out_bytes) != 0) { throw std::runtime_error("FP32 output mismatch"); }
    }
    cudaGraphExec_t graphs[5];
    for (int mode = 0; mode < 5; ++mode) {
        cudaGraph_t graph;
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "graph capture begin");
        launch_baseline(mode, output[mode]);
        check(cudaStreamEndCapture(stream, &graph), "graph capture end");
        check(cudaGraphInstantiate(&graphs[mode], graph, 0), "graph instantiate");
        check(cudaGraphDestroy(graph), "graph destroy");
    }
    for (int r = 0; r < 5; ++r) {
        for (int order = 0; order < 5; ++order) {
            const int mode = (r & 1) ? 4 - order : order;
            check(cudaGraphLaunch(graphs[mode], stream), "graph warmup");
        }
    }
    check(cudaStreamSynchronize(stream), "graph warmup sync");
    float samples_ms[5][5]{};
    cudaEvent_t start, stop;
    check(cudaEventCreate(&start), "timing start event");
    check(cudaEventCreate(&stop), "timing stop event");
    for (int r = 0; r < 5; ++r) {
        for (int order = 0; order < 5; ++order) {
            const int mode = (r & 1) ? 4 - order : order;
            check(cudaEventRecord(start, stream), "timing start record");
            for (int repeat = 0; repeat < 10; ++repeat) { check(cudaGraphLaunch(graphs[mode], stream), "timed graph launch"); }
            check(cudaEventRecord(stop, stream), "timing stop record");
            check(cudaEventSynchronize(stop), "timing stop sync");
            float elapsed = 0.0f;
            check(cudaEventElapsedTime(&elapsed, start, stop), "timing elapsed");
            samples_ms[mode][r] = elapsed / 10.0f;
            std::array<uint8_t, 64> guard{};
            check(cudaMemcpy(guard.data(), reinterpret_cast<uint8_t*>(output[mode]) + out_bytes, 64, cudaMemcpyDeviceToHost), "timed graph canary readback");
            if (!std::all_of(guard.begin(), guard.end(), [](uint8_t v) { return v == 0xA5; })) { throw std::runtime_error("timed graph output canary changed"); }
        }
    }
    for (int mode = 0; mode < 5; ++mode) {
        check(cudaGraphLaunch(graphs[mode], stream), "graph exactness launch");
        check(cudaStreamSynchronize(stream), "graph exactness sync");
        check(cudaMemcpy(actual.data(), output[mode], out_bytes, cudaMemcpyDeviceToHost), "graph exactness readback");
        if (std::memcmp(reference.data(), actual.data(), out_bytes) != 0) { throw std::runtime_error("graph FP32 output mismatch"); }
    }
    KernelResource baseline_resource = kernel_resource(vec::kernel<GGML_TYPE_Q4_K, 4, true, k>, 128);
    KernelResource scalar_resource = kernel_resource(scale_kernel<4, false, true, 3, 0, 3, 2>, 128);
    KernelResource mma_control_resource = kernel_resource(q4_mma_fused_kernel<0, 2, false>, 256);
    KernelResource mma_padded_resource = kernel_resource(q4_mma_fused_kernel<0, 2, true>, 256);
    KernelResource mma_transposed_resource = kernel_resource(q4_mma_transposed_kernel<0, 2>, 1024);
    __nv_bfloat16* bfout;
    alloc(&bfout, size_t(out_rows) * 4 * sizeof(__nv_bfloat16) + 64);
    check(cudaMemset(reinterpret_cast<uint8_t*>(bfout) + size_t(out_rows) * 4 * sizeof(__nv_bfloat16), 0xA5, 64), "bf16 canary initialize");
    std::vector<uint16_t> bf_reference(size_t(out_rows) * 4), bf_actual(bf_reference.size());
    args.out.f32 = nullptr;
    args.out.bf16 = bfout;
    for (int mode = 0; mode < 5; ++mode) {
        args.weight = raw_device;
        if (mode == 0) { vec::launch<GGML_TYPE_Q4_K, 4, true, k>(args, stream); }
        else if (mode == 1) { launch_candidate<4, false, true, false, 3, 0, 3, 2>(args, packed, dm_device, qs_device, metadata_device, d_device, dmin_device, stream); }
        else if (mode == 2) { q4_mma_fused_kernel<0, 2, false><<<out_rows / 16, 256, 0, stream>>>(ScaleArgs{args, dm_device, qs_device, metadata_device, d_device, dmin_device, 0, 2, 3, 3, 0, true}); check(cudaGetLastError(), "mma control bf16 launch"); }
        else if (mode == 3) { q4_mma_fused_kernel<0, 2, true><<<out_rows / 16, 256, 0, stream>>>(ScaleArgs{args, dm_device, qs_device, metadata_device, d_device, dmin_device, 0, 2, 3, 3, 0, true}); check(cudaGetLastError(), "mma padded bf16 launch"); }
        else { q4_mma_transposed_kernel<0, 2><<<out_rows / 16, 1024, 0, stream>>>(ScaleArgs{args, dm_device, transposed_device, metadata_device, d_device, dmin_device, 0, 2, 3, 3, 0, true}); check(cudaGetLastError(), "mma transposed bf16 launch"); }
        check(cudaStreamSynchronize(stream), "bf16 sync");
        std::array<uint8_t, 64> bf_guard{};
        check(cudaMemcpy(bf_guard.data(), reinterpret_cast<uint8_t*>(bfout) + size_t(out_rows) * 4 * sizeof(__nv_bfloat16), 64, cudaMemcpyDeviceToHost), "bf16 canary readback");
        if (!std::all_of(bf_guard.begin(), bf_guard.end(), [](uint8_t v) { return v == 0xA5; })) { throw std::runtime_error("BF16 output canary changed"); }
        check(cudaMemcpy(bf_actual.data(), bfout, bf_actual.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost), "bf16 readback");
        if (mode == 0) { bf_reference = bf_actual; }
        else if (bf_reference != bf_actual) { throw std::runtime_error("BF16 output mismatch"); }
    }
    cudaGraphExec_t transposed_bf_graph;
    args.weight = raw_device;
    args.out.bf16 = bfout;
    check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "transposed BF16 graph capture begin");
    q4_mma_transposed_kernel<0, 2><<<out_rows / 16, 1024, 0, stream>>>(ScaleArgs{args, dm_device, transposed_device, metadata_device, d_device, dmin_device, 0, 2, 3, 3, 0, true});
    check(cudaGetLastError(), "transposed BF16 graph candidate launch");
    cudaGraph_t transposed_bf_raw_graph;
    check(cudaStreamEndCapture(stream, &transposed_bf_raw_graph), "transposed BF16 graph capture end");
    check(cudaGraphInstantiate(&transposed_bf_graph, transposed_bf_raw_graph, 0), "transposed BF16 graph instantiate");
    check(cudaGraphDestroy(transposed_bf_raw_graph), "transposed BF16 graph destroy");
    check(cudaGraphLaunch(transposed_bf_graph, stream), "transposed BF16 graph launch");
    check(cudaStreamSynchronize(stream), "transposed BF16 graph sync");
    check(cudaMemcpy(bf_actual.data(), bfout, bf_actual.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost), "transposed BF16 graph readback");
    if (bf_actual != bf_reference) { throw std::runtime_error("transposed BF16 graph output mismatch"); }
    std::array<uint8_t, 64> transposed_bf_guard{};
    check(cudaMemcpy(transposed_bf_guard.data(), reinterpret_cast<uint8_t*>(bfout) + size_t(out_rows) * 4 * sizeof(__nv_bfloat16), 64, cudaMemcpyDeviceToHost), "transposed BF16 graph canary readback");
    if (!std::all_of(transposed_bf_guard.begin(), transposed_bf_guard.end(), [](uint8_t v) { return v == 0xA5; })) { throw std::runtime_error("transposed BF16 graph canary changed"); }
    check(cudaGraphExecDestroy(transposed_bf_graph), "transposed BF16 graph executable destroy");
    check(cudaEventDestroy(start), "timing event destroy");
    check(cudaEventDestroy(stop), "timing event destroy");
    std::cout << "{\"matrix\":\"weight/000001\",\"T\":4,\"K\":5120,\"fused_swiglu\":true,\"fp32_bit_exact\":true,\"bf16_bit_exact\":true,\"graph_fp32_bit_exact\":true,\"gpu_all_payload_bytes_exact\":true,\"transposed_cpu_full_payload_roundtrip_exact\":true,\"transposed_gpu_full_payload_reconstruction_exact\":true,\"transposed_fp32_bit_exact\":true,\"transposed_bf16_bit_exact\":true,\"transposed_graph_fp32_bit_exact\":true,\"transposed_graph_bf16_bit_exact\":true,\"output_canaries_exact\":true,\"cpu_fragment_mapping_exact\":true,\"cpu_shared_bank_mapping_exact\":true,\"mma_bound\":61440,\"shared_bytes_control\":16384,\"shared_bytes_padded\":16896,\"shared_bytes_transposed\":16896,\"shared_bytes_delta\":512,\"store_conflict_degree_control\":16,\"store_conflict_degree_padded\":1,\"logical_global_bytes\":" << bytes + activation.qs.size() + activation.ds.size() * sizeof(half2) << ",\"transposed_ql_bytes\":" << transposed_host.size() << ",\"resources\":{";
    print_resource("raw_ggml", baseline_resource);
    std::cout << ',';
    print_resource("fixed13_scalar", scalar_resource);
    std::cout << ',';
    print_resource("register_mma_fixed32_control", mma_control_resource);
    std::cout << ',';
    print_resource("register_mma_padded33", mma_padded_resource);
    std::cout << ',';
    print_resource("register_mma_transposed_ql_32warp", mma_transposed_resource);
    std::cout << "},\"samples_ms\":[";
    for (int mode = 0; mode < 5; ++mode) {
        if (mode) { std::cout << ','; }
        const char* name = mode == 0 ? "raw_ggml" : mode == 1 ? "fixed13_scalar" : mode == 2 ? "register_mma_fixed32_control" : mode == 3 ? "register_mma_padded33" : "transposed_ql_register_mma_32warp";
        std::cout << "{\"mode\":\"" << name << "\",\"values\":[";
        for (int r = 0; r < 5; ++r) { if (r) { std::cout << ','; } std::cout << samples_ms[mode][r]; }
        std::cout << "]}";
    }
    std::cout << "]}\n";
    for (auto graph : graphs) { check(cudaGraphExecDestroy(graph), "graph executable destroy"); }
    check(cudaStreamDestroy(stream), "stream destroy");
    for (auto p : output) { check(cudaFree(p), "free output"); }
    check(cudaFree(bfout), "free bf16");
    check(cudaFree(raw_device), "free raw");
    check(cudaFree(dm_device), "free dm");
    check(cudaFree(qs_device), "free qs");
    check(cudaFree(metadata_device), "free metadata");
    check(cudaFree(d_device), "free d");
    check(cudaFree(dmin_device), "free dmin");
    check(cudaFree(transposed_device), "free transposed QL");
    check(cudaFree(restored_qs_device), "free restored QL");
    check(cudaFree(x_device), "free x");
    check(cudaFree(ds_device), "free ds");
    return 0;
}
