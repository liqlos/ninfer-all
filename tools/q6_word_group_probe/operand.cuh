#pragma once

#include "probe_api.h"
#include <cuda_fp16.h>

namespace q6wg {

struct BlockAddress {
    const std::uint8_t* slab;
    unsigned slot;
};

__device__ __forceinline__ BlockAddress block_address(const Q6WGView& view,
                                                      std::uint64_t block) {
    return {view.base + (block >> 4) * 3360, static_cast<unsigned>(block & 15)};
}

__device__ __forceinline__ uint4 unsigned_words(BlockAddress b, unsigned group) {
    const uint2 lo = *reinterpret_cast<const uint2*>(b.slab + b.slot * 128 + group * 8);
    const unsigned hi = *reinterpret_cast<const unsigned*>(
        b.slab + 2048 + b.slot * 64 + group * 4);
    return make_uint4(
        (lo.x & 0x0f0f0f0fu) | ((hi << 4) & 0x30303030u),
        ((lo.x >> 4) & 0x0f0f0f0fu) | ((hi << 2) & 0x30303030u),
        (lo.y & 0x0f0f0f0fu) | (hi & 0x30303030u),
        ((lo.y >> 4) & 0x0f0f0f0fu) | ((hi >> 2) & 0x30303030u));
}

__device__ __forceinline__ float block_d(BlockAddress b) {
    return __half2float(*reinterpret_cast<const half*>(b.slab + 3328 + b.slot * 2));
}

__device__ __forceinline__ int block_scale(BlockAddress b, unsigned group) {
    return reinterpret_cast<const std::int8_t*>(b.slab + 3072 + b.slot * 16)[group];
}

__device__ __forceinline__ int vector_center(unsigned w) {
    return static_cast<int>(w ^ 0xe0e0e0e0u ^ ((w & 0x20202020u) * 6u));
}

}
