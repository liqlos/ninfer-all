#pragma once

#include "operand.cuh"

namespace ninfer::ops::gguf::detail::vec {

__device__ __forceinline__ void q6_word_group_decode(const Q6WGView& operand,
                                                     std::uint64_t block, int sub,
                                                     Slice& s) {
    const auto b = q6wg::block_address(operand, block);
    const float d = q6wg::block_d(b);
    s.f[0] = d * float(q6wg::block_scale(b, 2 * sub));
    s.f[1] = d * float(q6wg::block_scale(b, 2 * sub + 1));
#pragma unroll
    for (int half_group = 0; half_group < 2; ++half_group) {
        const uint4 w = q6wg::unsigned_words(b, 2 * sub + half_group);
        s.w[4 * half_group + 0] = q6wg::vector_center(w.x);
        s.w[4 * half_group + 1] = q6wg::vector_center(w.y);
        s.w[4 * half_group + 2] = q6wg::vector_center(w.z);
        s.w[4 * half_group + 3] = q6wg::vector_center(w.w);
    }
}

}
