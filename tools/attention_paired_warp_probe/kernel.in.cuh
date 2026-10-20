__global__ __launch_bounds__(512, 1)
void causal_attention_prompt_i8_paired_on_kernel(
    const __nv_bfloat16* __restrict__ q,
    const std::int8_t* __restrict__ cache_k,
    const std::int8_t* __restrict__ cache_v,
    const __half* __restrict__ cache_k_scale,
    const __half* __restrict__ cache_v_scale,
    PagedKVBatchMetadata<false> metadata,
    const std::int32_t* __restrict__ positions,
    float scale, __nv_bfloat16* __restrict__ out, std::int32_t width) {
    using Geometry = CausalD256H24Kv4;
    using Stage = CausalPromptI8FastStage<false>;
    constexpr int D = 256;
    constexpr int DB16 = 128;
    constexpr int Threads = 512;
    constexpr int Br = 128;
    constexpr int Bc = 64;
    constexpr int Groups = 4;
    constexpr int VGroups = 4;
    constexpr int GroupKc = 2;
    constexpr int QKNt = 4;
    constexpr int PVKs = 4;
    constexpr int DBlocks = 8;
    constexpr int GroupDBlocks = 4;
    constexpr int PassDBlocks = 2;
    constexpr int QBytes = Br * D;
    constexpr float Log2E = 1.4426950408889634074f;
    constexpr unsigned FullMask = 0xffffffffu;
    constexpr bool PackedValues = false;
    static_assert(QBytes + 2 * Stage::Bytes + 2 * Br * sizeof(float) == 101376);

    extern __shared__ __align__(16) unsigned char smem_raw[];
    std::int8_t* q_i8 = reinterpret_cast<std::int8_t*>(smem_raw);
    __nv_bfloat16* q_b16 = reinterpret_cast<__nv_bfloat16*>(q_i8);
    float* max_s = reinterpret_cast<float*>(smem_raw + QBytes + 2 * Stage::Bytes);
    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int pair = warp >> 1;
    const int side = warp & 1;
    const int lane = tid & 31;
    const int gid = lane >> 2;
    const int lid = lane & 3;
@DISPATCH_AND_ZERO@
    const int base_pos = positions[0];
    const std::int32_t* block_table = metadata.block_table();
    const int rows = min(Br, tokens - q0);
    const int max_query_abs = base_pos + q0 + rows - 1;
    const int key_blocks = max_query_abs / Bc + 1;
    const int row_base = pair * 16;
    float* prepare_scales = reinterpret_cast<float*>(smem_raw + QBytes);
#pragma unroll 1
    for (int r = 0; r < 8; ++r) {
        const int row = row_base + side * 8 + r;
        const int token = q0 + row;
        const bool valid = row < rows;
@Q_PREPARE@
    }
    __syncthreads();
    const float q_scale0 = prepare_scales[(row_base + gid) * Groups + lid];
    const float q_scale1 = prepare_scales[(row_base + gid + 8) * Groups + lid];
    __syncthreads();

    const auto stage_base = [&](int stage) {
        return smem_raw + QBytes + stage * Stage::Bytes;
    };
    const auto issue_tile = [&](int kb) {
        unsigned char* base = stage_base(kb & 1);
        std::int8_t* k_s = reinterpret_cast<std::int8_t*>(base);
        std::int8_t* v_s = k_s + kCausalPromptI8FastTileBytes;
        __half* ks_s = reinterpret_cast<__half*>(base + Stage::KScaleOffset);
        __half* vs_s = ks_s + Bc * Groups;
        const int page = block_table[kb];
        const int valid = min(Bc, max_query_abs + 1 - kb * Bc);
        const std::int64_t code_base =
            kv_cache_int8_quant_code_index<Geometry>(page, kv_head, 0, 0);
#pragma unroll
        for (int i = 0; i < kCausalPromptI8FastTileBytes / 16 / Threads; ++i) {
            const int chunk = tid + i * Threads;
            const int key = chunk >> 4;
            const int c = chunk & 15;
            const int dst = key * D + ((c ^ (key & 7)) << 4);
            const int bytes = key < valid ? 16 : 0;
            const std::int64_t src = code_base + key * D + c * 16;
            cp_async_zfill<16, Cache::cg>(k_s + dst, cache_k + src, bytes);
            cp_async_zfill<16, Cache::cg>(v_s + dst, cache_v + src, bytes);
        }
        if (tid < Bc) {
            cp_async_zfill<8>(
                ks_s + tid * Groups,
                cache_k_scale + kv_cache_int8_quant_scale_index<Geometry>(
                                    page, kv_head, 0, tid),
                tid < valid ? 8 : 0);
        } else if (tid < 2 * Bc) {
            const int key = tid - Bc;
            cp_async_zfill<8>(
                vs_s + key * Groups,
                cache_v_scale + kv_cache_int8_quant_scale_index<Geometry>(
                                    page, kv_head, 0, key),
                key < valid ? 8 : 0);
        }
        cp_commit();
    };

    const int a_mat = lane >> 3;
    const int a_rin = lane & 7;
    const int a_rowoff = a_rin + ((a_mat & 1) << 3);
    const int a_coloff = (a_mat >> 1) << 3;
    const int b_koff = ((lane >> 3) & 1) << 3;
    const bool warp_active = row_base < rows;
    const int warp_min_qabs = base_pos + q0 + row_base;
    const int warp_max_qabs = base_pos + q0 + min(row_base + 15, rows - 1);
    const int row0 = row_base + gid;
    const int row1 = row0 + 8;
    const int qabs0 = row0 < rows ? base_pos + q0 + row0 : -1;
    const int qabs1 = row1 < rows ? base_pos + q0 + row1 : -1;
    float acc[DBlocks][2][4];
#pragma unroll
    for (int b = 0; b < DBlocks; ++b) {
#pragma unroll
        for (int p = 0; p < 2; ++p) {
#pragma unroll
            for (int i = 0; i < 4; ++i) { acc[b][p][i] = 0.0f; }
        }
    }
    float running_m0 = -CUDART_INF_F;
    float running_m1 = -CUDART_INF_F;
    float running_l0 = 0.0f;
    float running_l1 = 0.0f;
    const float scale_l2 = scale * Log2E;

    issue_tile(0);
#pragma unroll 1
    for (int kb = 0; kb < key_blocks; ++kb) {
        cp_wait<0>();
        __syncthreads();
        if (kb + 1 < key_blocks) { issue_tile(kb + 1); }
        const int k0 = kb * Bc;
        const bool tile_live = warp_active && k0 <= warp_max_qabs;
        unsigned char* base = stage_base(kb & 1);
        const __nv_bfloat16* k_b16 = reinterpret_cast<const __nv_bfloat16*>(base);
        const __half* ks_s = reinterpret_cast<const __half*>(base + Stage::KScaleOffset);
        const __half* vs_s = ks_s + Bc * Groups;
        unsigned* p_s = reinterpret_cast<unsigned*>(base);
        int* shift_s = reinterpret_cast<int*>(base + Stage::KScaleOffset);
        float tile_alpha0 = 0.0f;
        float tile_alpha1 = 0.0f;
        int tile_shift = 0;
        {
            float score[QKNt][4];
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                score[nt][0] = score[nt][1] = score[nt][2] = score[nt][3] = 0.0f;
            }
            float bm0 = -CUDART_INF_F;
            float bm1 = -CUDART_INF_F;
            if (tile_live) {
@QK_GROUP@
@QK_MAX@
            }
            if (lid == 0) {
                max_s[side * Br + row0] = bm0;
                max_s[side * Br + row1] = bm1;
            }
            __syncthreads();
            unsigned pa_local[2][4];
            float high_pair_sum[QKNt][2];
            if (tile_live) {
                bm0 = fmaxf(bm0, max_s[(1 - side) * Br + row0]);
                bm1 = fmaxf(bm1, max_s[(1 - side) * Br + row1]);
@ONLINE_MAX@
                float bl0 = 0.0f;
                float bl1 = 0.0f;
@PROBABILITIES@
                if (side == 0) {
#pragma unroll
                    for (int j = 0; j < 2; ++j) {
                        const uint4 packed = make_uint4(pa_local[j][0], pa_local[j][1],
                                                       pa_local[j][2], pa_local[j][3]);
                        store_vec(p_s + ((pair * 2) * 2 + j) * 128 + lane * 4, packed);
                    }
                    float* carry = reinterpret_cast<float*>(p_s + (pair * 2 + 1) * 256);
                    carry[lane * 2 + 0] = bl0;
                    carry[lane * 2 + 1] = bl1;
@TILE_SHIFT@
                    if (lane == 0) { shift_s[pair] = tile_shift; }
                }
            }
            __syncthreads();
            if (tile_live && side == 1) {
                const float* carry =
                    reinterpret_cast<const float*>(p_s + (pair * 2 + 1) * 256);
                float bl0 = carry[lane * 2 + 0];
                float bl1 = carry[lane * 2 + 1];
#pragma unroll
                for (int nt = 0; nt < QKNt; ++nt) {
                    bl0 += high_pair_sum[nt][0];
                    bl1 += high_pair_sum[nt][1];
                }
@DENOMINATOR_UPDATE@
                tile_shift = shift_s[pair];
                __syncwarp(FullMask);
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    const uint4 packed = make_uint4(pa_local[j][0], pa_local[j][1],
                                                   pa_local[j][2], pa_local[j][3]);
                    store_vec(p_s + ((pair * 2 + 1) * 2 + j) * 128 + lane * 4, packed);
                }
            }
        }
        __syncthreads();
        if (tile_live) {
            unsigned pa[PVKs][4];
#pragma unroll
            for (int j = 0; j < PVKs; ++j) {
                const uint4 packed = load_vec<uint4>(
                    p_s + ((pair * 2 + (j >> 1)) * 2 + (j & 1)) * 128 + lane * 4);
                pa[j][0] = packed.x;
                pa[j][1] = packed.y;
                pa[j][2] = packed.z;
                pa[j][3] = packed.w;
            }
            const std::int8_t* v_s =
                reinterpret_cast<const std::int8_t*>(base) + kCausalPromptI8FastTileBytes;
            const __half2 mul = __float2half2_rn(ldexpf(1.0f, -tile_shift));
            const float unscale = ldexpf(1.0f, tile_shift);
@PV_BODY@
        }
    }
    if (side == 1) {
@FINAL_DENOMINATOR@
        if (lid == 0) {
            max_s[row0] = inv_l0;
            max_s[row1] = inv_l1;
        }
    }
    __syncthreads();
    const float inv_l0 = max_s[row0];
    const float inv_l1 = max_s[row1];
#pragma unroll
    for (int b = 0; b < DBlocks; ++b) {
        const int d0 = side * 128 + b * 16 + 4 * lid;
        store_row(row0, d0, acc[b][0][0] * inv_l0, acc[b][1][0] * inv_l0,
                  acc[b][0][1] * inv_l0, acc[b][1][1] * inv_l0);
        store_row(row1, d0, acc[b][0][2] * inv_l1, acc[b][1][2] * inv_l1,
                  acc[b][0][3] * inv_l1, acc[b][1][3] * inv_l1);
    }
}
