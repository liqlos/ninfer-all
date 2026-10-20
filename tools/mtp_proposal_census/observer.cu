#include "observer.h"
#include "ops/kernel/sampling_device.cuh"

#include <cuda_bf16.h>
#include <climits>

namespace ninfer::lab::mtp_census {
namespace {

constexpr int kThreads = 256;

__device__ void fail(State* s, Error error) {
    atomicCAS(&s->error, std::uint32_t(None), std::uint32_t(error));
}

__device__ bool enabled(const State* s) {
    return s->active != 0 && s->error == None && s->completed < kRecords;
}

__device__ bool token(int id) { return id >= 0 && id < kDomain; }

__device__ bool sampling_ok(const State* s, const ops::SamplingConfig& c) {
    return c.temperature == 0.0F &&
           c.top_k == 1 && c.token_mask == nullptr && c.token_mask_stride == 0 &&
           c.presence_penalty == 1.5F && c.frequency_penalty == 0.0F &&
           c.token_counts == s->counts && c.token_counts != nullptr;
}

__device__ Score score(const __nv_bfloat16* logits, int id, const ops::SamplingConfig& c,
                      const std::int32_t* prefix, int length) {
    const float raw = __bfloat162float(logits[id]);
    int overlay = 0;
    for (int j = 0; j < length; ++j) { overlay += prefix[j] == id; }
    return {id, __float_as_uint(raw),
            __float_as_uint(ops::sampling_adjusted_logit(raw, id, c, prefix, length)),
            c.token_counts == nullptr ? 0 : c.token_counts[id], overlay};
}

struct Pair {
    float value[2];
    int id[2];
};

struct Reduction {
    Pair raw;
    Pair adjusted;
    int finite;
};

__device__ Pair empty_pair() {
    return {{-CUDART_INF_F, -CUDART_INF_F}, {INT_MAX, INT_MAX}};
}

__device__ void insert(Pair& p, float value, int id) {
    if (id == INT_MAX || id == p.id[0] || id == p.id[1]) { return; }
    if (ops::sampling_better(value, id, p.value[0], p.id[0])) {
        p.value[1] = p.value[0];
        p.id[1] = p.id[0];
        p.value[0] = value;
        p.id[0] = id;
    } else if (ops::sampling_better(value, id, p.value[1], p.id[1])) {
        p.value[1] = value;
        p.id[1] = id;
    }
}

__device__ Reduction reduce(const __nv_bfloat16* logits, const ops::SamplingConfig& c,
                            const std::int32_t* prefix, int length, Reduction* shared) {
    Reduction local{empty_pair(), empty_pair(), 1};
    for (int id = threadIdx.x; id < kDomain; id += kThreads) {
        const float raw = __bfloat162float(logits[id]);
        const float adjusted = ops::sampling_adjusted_logit(raw, id, c, prefix, length);
        if (!ops::sampling_value_is_finite(raw) ||
            !ops::sampling_value_is_finite(adjusted)) {
            local.finite = 0;
            continue;
        }
        insert(local.raw, raw, id);
        insert(local.adjusted, adjusted, id);
    }
    shared[threadIdx.x] = local;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            const Reduction other = shared[threadIdx.x + stride];
            Reduction& own = shared[threadIdx.x];
            for (int j = 0; j < 2; ++j) {
                insert(own.raw, other.raw.value[j], other.raw.id[j]);
                insert(own.adjusted, other.adjusted.value[j], other.adjusted.id[j]);
            }
            own.finite &= other.finite;
        }
        __syncthreads();
    }
    return shared[0];
}

__device__ Top2 top2(const Reduction& r, const __nv_bfloat16* logits,
                     const ops::SamplingConfig& c, const std::int32_t* prefix, int length) {
    Top2 out{};
    for (int j = 0; j < 2; ++j) {
        out.raw[j] = score(logits, r.raw.id[j], c, prefix, length);
        out.adjusted[j] = score(logits, r.adjusted.id[j], c, prefix, length);
    }
    return out;
}

__global__ void census_reset(State* s, int active, const std::int32_t* counts) {
    __shared__ std::uint32_t epoch;
    if (threadIdx.x == 0) { epoch = s->epoch + 1; }
    __syncthreads();
    auto* bytes = reinterpret_cast<unsigned char*>(s);
    for (unsigned int i = threadIdx.x; i < sizeof(State); i += blockDim.x) { bytes[i] = 0; }
    __syncthreads();
    if (threadIdx.x == 0) {
        s->epoch = epoch;
        s->active = active;
        s->slot = -1;
        s->counts = counts;
    }
}

__global__ void census_begin(State* s, View v) {
    if (!enabled(s)) { return; }
    s->slot = -1;
    ++s->round;
    const ops::SamplingConfig c = *v.sampling;
    if (!sampling_ok(s, c)) { fail(s, Sampling); return; }
    const int extent = *v.extent;
    if (extent < 0 || extent > kDrafts || *v.valid_columns != extent + 1 ||
        *v.budget < extent + 1 || *v.frontier < 0 || !token(*v.anchor)) {
        fail(s, Extent);
        return;
    }
    if (s->round == 1) {
        s->sampling = c;
        s->first_frontier = *v.frontier;
        s->first_anchor = *v.anchor;
        s->first_budget = *v.budget;
        return;
    }
    if (extent == 0) { return; }
    const Pending& q = s->pending;
    if (!q.valid || q.epoch != s->epoch || q.round + 1 != s->round ||
        q.frontier != *v.frontier || q.anchor != *v.anchor || extent > q.extent) {
        fail(s, Join);
        return;
    }
    for (int j = 0; j < extent; ++j) {
        if (!token(v.drafts[j]) || v.drafts[j] != q.q[j].actual) {
            fail(s, Join);
            return;
        }
    }
    s->slot = static_cast<int>(s->completed);
    Record& r = s->records[s->slot];
    r.epoch = s->epoch;
    r.round = s->round;
    r.producer_round = q.round;
    r.q_epoch = q.epoch;
    r.q_frontier = q.frontier;
    r.q_anchor = q.anchor;
    r.q_extent = q.extent;
    r.frontier = *v.frontier;
    r.anchor = *v.anchor;
    r.extent = extent;
    for (int j = 0; j < extent; ++j) {
        r.drafts[j] = v.drafts[j];
        r.q[j] = q.q[j];
    }
}

__global__ void census_target(State* s, const __nv_bfloat16* logits,
                               const std::int32_t* actual, View v) {
    __shared__ int run;
    if (threadIdx.x == 0) { run = enabled(s) && s->slot >= 0; }
    __syncthreads();
    if (!run) { return; }
    Record& r = s->records[s->slot];
    const int col = blockIdx.x;
    if (col > r.extent) { return; }
    const ops::SamplingConfig c = *v.sampling;
    const auto* row = logits + col * kPhysical;
    __shared__ Reduction shared[kThreads];
    const Reduction result = reduce(row, c, v.drafts, col, shared);
    if (threadIdx.x != 0) { return; }
    if (!result.finite) { fail(s, NonFinite); return; }
    if (actual[col] != result.raw.id[0]) { fail(s, RawWinner); return; }
    Target& p = r.p[col];
    p.top = top2(result, row, c, v.drafts, col);
    p.actual_raw = actual[col];
    if (col >= r.extent) { return; }
    const Proposal& q = r.q[col];
    p.candidate_count = q.candidate_count;
    for (int j = 0; j < q.candidate_count; ++j) {
        p.candidates[j] = score(row, q.candidates[j].id, c, v.drafts, col);
        if (p.candidates[j].base_count != q.candidates[j].base_count ||
            p.candidates[j].prefix_count != q.candidates[j].prefix_count) {
            fail(s, CountJoin);
        }
    }
}

__device__ bool updated_count(const Score& old, const Record& r,
                             const ops::SamplingConfig& c) {
    if (c.token_counts == nullptr) { return old.base_count == 0; }
    int increment = 0;
    for (int j = 0; j < r.licensed_count; ++j) { increment += r.licensed[j] == old.id; }
    return c.token_counts[old.id] == old.base_count + increment;
}

__global__ void census_accept(State* s, View v) {
    if (!enabled(s) || s->slot < 0) { return; }
    Record& r = s->records[s->slot];
    int expected = 0;
    while (expected < r.extent && r.drafts[expected] == r.p[expected].top.adjusted[0].id) {
        ++expected;
    }
    r.accepted = *v.accepted;
    r.licensed_count = *v.licensed_count;
    r.post_frontier = *v.frontier;
    r.post_anchor = *v.anchor;
    if (r.accepted != expected || r.licensed_count != expected + 1 ||
        r.post_frontier != r.frontier + r.licensed_count ||
        r.post_anchor != r.p[expected].top.adjusted[0].id) {
        fail(s, Acceptance);
        return;
    }
    for (int j = 0; j < r.licensed_count; ++j) {
        r.licensed[j] = v.licensed[j];
        const int wanted = j < expected ? r.drafts[j] : r.p[j].top.adjusted[0].id;
        if (r.licensed[j] != wanted) { fail(s, Acceptance); return; }
    }
    const ops::SamplingConfig c = *v.sampling;
    for (int col = 0; col <= r.extent; ++col) {
        for (int j = 0; j < 2; ++j) {
            if (!updated_count(r.p[col].top.raw[j], r, c) ||
                !updated_count(r.p[col].top.adjusted[j], r, c)) {
                fail(s, CountUpdate);
                return;
            }
        }
        for (int j = 0; j < r.p[col].candidate_count; ++j) {
            if (!updated_count(r.p[col].candidates[j], r, c)) {
                fail(s, CountUpdate);
                return;
            }
        }
    }
    r.causal_columns = (1U << r.licensed_count) - 1U;
    ++s->completed;
}

__global__ void census_proposal(State* s, const __nv_bfloat16* logits,
                                 const std::int32_t* actual, int step, View v) {
    __shared__ int run;
    if (threadIdx.x == 0) { run = enabled(s); }
    __syncthreads();
    if (!run) { return; }
    Pending& q = s->pending;
    if (threadIdx.x == 0 && step == 0) {
        q.valid = 0;
        q.mask = 0;
        q.epoch = s->epoch;
        q.round = s->round;
        q.frontier = *v.frontier;
        q.anchor = *v.anchor;
        q.extent = *v.next_extent;
        if (q.extent < 0 || q.extent > kDrafts || s->round == 0) { fail(s, Extent); }
    }
    __syncthreads();
    if (s->error != None) { return; }
    const int prefix = step < q.extent ? step : q.extent;
    if (q.round != s->round || q.mask != ((1U << prefix) - 1U)) {
        if (threadIdx.x == 0) { fail(s, ProposalOrder); }
        return;
    }
    if (step < q.extent) {
        const ops::SamplingConfig c = *v.sampling;
        __shared__ Reduction shared[kThreads];
        const Reduction result = reduce(logits, c, v.next_drafts, step, shared);
        if (threadIdx.x == 0) {
            if (!result.finite) { fail(s, NonFinite); }
            else if (*actual != result.raw.id[0]) { fail(s, RawWinner); }
            else {
                Proposal& out = q.q[step];
                out.top = top2(result, logits, c, v.next_drafts, step);
                out.actual = *actual;
                out.candidate_count = 0;
                const int ids[kCandidates] = {result.raw.id[0], result.raw.id[1],
                                              result.adjusted.id[0], result.adjusted.id[1]};
                for (int id : ids) {
                    bool found = false;
                    for (int j = 0; j < out.candidate_count; ++j) {
                        found |= out.candidates[j].id == id;
                    }
                    if (!found) {
                        out.candidates[out.candidate_count++] =
                            score(logits, id, c, v.next_drafts, step);
                    }
                }
                q.mask |= 1U << step;
            }
        }
    }
    __syncthreads();
    if (threadIdx.x == 0 && step == kDrafts - 1 && s->error == None) {
        q.valid = q.extent > 0 && q.mask == ((1U << q.extent) - 1U);
    }
}

}

cudaError_t reset(State* s, bool active, const std::int32_t* counts, cudaStream_t stream) {
    census_reset<<<1, kThreads, 0, stream>>>(s, active, counts);
    return cudaGetLastError();
}

cudaError_t before_accept(State* s, const void* logits, const std::int32_t* actual,
                         View view, cudaStream_t stream) {
    census_begin<<<1, 1, 0, stream>>>(s, view);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) { return error; }
    census_target<<<kWidth, kThreads, 0, stream>>>(
        s, static_cast<const __nv_bfloat16*>(logits), actual, view);
    return cudaGetLastError();
}

cudaError_t after_accept(State* s, View view, cudaStream_t stream) {
    census_accept<<<1, 1, 0, stream>>>(s, view);
    return cudaGetLastError();
}

cudaError_t proposal(State* s, const void* logits, const std::int32_t* actual, int step,
                    View view, cudaStream_t stream) {
    census_proposal<<<1, kThreads, 0, stream>>>(
        s, static_cast<const __nv_bfloat16*>(logits), actual, step, view);
    return cudaGetLastError();
}

}
