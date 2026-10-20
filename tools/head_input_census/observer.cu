#include "observer.h"

namespace ninfer::lab::head_census {
namespace {

__global__ void capture(const mtp_census::State* s, Payload* output, int role,
                         const std::uint8_t* activation, const std::uint16_t* logits,
                         const void* weight, mtp_census::View v) {
    if (!s->active || s->error != mtp_census::None || s->completed >= mtp_census::kRecords) {
        return;
    }
    const bool target = role == 3;
    const int slot = static_cast<int>(s->round) - 1;
    if (slot < 0 || slot >= mtp_census::kRecords) { return; }
    const int columns = target ? 4 : 1;
    const int at = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = gridDim.x * blockDim.x;
    for (int column = 0; column < columns; ++column) {
        Sample& sample = output->samples[slot][role + column];
        if (at == 0) {
            sample.epoch = s->epoch;
            sample.round = s->round + static_cast<unsigned>(target);
            sample.role = role + column;
            sample.frontier = *v.frontier;
            sample.anchor = *v.anchor;
            sample.extent = target ? *v.extent : *v.next_extent;
            sample.weight_address = reinterpret_cast<std::uint64_t>(weight);
        }
        for (int i = at; i < kK; i += stride) {
            sample.packed[i] = activation[column * kK + i];
        }
        for (int i = at; i < kPacked - kK; i += stride) {
            sample.packed[kK + i] = activation[columns * kK + column * (kPacked - kK) + i];
        }
        for (int i = at; i < mtp_census::kPhysical; i += stride) {
            sample.logits[i] = logits[column * mtp_census::kPhysical + i];
        }
    }
    if (target && slot == 0) {
        const auto* counts = v.sampling->token_counts;
        if (counts == nullptr || counts != s->counts) { return; }
        if (at == 0) { output->counts_epoch = s->epoch; }
        for (int i = at; i < mtp_census::kDomain; i += stride) {
            output->counts[i] = counts[i];
        }
    }
}

}

cudaError_t snapshot(const mtp_census::State* s, Payload* output, int role,
                     const void* activation, const void* logits, const void* weight,
                     mtp_census::View view, cudaStream_t stream) {
    capture<<<128, 256, 0, stream>>>(s, output, role,
        static_cast<const std::uint8_t*>(activation),
        static_cast<const std::uint16_t*>(logits), weight, view);
    return cudaGetLastError();
}

}
