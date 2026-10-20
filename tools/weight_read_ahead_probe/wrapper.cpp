#include "artifact/file_io.h"
#include "artifact/materializer.h"

#include <cstdio>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>

#ifndef NINFER_LAB_WEIGHT_READ_AHEAD
#define NINFER_LAB_WEIGHT_READ_AHEAD 0
#endif

#if NINFER_LAB_WEIGHT_READ_AHEAD != 0 && NINFER_LAB_WEIGHT_READ_AHEAD != 1
#error NINFER_LAB_WEIGHT_READ_AHEAD_must_be_zero_or_one
#endif

#define PROBE_READ "_ZNK6ninfer8artifact9InputFile11read_directEmSt4spanISt4byteLm18446744073709551615EE"
#define PROBE_DESTROY "_ZN6ninfer8artifact9InputFileD1Ev"
#define PROBE_DESTROY_BASE "_ZN6ninfer8artifact9InputFileD2Ev"
#define PROBE_MATERIALIZE "_ZN6ninfer8artifact11materializeERKNS0_6ReaderEONS0_19MaterializationPlanERNS_13DeviceContextEPKNS_15StartupObserverESt10unique_ptrINS_19EvictableWeightPoolESt14default_deleteISC_EE"

namespace na = ninfer::artifact;

extern "C" std::size_t real_read(const na::InputFile*, std::uint64_t,
    std::span<std::byte>) asm("__real_" PROBE_READ);
extern "C" std::size_t wrap_read(const na::InputFile*, std::uint64_t,
    std::span<std::byte>) asm("__wrap_" PROBE_READ);
extern "C" void real_destroy(na::InputFile*) noexcept asm("__real_" PROBE_DESTROY);
extern "C" void wrap_destroy(na::InputFile*) noexcept asm("__wrap_" PROBE_DESTROY);
extern "C" void real_destroy_base(na::InputFile*) noexcept asm("__real_" PROBE_DESTROY_BASE);
extern "C" void wrap_destroy_base(na::InputFile*) noexcept asm("__wrap_" PROBE_DESTROY_BASE);
extern "C" na::MaterializedArtifact real_materialize(const na::Reader&,
    na::MaterializationPlan&&, ninfer::DeviceContext&, const ninfer::StartupObserver*,
    std::unique_ptr<ninfer::EvictableWeightPool>) asm("__real_" PROBE_MATERIALIZE);
extern "C" na::MaterializedArtifact wrap_materialize(const na::Reader&,
    na::MaterializationPlan&&, ninfer::DeviceContext&, const ninfer::StartupObserver*,
    std::unique_ptr<ninfer::EvictableWeightPool>) asm("__wrap_" PROBE_MATERIALIZE);

#if NINFER_LAB_WEIGHT_READ_AHEAD
#include "read_ahead.h"

namespace {

using Queue = weight_read_ahead_probe::Queue;
using Clock = weight_read_ahead_probe::Clock;

std::size_t native_read(const void* file, std::uint64_t offset,
                        std::span<std::byte> destination) {
    return real_read(static_cast<const na::InputFile*>(file), offset, destination);
}

struct State {
    std::mutex mutex;
    std::thread::id owner;
    std::unique_ptr<Queue> queue;
};

State& state() {
    static State value;
    return value;
}

class LoadScope {
public:
    LoadScope() : started_(Clock::now()) {
        auto& value = state();
        std::lock_guard lock(value.mutex);
        if (value.queue) { return; }
        try {
            value.queue = std::make_unique<Queue>(native_read);
            value.owner = std::this_thread::get_id();
            active_ = true;
        } catch (...) {}
    }

    ~LoadScope() { finish(true); }

    void finish(bool failed) noexcept {
        if (!active_) { return; }
        auto& value = state();
        std::lock_guard lock(value.mutex);
        value.queue->retire();
        const auto s = value.queue->stats();
        value.queue.reset();
        value.owner = {};
        active_ = false;
        const auto elapsed = std::chrono::duration<double>(Clock::now() - started_).count();
        std::fprintf(stderr,
            "NINFER_WEIGHT_READ_AHEAD {\"failed\":%s,\"scope_seconds\":%.9f,"
            "\"calls\":%llu,\"files\":%llu,\"native_calls\":%llu,\"native_bytes\":%llu,"
            "\"hits\":%llu,\"copied_bytes\":%llu,\"launched\":%llu,\"async_bytes\":%llu,"
            "\"discarded_jobs\":%llu,\"discarded_bytes\":%llu,\"requested_errors\":%llu,"
            "\"discarded_errors\":%llu,\"setup_failures\":%llu,\"peak_jobs\":%llu,"
            "\"peak_buffer_bytes\":%llu,\"native_seconds\":%.9f,"
            "\"async_seconds_sum\":%.9f,\"wait_seconds\":%.9f,\"copy_seconds\":%.9f,"
            "\"drain_seconds\":%.9f,\"setup_seconds\":%.9f}\n",
            failed ? "true" : "false", elapsed,
            static_cast<unsigned long long>(s.calls), static_cast<unsigned long long>(s.files),
            static_cast<unsigned long long>(s.native_calls), static_cast<unsigned long long>(s.native_bytes),
            static_cast<unsigned long long>(s.hits), static_cast<unsigned long long>(s.copied_bytes),
            static_cast<unsigned long long>(s.launched), static_cast<unsigned long long>(s.async_bytes),
            static_cast<unsigned long long>(s.discarded_jobs), static_cast<unsigned long long>(s.discarded_bytes),
            static_cast<unsigned long long>(s.requested_errors), static_cast<unsigned long long>(s.discarded_errors),
            static_cast<unsigned long long>(s.setup_failures), static_cast<unsigned long long>(s.peak_jobs),
            static_cast<unsigned long long>(s.peak_buffer_bytes), s.native_seconds,
            s.async_seconds_sum, s.wait_seconds, s.copy_seconds, s.drain_seconds, s.setup_seconds);
    }

private:
    Clock::time_point started_;
    bool active_ = false;
};

void retire_file(const na::InputFile* file) noexcept {
    auto& value = state();
    std::lock_guard lock(value.mutex);
    if (value.queue && value.queue->owner() == file) { value.queue->retire(); }
}

}
#endif

extern "C" std::size_t wrap_read(const na::InputFile* file, std::uint64_t offset,
                                  std::span<std::byte> destination) {
#if NINFER_LAB_WEIGHT_READ_AHEAD
    auto& value = state();
    std::lock_guard lock(value.mutex);
    if (value.queue && value.owner == std::this_thread::get_id()) {
        return value.queue->read(file, file->bytes(), offset, destination);
    }
#endif
    return real_read(file, offset, destination);
}

extern "C" void wrap_destroy(na::InputFile* file) noexcept {
#if NINFER_LAB_WEIGHT_READ_AHEAD
    retire_file(file);
#endif
    real_destroy(file);
}

extern "C" void wrap_destroy_base(na::InputFile* file) noexcept {
#if NINFER_LAB_WEIGHT_READ_AHEAD
    retire_file(file);
#endif
    real_destroy_base(file);
}

extern "C" na::MaterializedArtifact wrap_materialize(const na::Reader& reader,
    na::MaterializationPlan&& plan, ninfer::DeviceContext& device,
    const ninfer::StartupObserver* observer, std::unique_ptr<ninfer::EvictableWeightPool> backing) {
#if NINFER_LAB_WEIGHT_READ_AHEAD
    LoadScope scope;
    auto result = real_materialize(reader, std::move(plan), device, observer, std::move(backing));
    scope.finish(false);
    return result;
#else
    return real_materialize(reader, std::move(plan), device, observer, std::move(backing));
#endif
}
