#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <future>
#include <memory>
#include <new>
#include <span>
#include <stdexcept>

namespace weight_read_ahead_probe {

using Clock = std::chrono::steady_clock;
using Read = std::size_t (*)(const void*, std::uint64_t, std::span<std::byte>);

struct Stats {
    std::uint64_t calls = 0;
    std::uint64_t files = 0;
    std::uint64_t native_calls = 0;
    std::uint64_t native_bytes = 0;
    std::uint64_t hits = 0;
    std::uint64_t copied_bytes = 0;
    std::uint64_t launched = 0;
    std::uint64_t async_bytes = 0;
    std::uint64_t discarded_jobs = 0;
    std::uint64_t discarded_bytes = 0;
    std::uint64_t requested_errors = 0;
    std::uint64_t discarded_errors = 0;
    std::uint64_t setup_failures = 0;
    std::uint64_t peak_jobs = 0;
    std::uint64_t peak_buffer_bytes = 0;
    double native_seconds = 0;
    double async_seconds_sum = 0;
    double wait_seconds = 0;
    double copy_seconds = 0;
    double drain_seconds = 0;
    double setup_seconds = 0;
};

class Queue {
public:
    static constexpr std::size_t maximum_slots = 4;
    static constexpr std::size_t default_slot_bytes = 64ULL * 1024 * 1024;
    static constexpr std::size_t alignment = 4096;

    explicit Queue(Read read, std::size_t slot_bytes = default_slot_bytes)
        : read_(read), slot_bytes_(slot_bytes) {
        if (read == nullptr || slot_bytes == 0 || slot_bytes > default_slot_bytes ||
            slot_bytes % alignment != 0) {
            throw std::invalid_argument("invalid bounded read-ahead geometry");
        }
    }

    Queue(const Queue&) = delete;
    Queue& operator=(const Queue&) = delete;
    ~Queue() { retire(); }

    const Stats& stats() const noexcept { return stats_; }
    const void* owner() const noexcept { return owner_; }

    void retire() noexcept {
        const auto started = Clock::now();
        for (auto& slot : slots_) {
            if (slot && slot->job.valid()) {
                const Result result = collect(*slot);
                ++stats_.discarded_jobs;
                stats_.discarded_bytes += result.received;
                stats_.discarded_errors += result.error ? 1U : 0U;
            }
        }
        stats_.drain_seconds += seconds(started);
        previous_full_ = false;
        pending_ = 0;
        owner_ = nullptr;
        for (auto& slot : slots_) { slot.reset(); }
    }

    std::size_t read(const void* owner, std::uint64_t file_bytes, std::uint64_t offset,
                     std::span<std::byte> destination) {
        ++stats_.calls;
        if (owner_ != owner) {
            retire();
            owner_ = owner;
            file_bytes_ = file_bytes;
            ++stats_.files;
        }
        const bool eligible = !disabled_ && destination.size() == slot_bytes_ &&
                              destination.data() != nullptr &&
                              offset % alignment == 0 &&
                              reinterpret_cast<std::uintptr_t>(destination.data()) % alignment == 0 &&
                              offset <= file_bytes_ && slot_bytes_ <= file_bytes_ - offset;
        if (eligible && pending_ != 0 && offset == previous_end_) {
            for (auto& slot : slots_) {
                if (!slot || !slot->job.valid() || slot->offset != offset) { continue; }
                const auto waiting = Clock::now();
                const Result result = collect(*slot);
                stats_.wait_seconds += seconds(waiting);
                if (result.error) {
                    ++stats_.requested_errors;
                    retire();
                    std::rethrow_exception(result.error);
                }
                const auto copying = Clock::now();
                std::memcpy(destination.data(), slot->buffer.data, result.received);
                stats_.copy_seconds += seconds(copying);
                ++stats_.hits;
                stats_.copied_bytes += result.received;
                previous_end_ = offset + result.received;
                previous_full_ = result.received == slot_bytes_;
                if (!previous_full_) {
                    retire();
                } else {
                    try {
                        schedule(*slot);
                    } catch (...) {
                        ++stats_.setup_failures;
                        disabled_ = true;
                        retire();
                    }
                }
                return result.received;
            }
        }
        const bool consecutive = eligible && previous_full_ && offset == previous_end_;
        if (pending_ != 0 || !eligible) {
            retire();
            owner_ = owner;
            file_bytes_ = file_bytes;
        }
        const auto started = Clock::now();
        ++stats_.native_calls;
        std::size_t received = 0;
        try {
            received = read_(owner, offset, destination);
        } catch (...) {
            stats_.native_seconds += seconds(started);
            ++stats_.requested_errors;
            previous_full_ = false;
            throw;
        }
        stats_.native_seconds += seconds(started);
        stats_.native_bytes += received;
        previous_full_ = eligible && received == slot_bytes_;
        previous_end_ = previous_full_ ? offset + slot_bytes_ : 0;
        if (consecutive && previous_full_ && previous_end_ <= file_bytes_ &&
            file_bytes_ - previous_end_ >= 2 * slot_bytes_) {
            const auto setup = Clock::now();
            try {
                for (auto& slot : slots_) {
                    if (!slot) { slot = std::make_unique<Slot>(slot_bytes_); }
                }
                stats_.peak_buffer_bytes = maximum_slots * slot_bytes_;
                next_offset_ = previous_end_;
                for (auto& slot : slots_) { schedule(*slot); }
            } catch (...) {
                ++stats_.setup_failures;
                disabled_ = true;
                retire();
            }
            stats_.setup_seconds += seconds(setup);
        }
        return received;
    }

private:
    struct Buffer {
        std::byte* data = nullptr;
        explicit Buffer(std::size_t bytes) {
            void* pointer = nullptr;
            if (::posix_memalign(&pointer, alignment, bytes) != 0) { throw std::bad_alloc(); }
            data = static_cast<std::byte*>(pointer);
        }
        ~Buffer() { std::free(data); }
        Buffer(const Buffer&) = delete;
        Buffer& operator=(const Buffer&) = delete;
    };

    struct Result {
        std::size_t received = 0;
        double seconds = 0;
        std::exception_ptr error;
    };

    struct Slot {
        explicit Slot(std::size_t bytes) : buffer(bytes) {}
        Buffer buffer;
        std::uint64_t offset = 0;
        std::future<Result> job;
    };

    static double seconds(Clock::time_point started) noexcept {
        return std::chrono::duration<double>(Clock::now() - started).count();
    }

    Result collect(Slot& slot) noexcept {
        Result result;
        try {
            result = slot.job.get();
        } catch (...) {
            result.error = std::current_exception();
        }
        if (pending_ != 0) { --pending_; }
        stats_.async_bytes += result.received;
        stats_.async_seconds_sum += result.seconds;
        return result;
    }

    void schedule(Slot& slot) {
        if (next_offset_ > file_bytes_ || slot_bytes_ > file_bytes_ - next_offset_) { return; }
        const void* owner = owner_;
        const Read native = read_;
        const std::uint64_t offset = next_offset_;
        const std::size_t count = slot_bytes_;
        std::byte* destination = slot.buffer.data;
        slot.offset = offset;
        slot.job = std::async(std::launch::async, [owner, native, offset, count, destination] {
            Result result;
            const auto started = Clock::now();
            try {
                result.received = native(owner, offset, {destination, count});
                if (result.received > count) {
                    throw std::logic_error("native direct read exceeded destination");
                }
            } catch (...) {
                result.received = 0;
                result.error = std::current_exception();
            }
            result.seconds = seconds(started);
            return result;
        });
        next_offset_ += slot_bytes_;
        ++pending_;
        ++stats_.launched;
        if (pending_ > stats_.peak_jobs) { stats_.peak_jobs = pending_; }
    }

    Read read_;
    const std::size_t slot_bytes_;
    Stats stats_;
    const void* owner_ = nullptr;
    std::uint64_t file_bytes_ = 0;
    std::uint64_t previous_end_ = 0;
    std::uint64_t next_offset_ = 0;
    std::uint64_t pending_ = 0;
    bool previous_full_ = false;
    bool disabled_ = false;
    std::array<std::unique_ptr<Slot>, maximum_slots> slots_;
};

}
