#include "read_ahead.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <limits>
#include <string>
#include <thread>

namespace {

using weight_read_ahead_probe::Queue;
constexpr std::size_t page = Queue::alignment;

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

struct Buffer {
    std::byte* data = nullptr;
    Buffer() {
        void* pointer = nullptr;
        if (::posix_memalign(&pointer, page, 2 * page) != 0) { throw std::bad_alloc(); }
        data = static_cast<std::byte*>(pointer);
    }
    ~Buffer() { std::free(data); }
    std::span<std::byte> span() { return {data, page}; }
};

struct File {
    std::uint64_t bytes = 16 * page + 37;
    std::uint64_t fail_offset = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t short_offset = std::numeric_limits<std::uint64_t>::max();
    std::thread::id owner = std::this_thread::get_id();
    mutable std::atomic<bool> initialized{false};
    mutable std::atomic<bool> closed{false};
    mutable std::atomic<int> active{0};
    mutable std::atomic<int> peak{0};
    mutable std::atomic<int> calls{0};
    mutable std::atomic<int> async_calls{0};
};

std::byte represented(std::uint64_t offset) {
    return static_cast<std::byte>((offset * 37U + offset / page * 13U + 19U) & 255U);
}

std::size_t native(const void* input, std::uint64_t offset, std::span<std::byte> destination) {
    const auto& file = *static_cast<const File*>(input);
    require(!file.closed.load(), "read after close");
    require(offset % page == 0 && destination.size() % page == 0 &&
            reinterpret_cast<std::uintptr_t>(destination.data()) % page == 0,
            "native alignment error");
    if (destination.empty()) { return 0; }
    require(destination.data() != nullptr, "native destination error");
    const bool worker = std::this_thread::get_id() != file.owner;
    if (!file.initialized.load()) {
        require(!worker, "lazy descriptor initialized concurrently");
        file.initialized.store(true);
    }
    if (worker) {
        require(offset <= file.bytes && destination.size() <= file.bytes - offset,
                "speculative read exceeds file extent");
        ++file.async_calls;
    }
    ++file.calls;
    const int active = ++file.active;
    int previous = file.peak.load();
    while (active > previous && !file.peak.compare_exchange_weak(previous, active)) {}
    struct Retire {
        const File& file;
        ~Retire() { --file.active; }
    } retire{file};
    if (worker) { std::this_thread::sleep_for(std::chrono::milliseconds(8)); }
    require(offset != file.fail_offset, "injected direct read failure");
    auto count = offset >= file.bytes ? 0U :
        static_cast<std::size_t>(std::min<std::uint64_t>(destination.size(), file.bytes - offset));
    if (offset == file.short_offset) { count = std::min<std::size_t>(count, 37); }
    for (std::size_t index = 0; index < count; ++index) {
        destination[index] = represented(offset + index);
    }
    return count;
}

void read_and_check(Queue& queue, File& file, Buffer& buffer, std::uint64_t offset) {
    std::memset(buffer.data, 0xA5, 2 * page);
    const auto count = queue.read(&file, file.bytes, offset, buffer.span());
    auto expected = offset >= file.bytes ? 0U :
        static_cast<std::size_t>(std::min<std::uint64_t>(page, file.bytes - offset));
    if (offset == file.short_offset) { expected = std::min<std::size_t>(expected, 37); }
    require(count == expected, "native returned byte count changed");
    for (std::size_t index = 0; index < count; ++index) {
        require(buffer.data[index] == represented(offset + index), "returned bytes differ");
    }
    for (std::size_t index = count; index < 2 * page; ++index) {
        require(buffer.data[index] == std::byte{0xA5}, "destination tail or guard overwritten");
    }
}

void sequential_and_tail() {
    File file;
    Buffer buffer;
    Queue queue(native, page);
    for (std::uint64_t offset = 0; offset <= 16 * page; offset += page) {
        read_and_check(queue, file, buffer, offset);
    }
    queue.retire();
    require(file.active == 0, "threads survived retirement");
    require(file.peak > 1 && file.peak <= 4, "native concurrency was not bounded at four");
    require(queue.stats().hits == 14 && queue.stats().native_calls == 3,
            "sequential requests did not use the intended cache and tail fallback");
    require(queue.stats().peak_jobs <= 4 && queue.stats().peak_buffer_bytes == 4 * page,
            "bounded memory or queue contract changed");
    require(queue.stats().setup_failures == 0, "async setup unexpectedly fell back");
    file.closed = true;
}

void discontinuity_and_unused_error() {
    File file;
    file.fail_offset = 3 * page;
    Buffer buffer;
    Queue queue(native, page);
    read_and_check(queue, file, buffer, 0);
    read_and_check(queue, file, buffer, page);
    read_and_check(queue, file, buffer, 10 * page);
    require(queue.stats().hits == 0 && queue.stats().discarded_errors == 1,
            "unused speculative failure escaped or random read used a cache hit");
    require(queue.stats().native_calls == 3 && file.active == 0,
            "discontinuity did not drain before native fallback");
    queue.retire();
}

void requested_error() {
    File file;
    file.fail_offset = 3 * page;
    Buffer buffer;
    Queue queue(native, page);
    read_and_check(queue, file, buffer, 0);
    read_and_check(queue, file, buffer, page);
    read_and_check(queue, file, buffer, 2 * page);
    bool failed = false;
    try {
        (void)queue.read(&file, file.bytes, 3 * page, buffer.span());
    } catch (const std::runtime_error& error) {
        failed = std::string(error.what()) == "injected direct read failure";
    }
    require(failed && queue.stats().requested_errors == 1 && file.active == 0,
            "requested native exception or error cleanup changed");
}

void invalid_argument_and_file_switch() {
    File first;
    File second;
    Buffer buffer;
    Queue queue(native, page);
    read_and_check(queue, first, buffer, 0);
    read_and_check(queue, first, buffer, page);
    bool failed = false;
    try {
        (void)queue.read(&first, first.bytes, 2 * page, {buffer.data + 1, page});
    } catch (const std::runtime_error& error) {
        failed = std::string(error.what()) == "native alignment error";
    }
    require(failed && first.active == 0, "unaligned call did not drain and use native semantics");
    read_and_check(queue, first, buffer, 4 * page);
    read_and_check(queue, first, buffer, 5 * page);
    failed = false;
    try {
        (void)queue.read(&first, first.bytes, 6 * page, {static_cast<std::byte*>(nullptr), page});
    } catch (const std::runtime_error& error) {
        failed = std::string(error.what()) == "native destination error";
    }
    require(failed && first.active == 0, "null call did not drain and preserve native error semantics");
    read_and_check(queue, first, buffer, 8 * page);
    read_and_check(queue, first, buffer, 9 * page);
    read_and_check(queue, second, buffer, 0);
    require(first.active == 0 && queue.owner() == &second, "file owner transition was not drained");
    queue.retire();
    first.closed = true;
    second.closed = true;
}

void exceptional_scope_retirement() {
    File file;
    Buffer buffer;
    try {
        Queue queue(native, page);
        read_and_check(queue, file, buffer, 0);
        read_and_check(queue, file, buffer, page);
        throw std::runtime_error("consumer failure");
    } catch (const std::runtime_error&) {}
    require(file.active == 0 && file.async_calls > 0, "jobs survived exceptional queue destruction");
    file.closed = true;
}

void short_prefetch_and_empty_read() {
    File file;
    file.short_offset = 2 * page;
    Buffer buffer;
    Queue queue(native, page);
    read_and_check(queue, file, buffer, 0);
    read_and_check(queue, file, buffer, page);
    read_and_check(queue, file, buffer, 2 * page);
    require(file.active == 0 && queue.owner() == nullptr,
            "short speculative result did not retire all jobs");
    read_and_check(queue, file, buffer, 3 * page);
    read_and_check(queue, file, buffer, 4 * page);
    require(queue.read(&file, file.bytes, 5 * page, {buffer.data, 0}) == 0 && file.active == 0,
            "empty native call did not drain or preserve its return value");
    queue.retire();
}

}

int main() {
    try {
        sequential_and_tail();
        discontinuity_and_unused_error();
        requested_error();
        invalid_argument_and_file_switch();
        exceptional_scope_retirement();
        short_prefetch_and_empty_read();
        std::puts("{\"status\":\"CPU_FAKE_NATIVE_CONTRACT_PASS\",\"cases\":6,\"gpu_run\":false,\"real_direct_io_run\":false}");
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
