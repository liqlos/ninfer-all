#include "artifact/file_io.h"
#include "read_ahead.h"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <string>
#include <sys/stat.h>

namespace {

using ninfer::artifact::InputFile;
using weight_read_ahead_probe::Queue;
constexpr std::size_t slot_bytes = Queue::default_slot_bytes;
constexpr std::size_t guard_bytes = Queue::alignment;
constexpr std::uint64_t model_bytes = 17695430144ULL;

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

struct Buffer {
    std::byte* data = nullptr;
    Buffer() {
        void* pointer = nullptr;
        if (::posix_memalign(&pointer, Queue::alignment, slot_bytes + guard_bytes) != 0) {
            throw std::bad_alloc();
        }
        data = static_cast<std::byte*>(pointer);
    }
    ~Buffer() { std::free(data); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    void reset() { std::memset(data, 0xa5, slot_bytes + guard_bytes); }
    std::span<std::byte> span(std::size_t count) { return {data, count}; }
};

struct stat identity(const std::filesystem::path& path) {
    struct stat value {};
    require(::stat(path.c_str(), &value) == 0 && S_ISREG(value.st_mode), "model stat failed");
    return value;
}

std::size_t native(const void* owner, std::uint64_t offset, std::span<std::byte> destination) {
    return static_cast<const InputFile*>(owner)->read_direct(offset, destination);
}

}

int main(int argc, char** argv) {
    try {
        require(argc == 2, "expected explicit read-only model path");
        const std::filesystem::path path = argv[1];
        const auto before = identity(path);
        require(before.st_size == model_bytes, "model size changed");
        InputFile reference(path);
        InputFile candidate(path);
        Queue queue(native);
        Buffer expected;
        Buffer actual;
        std::uint64_t compared = 0;
        std::uint64_t calls = 0;
        auto compare = [&](const InputFile& owner, std::uint64_t offset, std::size_t count) {
            expected.reset();
            actual.reset();
            const auto a = reference.read_direct(offset, expected.span(count));
            const auto b = queue.read(&owner, owner.bytes(), offset, actual.span(count));
            require(a == b, "native read counts differ");
            require(std::memcmp(expected.data, actual.data, slot_bytes + guard_bytes) == 0,
                    "native read bytes, unread tail, or canary differ");
            require(std::all_of(actual.data + count, actual.data + slot_bytes + guard_bytes,
                    [](std::byte value) { return value == std::byte{0xa5}; }), "canary changed");
            compared += a;
            ++calls;
            return a;
        };
        constexpr std::uint64_t begin = 4ULL << 30;
        for (std::size_t index = 0; index < 16; ++index) {
            require(compare(candidate, begin + index * slot_bytes, slot_bytes) == slot_bytes,
                    "unexpected sequential short read");
        }
        require(queue.stats().hits > 0 && queue.stats().launched > 0, "read-ahead was not exercised");
        compare(candidate, begin + 3 * slot_bytes, slot_bytes);
        compare(reference, begin, guard_bytes);
        compare(candidate, begin + slot_bytes, guard_bytes);
        const std::uint64_t tail = model_bytes / guard_bytes * guard_bytes;
        require(compare(candidate, tail, guard_bytes) == model_bytes - tail, "native final tail differs");
        require(compare(candidate, tail + guard_bytes, guard_bytes) == 0, "native EOF differs");
        require(compare(candidate, 0, 0) == 0, "native empty read differs");
        std::string error_a;
        std::string error_b;
        try { static_cast<void>(reference.read_direct(begin + 1, expected.span(guard_bytes))); }
        catch (const std::exception& error) { error_a = error.what(); }
        try { queue.read(&candidate, candidate.bytes(), begin + 1, actual.span(guard_bytes)); }
        catch (const std::exception& error) { error_b = error.what(); }
        require(!error_a.empty() && error_a == error_b, "native alignment error differs");
        queue.retire();
        const auto& s = queue.stats();
        require(queue.owner() == nullptr && s.peak_jobs <= Queue::maximum_slots &&
                s.peak_buffer_bytes <= Queue::maximum_slots * slot_bytes && s.setup_failures == 0,
                "queue bound or retirement failed");
        require(s.launched == s.hits + s.discarded_jobs &&
                s.async_bytes == s.copied_bytes + s.discarded_bytes, "async jobs or bytes not retired");
        require(s.requested_errors == 1 && s.discarded_errors == 0, "unexpected native errors");
        const auto after = identity(path);
        require(before.st_dev == after.st_dev && before.st_ino == after.st_ino &&
                before.st_size == after.st_size && before.st_mtim.tv_sec == after.st_mtim.tv_sec &&
                before.st_mtim.tv_nsec == after.st_mtim.tv_nsec &&
                before.st_ctim.tv_sec == after.st_ctim.tv_sec &&
                before.st_ctim.tv_nsec == after.st_ctim.tv_nsec, "model identity changed");
        std::printf("{\"status\":\"PASS\",\"scope\":\"real native InputFile callback; not materializer wrapper qualification\","
                    "\"GPU_run\":false,\"performance_claim\":false,\"full_compared_bytes\":%llu,"
                    "\"paired_reads\":%llu,\"hits\":%llu,\"launched\":%llu,\"discarded_jobs\":%llu,"
                    "\"peak_jobs\":%llu,\"peak_buffer_bytes\":%llu,\"model_identity_unchanged\":true,"
                    "\"tail_EOF_empty_canaries_file_switch_alignment_error\":\"PASS\"}\n",
                    static_cast<unsigned long long>(compared), static_cast<unsigned long long>(calls),
                    static_cast<unsigned long long>(s.hits), static_cast<unsigned long long>(s.launched),
                    static_cast<unsigned long long>(s.discarded_jobs), static_cast<unsigned long long>(s.peak_jobs),
                    static_cast<unsigned long long>(s.peak_buffer_bytes));
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
