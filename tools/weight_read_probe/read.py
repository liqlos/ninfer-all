import argparse
import concurrent.futures
import ctypes
import hashlib
import json
import mmap
import os
import pathlib
import threading
import time


SLOT = 64 * 2**20
PAGE = 4096
OFFSET = 4 * 2**30
COUNT = 32
MODEL_BYTES = 17695430144
ORDER = (1, 2, 4, 4, 2, 1)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def identity(path):
    s = path.stat()
    return {key: getattr(s, key) for key in ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")}


def sample(buffer):
    digest = hashlib.sha256()
    digest.update(memoryview(buffer)[:PAGE])
    digest.update(memoryview(buffer)[SLOT - PAGE:SLOT])
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Read-only direct-I/O queue-depth screen; no CUDA.")
    parser.add_argument("--model", type=pathlib.Path, required=True)
    args = parser.parse_args()
    require(os.name == "posix" and hasattr(os, "preadv") and hasattr(os, "O_DIRECT"),
            "Use the existing Linux model bind")
    path = args.model.resolve(strict=True)
    before = identity(path)
    require(before["st_size"] == MODEL_BYTES and OFFSET + COUNT * SLOT <= MODEL_BYTES,
            "Model extent differs")
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECT)
    slots = [mmap.mmap(-1, SLOT) for _ in range(4)]
    page = mmap.mmap(-1, PAGE)
    try:
        require(all(ctypes.addressof(ctypes.c_char.from_buffer(b)) % PAGE == 0 for b in [*slots, page]),
                "Direct-I/O memory is not page aligned")
        for b in slots:
            ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(b)), 0, SLOT)
        reference = []
        for index in range(COUNT):
            digest = hashlib.sha256()
            for offset in (OFFSET + index * SLOT, OFFSET + (index + 1) * SLOT - PAGE):
                require(os.preadv(fd, [page], offset) == PAGE, "Short reference read")
                digest.update(page)
            reference.append(digest.hexdigest())
        trials = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            for depth in ORDER:
                barrier = threading.Barrier(depth + 1)

                def worker(lane):
                    barrier.wait()
                    blocks = []
                    for index in range(lane, COUNT, depth):
                        count = os.preadv(fd, [slots[lane]], OFFSET + index * SLOT)
                        require(count == SLOT, "Short direct read")
                        require(sample(slots[lane]) == reference[index], "Read bytes differ")
                        blocks.append(index)
                    return blocks

                futures = [pool.submit(worker, lane) for lane in range(depth)]
                started = time.perf_counter()
                barrier.wait()
                completed = [index for future in futures for index in future.result()]
                seconds = time.perf_counter() - started
                require(sorted(completed) == list(range(COUNT)), "Missing or duplicate blocks")
                trial = {"depth": depth, "bytes": COUNT * SLOT, "seconds": seconds,
                         "MiB_per_second": COUNT * SLOT / 2**20 / seconds,
                         "first_and_last_page_hashes_match": True}
                trials.append(trial)
                print(json.dumps({"event": "trial", **trial}), flush=True)
        require(identity(path) == before, "Read-only model identity changed")
        means = {depth: sum(r["seconds"] for r in trials if r["depth"] == depth) / 2
                 for depth in (1, 2, 4)}
        print(json.dumps({"event": "complete", "status": "PASS", "diagnostic_only": True,
                          "model_identity": before, "model_identity_unchanged": True,
                          "O_DIRECT": True, "GPU_run": False, "slot_bytes": SLOT,
                          "slot_count": 4, "window_offset": OFFSET, "window_bytes": COUNT * SLOT,
                          "order": ORDER, "mean_seconds": means,
                          "QD2_time_reduction_percent": 100 * (1 - means[2] / means[1]),
                          "QD4_time_reduction_percent": 100 * (1 - means[4] / means[1]),
                          "statistical_confirmation": False, "model_load_or_request_gain": False,
                          "scope": "One contiguous2GiB window in existing model bind;256KiB boundary hashes per trial. Not a complete artifact upload."}), flush=True)
    finally:
        os.close(fd)
        page.close()
        for b in slots:
            b.close()


if __name__ == "__main__":
    main()
