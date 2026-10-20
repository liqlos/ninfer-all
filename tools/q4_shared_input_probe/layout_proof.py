import collections
import json


def prove():
    columns, k, rows, blocks, threads = 4, 5120, 17408, 328, 128
    code_bytes = columns * k
    metadata_bytes = columns * (k // 32) * 4
    owners = collections.Counter()
    values = [-1] * (code_bytes // 4)
    for thread in range(threads):
        for index in range(thread, code_bytes // 16, threads):
            destination = index ^ ((index >> 3) & 1)
            for word in range(4):
                address = destination * 4 + word
                owners[address] += 1
                values[address] = index * 4 + word
    if set(owners) != set(range(code_bytes // 4)) or set(owners.values()) != {1}:
        raise ValueError("shared code preload is not a range-exact bijection")
    consumer_words = 0
    transaction_count = 0
    for column in range(columns):
        for iteration in range(k // 1024):
            for half in range(2):
                for first_lane in range(0, 32, 8):
                    banks = collections.Counter()
                    for lane in range(first_lane, first_lane + 8):
                        s = lane + 32 * iteration
                        for word in range(4):
                            logical = column * (k // 4) + s * 8 + half * 4 + word
                            address = column * (k // 4) + s * 8 + ((half * 4 + word) ^ (s & 4))
                            if values[address] != logical:
                                raise ValueError("consumer does not recover the native Q8 word")
                            banks[address % 32] += 1
                            consumer_words += 1
                    if len(banks) != 32 or set(banks.values()) != {1}:
                        raise ValueError("modeled 128-byte transaction has a bank conflict")
                    transaction_count += 1
    metadata_owners = collections.Counter()
    for thread in range(threads):
        for index in range(thread, metadata_bytes // 16, threads):
            for byte in range(16):
                metadata_owners[code_bytes + index * 16 + byte] += 1
    if set(metadata_owners) != set(range(code_bytes, code_bytes + metadata_bytes)):
        raise ValueError("literal half2 metadata range changed")
    if set(metadata_owners.values()) != {1}:
        raise ValueError("metadata is not written exactly once")
    row_owners = collections.Counter()
    for block in range(blocks):
        for warp in range(threads // 32):
            for row in range(block * 4 + warp, rows, blocks * 4):
                row_owners[row] += 1
    if set(row_owners) != set(range(rows)) or set(row_owners.values()) != {1}:
        raise ValueError("native grid-stride row coverage changed")
    return {
        "status": "PASS_CPU_LAYOUT_AND_RANGE",
        "code_words_roundtripped": consumer_words,
        "metadata_bytes_literal": metadata_bytes,
        "shared_bytes": code_bytes + metadata_bytes,
        "modeled_128byte_transactions": transaction_count,
        "modeled_max_bank_multiplicity": 1,
        "bank_model_is_not_gpu_evidence": True,
        "row_owners": len(row_owners),
        "grid_blocks": blocks,
        "block_threads": threads,
        "K_iterations_per_lane": k // 1024,
        "ordered_slices": "lane + 32 * it, it=0..4",
        "four_CTA_bytes_including_1KiB_reserved_each": 4 * (code_bytes + metadata_bytes + 1024),
        "SM86_shared_bytes": 102400,
        "FP32_metadata_expansion_four_CTA_bytes": 4 * (code_bytes + metadata_bytes * 2 + 1024),
        "baseline_logical_global_input_bytes_per_FFN": rows * (code_bytes + metadata_bytes),
        "candidate_preload_bytes_per_FFN": blocks * (code_bytes + metadata_bytes),
        "DRAM_traffic_measured": False,
    }


if __name__ == "__main__":
    print(json.dumps(prove(), indent=2))
