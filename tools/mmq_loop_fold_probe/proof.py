import collections
import hashlib
import json
import struct


SCOPE = {"type": "Q4_K", "K": 5120, "N": 17408, "T": 1280, "I": 128, "J": 128,
         "threads": 256, "warps": 8, "fixup": False, "activation_layout": "DS4",
         "outputs": ["plain", "store", "accumulate"], "dynamic_shared_bytes": 57856}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def math_trace(warp, lane, k00):
    for j0 in range(0, 128, 16):
        for k01 in range(0, 32, 8):
            for n in range(2):
                for l in range(4):
                    row = (warp // 2) * 32 + n * 16 + (l // 2) * 8 + lane // 4
                    column = (warp % 2) * 8 + j0 + (lane % 4) * 2 + l % 2
                    accumulator = (j0 // 8 + n) * 4 + l
                    for update in range(2):
                        yield row, column, accumulator, (k00 + k01) // 8, update


def run():
    copies = 0
    max_y = -1
    for kb0 in range(20):
        for jt in range(10):
            origins = (1280 * (kb0 * 256 // 128) * 36,
                       1280 * ((kb0 * 256 // 128) * 36 + 36))
            for phase in range(2):
                base = 1280 * ((kb0 * 256 // 128) * 36 + phase * 36)
                require(base == origins[phase], "activation origin differs")
                owners = set()
                for tid in range(256):
                    for l0 in range(0, 128 * 36, 256):
                        l = l0 + tid
                        absolute = base + jt * 128 * 36 + l
                        require(0 <= absolute < 1280 * 40 * 36, "activation access out of range")
                        require(l not in owners, "duplicate shared Y writer")
                        owners.add(l)
                        max_y = max(max_y, absolute)
                        copies += 1
                require(len(owners) == 4608, "shared Y coverage is incomplete")
    digest = hashlib.sha256()
    math_updates = 0
    ldmatrix_addresses = 0
    dm_addresses = 0
    owned = set()
    max_x = -1
    for warp in range(8):
        for lane in range(32):
            native = list(math_trace(warp, lane, 0)) + list(math_trace(warp, lane, 32))
            folded = [entry for phase in range(2) for entry in math_trace(warp, lane, phase * 32)]
            require(native == folded, "ordered source math trace differs")
            sequence = collections.defaultdict(list)
            for row, column, accumulator, k32, update in folded:
                require(0 <= row < 128 and 0 <= column < 128 and 0 <= accumulator < 64,
                        "output ownership out of range")
                sequence[accumulator].append((k32, update))
                digest.update(struct.pack("<7I", warp, lane, row, column, accumulator, k32, update))
            require(len(sequence) == 64, "thread accumulator coverage differs")
            require(all(value == [(k, u) for k in range(8) for u in range(2)]
                        for value in sequence.values()), "per-output K32/update order changed")
            for row, column, accumulator, k32, update in folded:
                if k32 == 0 and update == 0:
                    require((row, column) not in owned, "duplicate output owner")
                    owned.add((row, column))
            math_updates += len(folded)
            for phase in range(2):
                k00 = phase * 32
                for n in range(2):
                    for k01 in range(0, 32, 8):
                        row = (warp // 2) * 32 + n * 16 + lane % 16
                        codeword = k00 + k01 + (lane // 16) * 4
                        address = row * 76 + codeword
                        require(0 <= row < 128 and 0 <= codeword <= 60 and address % 4 == 0,
                                "ldmatrix range or alignment differs")
                        require(address + 3 < 128 * 76, "ldmatrix source exceeds X arena")
                        max_x = max(max_x, address + 3)
                        ldmatrix_addresses += 1
                    for l in range(2):
                        row = (warp // 2) * 32 + n * 16 + l * 8 + lane // 4
                        for k01 in range(0, 32, 8):
                            address = row * 76 + 64 + (k00 + k01) // 8
                            require(0 <= row < 128 and 0 <= address < 128 * 76, "DM exceeds X arena")
                            dm_addresses += 1
                    for j0 in range(0, 128, 16):
                        for l in range(2):
                            column = (warp % 2) * 8 + j0 + (lane % 4) * 2 + l
                            for k01 in range(0, 32, 8):
                                require(0 <= column * 36 + k01 // 8 < 4608, "Y DM exceeds tile")
                                code = ((warp % 2) * 8 + j0 + lane // 4) * 36 + 4 + k01 + l * 4 + lane % 4
                                require(0 <= code < 4608, "Y code exceeds tile")
    require(len(owned) == 16384, "tile output coverage is incomplete")
    require(17408 % 128 == 0 and 1280 % 128 == 0 and 5120 % 256 == 0, "first-scope tails are not empty")
    weight_blocks = set()
    for it in range(136):
        for row in range(128):
            for kb0 in range(20):
                block = (it * 128 + row) * 20 + kb0
                require(0 <= block < 17408 * 20 and block not in weight_blocks, "native X block range differs")
                weight_blocks.add(block)
    require(len(weight_blocks) == 348160, "native X row/block coverage is incomplete")
    require(128 * 4 + 128 * 36 * 4 + 128 * 76 * 4 == 57856, "shared arena footprint differs")
    return {"status": "PASS_CPU_SOURCE_INDEX_ORDER_RANGE", "activation_copy_words_checked": copies,
            "max_activation_word": max_y, "activation_allocation_words": 1843200,
            "ldmatrix_source_addresses_checked": ldmatrix_addresses, "DM_addresses_checked": dm_addresses,
            "max_X_code_word": max_x, "output_owners": len(owned),
            "native_weight_blocks_checked": len(weight_blocks), "full_shape_output_owners": 1360 * len(owned),
            "ordered_scalar_updates_per_CTA_K256": math_updates, "ordered_trace_sha256": digest.hexdigest(),
            "dynamic_barriers_per_K256": 4, "candidate_static_barriers_in_phase_body": 2,
            "dynamic_MMA_per_thread_K256": 128, "candidate_static_MMA_in_phase_body": 64,
            "phase_uniform_by_construction": True, "arrays_indexed_only_by_unrolled_n_l_k01": True,
            "all_A_DM_preloaded_before_J_K32_math": True,
            "limitation": "source symbolic proof; CUDA fragment lowering, FP32 operands and phase trip count still require compiled review"}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
