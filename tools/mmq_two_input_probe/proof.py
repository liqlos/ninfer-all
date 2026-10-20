import argparse
import itertools
import json
import pathlib
import sys

sys.dont_write_bytecode = True

import generate


INT_MAX = (1 << 31) - 1


def check(condition, message):
    if not condition:
        raise ValueError(message)


def fastdiv(n, divisor):
    shift = (divisor - 1).bit_length()
    multiplier = (((1 << 32) * ((1 << shift) - divisor)) // divisor + 1) & 0xffffffff
    high = n * multiplier >> 32
    return ((high + n) & 0xffffffff) >> shift


def divmod_native(n, divisor):
    quotient = fastdiv(n, divisor)
    check(quotient == n // divisor, "native fastdiv differs from integer division")
    return quotient, n - quotient * divisor


def local_outputs():
    owners = {}
    for warp, lane in itertools.product(range(8), range(32)):
        slots = set()
        for j0, n, l in itertools.product(range(0, 128, 16), range(2), range(4)):
            row = (warp // 2) * 32 + n * 16 + (l // 2) * 8 + lane // 4
            column = j0 + (warp % 2) * 8 + (lane % 4) * 2 + l % 2
            slot = (j0 // 8 + n) * 4 + l
            check((row, column) not in owners, "duplicate local output writer")
            owners[row, column] = (warp, lane, slot)
            slots.add(slot)
        check(slots == set(range(64)), "thread accumulator ownership differs")
    check(set(owners) == set(itertools.product(range(128), repeat=2)),
          "independent dense local output coverage failed")
    return {"outputs": len(owners), "threads": 256, "accumulators_per_thread": 64,
            "unique_writer": True, "all_local_rows_columns": True}


def tiles(profile):
    row_tiles = profile["N"] // 128
    kblocks = profile["K"] // 256
    grid = row_tiles * 2 * 10
    seen = set()
    for block in range(grid):
        start = block * (2 * 10 * row_tiles * kblocks) // grid
        stop = (block + 1) * (2 * 10 * row_tiles * kblocks) // grid
        first_tile, first_k = divmod_native(start, kblocks)
        check(first_k == 0 and stop - start == kblocks, "partial K/fixup would be required")
        temporary, jt = divmod_native(first_tile, 10)
        temporary, channel = divmod_native(temporary, 1)
        row_tile, sample = divmod_native(temporary, 2)
        check(channel == 0 and fastdiv(sample, 2) == 0, "weight sample selection changed")
        key = (sample, row_tile, jt)
        check(key not in seen, "duplicate global tile")
        seen.add(key)
        weight_first = fastdiv(sample, 2) * 0 + row_tile * 128 * kblocks
        check(weight_first == row_tile * 128 * kblocks, "weight offset differs")
        check(weight_first + 128 * kblocks <= profile["N"] * kblocks,
              "weight panel overrun")
    expected = set(itertools.product(range(2), range(row_tiles), range(10)))
    check(seen == expected, "independent sample/row/token tile coverage failed")
    return {"candidate_grid": grid, "control_grids": [grid // 2, grid // 2],
            "full_K_blocks_per_CTA": kblocks, "fixup_tiles": 0,
            "logical_outputs": 2 * profile["N"] * 1280,
            "sample_row_column_bijection": True,
            "same_weight_pointer_between_samples": True,
            "CTA_order_inner_to_outer": ["token_tile", "sample", "row_tile"],
            "first_20_CTAs": [
                {"block": block, "row_tile": 0, "sample": block // 10,
                 "token_tile": block % 10} for block in range(20)],
            "same_weight_tile_CTA_distance": {"control": grid // 2, "candidate": 10},
            "weight_bytes_per_row_tile": 128 * kblocks * profile["weight_block_bytes"],
            "explicit_cross_CTA_shared_reuse": False,
            "L2_reuse": "hypothesis; block issue order and cache residency are not guaranteed"}


def input_addresses(profile, padding_bytes):
    chunks = profile["K"] // 128
    panel_words = chunks * 1280 * 36
    sample_words = panel_words + padding_bytes // 4
    seen = set()
    intervals = []
    for sample, kblock, half, jt in itertools.product(
            range(2), range(chunks // 2), range(2), range(10)):
        native = sample * sample_words + jt * 128 * 36 + 1280 * (kblock * 2 + half) * 36
        expected = sample * sample_words + ((kblock * 2 + half) * 1280 + jt * 128) * 36
        check(native == expected, "native packed activation base differs")
        key = (sample, kblock * 2 + half, jt)
        check(key not in seen, "duplicate activation tile")
        seen.add(key)
        intervals.append((native, native + 128 * 36))
        for word in (0, 35, 36, 127 * 36, 128 * 36 - 1):
            column, component = divmod(word, 36)
            independent = sample * sample_words + (
                (kblock * 2 + half) * 1280 + jt * 128 + column) * 36 + component
            check(native + word == independent, "native packed activation word differs")
    check(seen == set(itertools.product(range(2), range(chunks), range(10))),
          "packed sample/chunk/token coverage differs")
    intervals.sort()
    for first, second in zip(intervals, intervals[1:]):
        check(first[1] <= second[0], "activation panel alias")
    check(sum(last - first for first, last in intervals) == 2 * panel_words,
          "packed activation coverage length differs")
    check(intervals[0][0] == 0 and intervals[-1][1] == sample_words + panel_words,
          "packed activation bounds differ")
    check(sample_words + panel_words <= INT_MAX, "activation offset overflow")
    return {"panel_bytes": panel_words * 4, "sample_stride_bytes": sample_words * 4,
            "independent_panel_packing": True, "guard_bytes_between_samples": padding_bytes,
            "copied_word_intervals": len(intervals), "no_T2560_reinterpretation": True}


def output_addresses(profile, element_bytes, column_padding, sample_padding):
    rows = profile["N"]
    stride = rows + column_padding
    span = 1279 * stride + rows
    sample_stride = span + sample_padding
    intervals = []
    for sample, column in itertools.product(range(2), range(1280)):
        expected_first = sample * sample_stride + column * stride
        row_parts = []
        jt, j = divmod(column, 128)
        for it in range(rows // 128):
            native = sample * sample_stride + jt * 128 * stride + it * 128 + j * stride
            check(native == expected_first + it * 128, "output addressing differs")
            row_parts.append((native, native + 128))
        check(row_parts[0][0] == expected_first and row_parts[-1][1] == expected_first + rows,
              "output row coverage differs")
        check(all(a[1] == b[0] for a, b in zip(row_parts, row_parts[1:])),
              "output row gap or overlap")
        intervals.append((expected_first, expected_first + rows))
    check(all(a[1] <= b[0] for a, b in zip(intervals, intervals[1:])),
          "output/carry aliases between columns or samples")
    touched = sum(last - first for first, last in intervals)
    check(touched == 2 * rows * 1280, "output coverage count differs")
    check(intervals[-1][1] == sample_stride + span <= INT_MAX, "output index overflow")
    check(sample_stride * element_bytes % 16 == 0, "sample pointer alignment differs")
    guards = sample_stride + span - touched
    check(guards == 2 * 1279 * column_padding + sample_padding, "output guard span differs")
    return {"element_bytes": element_bytes, "column_stride_elements": stride,
            "sample_stride_elements": sample_stride, "output_elements": touched,
            "untouched_guard_elements": guards,
            "carry_read_address_equals_unique_write_address": True,
            "carry_cross_sample_alias": False}


def prove(root):
    _, receipt = generate.prepare(root)
    variant = (pathlib.Path(__file__).resolve().parent / "variant.cu").read_text()
    for literal in (
        "mul_mat_q<Type, kJ, false, GGML_PREC_Q8, Store, Accumulate, false, false>",
        "samples_fd, samples_fd, 0,", "constexpr int kSharedBytes = 57856;",
        "a.packed_sample_stride_bytes / 4 > INT_MAX - packed_panel_bytes / 4",
        "a.output_sample_stride_elements > INT_MAX - output_panel_elements",
        "overlaps(output, weights) || overlaps(output, inputs)",
        "NINFER_LAB_MMQ_TWO_INPUT ? input_sample_words : 0",
        "NINFER_LAB_MMQ_TWO_INPUT ? output_sample_elements : 0",
    ):
        generate.require(variant, literal)
    result = {"status": "CPU_ADDRESS_PROOF_PASS", "native_pins": receipt["native_pins"],
              "local_output_ownership": local_outputs(), "profiles": {}}
    for name, profile in generate.PROFILES.items():
        result["profiles"][name] = {
            "tiles": tiles(profile),
            "input_address_cases": [input_addresses(profile, pad) for pad in (0, 256)],
            "output_address_cases": [output_addresses(profile, width, cp, sp)
                                     for width in (4, 2) for cp, sp in ((0, 0), (8, 32))],
            "equal_logical_traffic_per_two_sample_projection": {
                "weight_read_bytes": 2 * 10 * profile["N"] * (profile["K"] // 256)
                                     * profile["weight_block_bytes"],
                "activation_read_bytes": 2 * (profile["N"] // 128)
                                         * (profile["K"] // 128) * 1280 * 144,
                "plain_output_write_bytes": 2 * profile["N"] * 1280 * 4,
                "store_output_write_bytes": 2 * profile["N"] * 1280 * 2,
                "accumulate_output_read_plus_write_bytes": 2 * profile["N"] * 1280 * 4,
                "DRAM_bytes_measured": False,
            },
        }
    result.update({
        "proof_method": "tile/local Cartesian bijections and disjoint affine address intervals",
        "numerical_math": "native header unchanged; numerical GPU oracle not run",
        "accumulate": "pinned native old BF16 read and final BF16 write share the same at index",
        "compiled_API_negative_calls": "NOT_RUN; require the future compiled host",
        "compiled": False, "GPU_run": False, "performance_claim": False,
    })
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=pathlib.Path, default=generate.ROOT)
    args = parser.parse_args()
    print(json.dumps(prove(args.source_root.resolve()), indent=2))


if __name__ == "__main__":
    main()
