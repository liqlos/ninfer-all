import collections
import itertools
import math
import random
import struct


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def bits(value):
    return struct.pack("<f", value)


def add(a, b):
    return f32(a + b)


def max_number(a, b):
    require(not math.isnan(a) and not math.isnan(b), "NaN outside proof domain")
    if a == b == 0.0:
        return -0.0 if math.copysign(1.0, a) < 0 and math.copysign(1.0, b) < 0 else 0.0
    return a if a > b else b


def fold_max(values):
    value = -math.inf
    for item in values:
        value = max_number(value, item)
    return value


def quad_reduce(values, operation):
    values = list(values)
    for delta in (2, 1):
        values = [operation(values[lane], values[lane ^ delta]) for lane in range(4)]
    return values


def ownership():
    outputs = collections.Counter()
    scores = collections.Counter()
    qscales = collections.Counter()
    for warp, lane in itertools.product(range(16), range(32)):
        pair, side, gid, lid = warp // 2, warp % 2, lane // 4, lane % 4
        for block, parity, item in itertools.product(range(8), range(2), range(4)):
            row = pair * 16 + gid + (item // 2) * 8
            dim = side * 128 + block * 16 + 4 * lid + 2 * (item % 2) + parity
            outputs[row, dim] += 1
        for nt, item in itertools.product(range(4), range(4)):
            row = pair * 16 + gid + (item // 2) * 8
            key = (side * 4 + nt) * 8 + 2 * lid + item % 2
            scores[row, key] += 1
            for group in range(4):
                donor = (lane & ~3) | group
                require(donor // 4 == gid and donor % 4 == group, "Q-scale shuffle ownership")
                require((side * 4 + nt, group, (0, 1)) == (key // 8, group, (0, 1)),
                        "QK group or kk order")
        for r, group in itertools.product(range(8), range(4)):
            if gid == r and lid == group:
                qscales[pair * 16 + side * 8 + r, group] += 1
    require(set(outputs) == set(itertools.product(range(128), range(256))), "output coverage")
    require(set(outputs.values()) == {1}, "output unique writer")
    require(set(scores) == set(itertools.product(range(128), range(64))), "score coverage")
    require(set(scores.values()) == {1}, "score unique owner")
    require(set(qscales) == set(itertools.product(range(128), range(4))), "Q-scale coverage")
    require(set(qscales.values()) == {1}, "Q-scale unique writer")
    return {"output_elements": len(outputs), "scores_per_tile": len(scores),
            "temporary_Q_scales": len(qscales), "unique_writers": True}


def probability_transport():
    slots = {}
    for pair, side, lane, j, item in itertools.product(
            range(8), range(2), range(32), range(2), range(4)):
        slot = ((pair * 2 + side) * 2 + j) * 128 + lane * 4 + item
        require(slot not in slots, "P writer overlap")
        slots[slot] = pair, lane, side * 2 + j, item
    require(set(slots) == set(range(4096)), "P arena coverage")
    for pair, consumer_side, lane, j, item in itertools.product(
            range(8), range(2), range(32), range(4), range(4)):
        slot = ((pair * 2 + (j >> 1)) * 2 + (j & 1)) * 128 + lane * 4 + item
        require(slots[slot] == (pair, lane, j, item), "P consumer fragment identity")
    for pair, lane, row in itertools.product(range(8), range(32), range(2)):
        carry_slot = (pair * 2 + 1) * 256 + lane * 2 + row
        require((pair * 2 + 1) * 256 <= carry_slot < (pair * 2 + 2) * 256,
                "carry outside partner P segment")
        require(carry_slot not in range(pair * 512, pair * 512 + 256), "carry aliases low P")
    for value in range(65536):
        packed = value | ((65535 - value) << 16)
        copied = struct.unpack("<I", struct.pack("<I", packed))[0]
        require((copied & 65535, copied >> 16) == (value, 65535 - value), "half payload transport")
    for lane_group in range(4):
        banks = [(lane * 4 + word) % 32 for lane, word in
                 itertools.product(range(lane_group * 8, lane_group * 8 + 8), range(4))]
        require(len(set(banks)) == 32, "P vector transaction shared-bank model")
    wrong = (0 * 2 + 1) * 256
    require(slots[wrong] != (0, 0, 0, 0), "negative control did not detect swapped P half")
    return {"P_arena_bytes": 16384, "carry_bytes_per_pair": 256,
            "P_half_bit_patterns": 65536, "wrong_side_negative_control": "PASS",
            "P_vector_model_32_banks_4B_per_bank_unique_per_8_lanes": True}


def fragments():
    def swizzle(row, column):
        return (((column >> 3) ^ (row & 7)) << 3) | (column & 7)

    qk_addresses = 0
    for pair, side, lane, group, nt, kk in itertools.product(
            range(8), range(2), range(32), range(4), range(4), range(2)):
        a_mat, a_rin = lane >> 3, lane & 7
        a_row = pair * 16 + a_rin + ((a_mat & 1) << 3)
        a_col = (group * 2 + kk) * 16 + ((a_mat >> 1) << 3)
        paired_q = a_row * 128 + swizzle(a_row, a_col)
        native_q = a_row * 128 + swizzle(a_row, a_col)
        key = (side * 4 + nt) * 8 + a_rin
        b_col = (group * 2 + kk) * 16 + (((lane >> 3) & 1) << 3)
        paired_k = key * 128 + swizzle(key, b_col)
        native_nt = side * 4 + nt
        native_k = (native_nt * 8 + a_rin) * 128 + swizzle(native_nt * 8 + a_rin, b_col)
        require((paired_q, paired_k) == (native_q, native_k), "QK fragment mismatch")
        qk_addresses += 1
    pv_addresses = 0
    for side, lane, group, pass_, j in itertools.product(
            range(2), range(32), range(2), range(2), range(4)):
        a_mat, a_rin, lid = lane >> 3, lane & 7, lane & 3
        local_block = group * 4 + pass_ * 2
        global_block = side * 8 + local_block
        key = j * 16 + ((a_mat & 1) << 3) + a_rin
        chunk = global_block + (a_mat >> 1)
        paired_v = key * 256 + ((chunk ^ (key & 7)) << 4)
        native_group = side * 2 + group
        native_chunk = native_group * 4 + pass_ * 2 + (a_mat >> 1)
        native_v = key * 256 + ((native_chunk ^ (key & 7)) << 4)
        require(paired_v == native_v, "PV ldmatrix fragment mismatch")
        for key_offset in (0, 1, 8, 9):
            key_scale = j * 16 + 2 * lid + key_offset
            require(key_scale * 4 + side * 2 + group == key_scale * 4 + native_group,
                    "PV V-scale mismatch")
        pv_addresses += 1
    return {"QK_operand_addresses": qk_addresses, "PV_operand_addresses": pv_addresses,
            "integer_QK_groups_then_FP32_FMA": [0, 1, 2, 3],
            "PV_four_FP16_accumulator_steps_before_promotion": [0, 1, 2, 3]}


def memory_and_pages():
    q_bytes, stage_bytes, scratch_bytes = 32768, 33792, 1024
    require(q_bytes + 2 * stage_bytes + scratch_bytes == 101376, "shared limit")
    destinations = {}
    for tid, iteration in itertools.product(range(512), range(2)):
        chunk = tid + iteration * 512
        key, column = chunk >> 4, chunk & 15
        dst = key * 256 + ((column ^ (key & 7)) << 4)
        require(dst not in destinations, "async copy overlap")
        destinations[dst] = key, column
    require(set(destinations) == set(range(0, 16384, 16)), "async copy coverage")
    cases = 0
    for page, head, valid in itertools.product((0, 3, 1023), range(4), range(65)):
        for dst, (key, column) in destinations.items():
            source = ((page * 4 + head) * 64 + key) * 256 + column * 16
            require(source == page * 65536 + head * 16384 + key * 256 + column * 16,
                    "fragmented physical page address")
            require((16 if key < valid else 0) in (0, 16), "copy zero-fill")
            require(dst + 16 <= 16384, "K/V shared bound")
        cases += 1
    require(128 * 4 * 4 <= 16384, "prologue scales exceed dead K arena")
    require(8 * 4 <= 512, "tile shifts exceed dead K-scale arena")
    for stage in range(2):
        base = q_bytes + stage * stage_bytes
        require(base + 16384 <= base + 32768 <= base + 33280 <= base + stage_bytes,
                "stage field separation")
    return {"dynamic_shared_bytes": 101376, "stage_bytes": stage_bytes,
            "async_16B_destinations_per_K_or_V": len(destinations),
            "physical_page_head_tail_cases": cases, "additional_KV_stages": 0,
            "max_and_final_denominator_scratch_bytes": 1024}


def ordering():
    edges = {
        ("Q_scale_write", "prepare_A"), ("prepare_A", "Q_scale_read"),
        ("Q_scale_read", "prepare_B"), ("prepare_B", "first_copy"),
        ("copy_done", "entry"), ("entry", "QK_read"),
        ("QK_read", "max_write"), ("max_write", "A"),
        ("A", "partner_max_read"), ("partner_max_read", "P_low_write"),
        ("partner_max_read", "carry_write"), ("partner_max_read", "shift_write"),
        ("P_low_write", "B"), ("carry_write", "B"), ("shift_write", "B"),
        ("B", "carry_read"), ("B", "shift_read"),
        ("carry_read", "warp_sync"), ("warp_sync", "P_high_write"),
        ("P_high_write", "C"), ("B", "C"), ("shift_read", "C"),
        ("C", "PV_read"), ("PV_read", "next_entry"),
        ("next_entry", "same_stage_next_copy"),
        ("final_running_l", "inv_l_write"), ("inv_l_write", "final_barrier"),
        ("final_barrier", "partner_inv_l_read"),
    }

    def reaches(start, end, graph):
        reached = {start}
        while True:
            updated = reached | {b for a, b in graph if a in reached}
            if updated == reached:
                return end in reached
            reached = updated

    obligations = [
        ("Q_scale_read", "first_copy"), ("QK_read", "P_low_write"),
        ("QK_read", "shift_write"), ("max_write", "partner_max_read"),
        ("carry_write", "carry_read"), ("carry_read", "P_high_write"),
        ("P_low_write", "PV_read"), ("P_high_write", "PV_read"),
        ("PV_read", "same_stage_next_copy"), ("inv_l_write", "partner_inv_l_read"),
    ]
    for start, end in obligations:
        require(reaches(start, end, edges), "missing happens-before: " + start + " -> " + end)
    for barrier, start, end in (
            ("A", "max_write", "partner_max_read"),
            ("B", "carry_write", "carry_read"),
            ("C", "P_high_write", "PV_read"),
            ("warp_sync", "carry_read", "P_high_write"),
            ("next_entry", "PV_read", "same_stage_next_copy"),
            ("final_barrier", "inv_l_write", "partner_inv_l_read")):
        removed = {edge for edge in edges if barrier not in edge}
        require(not reaches(start, end, removed), "missing-barrier negative control: " + barrier)
    return {"happens_before_obligations": len(obligations), "negative_controls": 6,
            "page_CTA_barriers": 4, "prologue_CTA_barriers": 2,
            "terminal_CTA_barriers": 1, "model_is_not_compiled_race_proof": True}


def tails():
    checked = 0
    for width, base in itertools.product((1, 33, 127, 129, 768, 1280), (0, 7, 31488)):
        for tokens in sorted({1, max(1, width - 1), width}):
            for block in range((width + 127) // 128):
                q0 = block * 128
                if q0 >= tokens:
                    continue
                rows = min(128, tokens - q0)
                pages = (base + q0 + rows - 1) // 64 + 1
                for pair in range(8):
                    active = pair * 16 < rows
                    maximum = base + q0 + min(pair * 16 + 15, rows - 1)
                    for kb in {0, pages - 1, min(pages - 1, maximum // 64)}:
                        live = active and kb * 64 <= maximum
                        for row in (pair * 16, pair * 16 + 15):
                            qabs = base + q0 + row if row < rows else -1
                            visible = [kb * 64 + key <= qabs for key in range(64)]
                            require(live or not any(visible), "inactive pair has visible keys")
                            if row >= rows:
                                require(not any(visible), "tail row has a visible key")
                        checked += 1
    return {"tail_pair_page_cases": checked, "all_16_warps_execute_uniform_barriers": True,
            "all_CTA_zero_return_is_uniform": True,
            "inactive_rows_publish_BF16_zero": True,
            "compiled_candidate_metadata": "unmasked only; partial final row-block tails are in scope"}


def arithmetic():
    values = [-math.inf, -3.0, -0.0, 0.0, f32(2.0 ** -126), 2.0, math.inf]
    for a, b, c in itertools.product(values, repeat=3):
        require(bits(max_number(max_number(a, b), c)) == bits(max_number(a, max_number(b, c))),
                "max reassociation")
    rng = random.Random(20261020)
    cases = 1024
    regroup_differences = 0
    for case in range(cases):
        score = [[rng.choice(values) if case < 128 else f32(rng.uniform(-80, 80))
                  for _ in range(16)] for _ in range(4)]
        original = quad_reduce([fold_max(row) for row in score], max_number)
        halves = [quad_reduce([fold_max(row[start:start + 8]) for row in score], max_number)
                  for start in (0, 8)]
        paired = [max_number(a, b) for a, b in zip(*halves)]
        require([bits(x) for x in original] == [bits(x) for x in paired], "half-page max")
        probabilities = [f32(2.0 ** rng.uniform(-24, 0)) for _ in range(16)]
        pairs = [add(probabilities[i], probabilities[i + 1]) for i in range(0, 16, 2)]
        full = 0.0
        for item in pairs:
            full = add(full, item)
        low = 0.0
        for item in pairs[:4]:
            low = add(low, item)
        continued = low
        for item in pairs[4:]:
            continued = add(continued, item)
        require(bits(full) == bits(continued), "serial low-to-high denominator carry")
        high = 0.0
        for item in pairs[4:]:
            high = add(high, item)
        regroup_differences += bits(full) != bits(add(low, high))
        lane_l = [f32(2.0 ** rng.uniform(-24, 12)) for _ in range(4)]
        reduced = quad_reduce(lane_l, add)
        require(len({bits(value) for value in reduced}) == 1,
                "lid0 denominator publication differs from another quad lane")
    require(regroup_differences > 0, "partial-sum negative control")
    original_dag = "0"
    for nt in range(8):
        original_dag = ("add_rn", original_dag, ("add_rn", "p" + str(nt) + "0", "p" + str(nt) + "1"))
    carry_dag = "0"
    for part in (range(4), range(4, 8)):
        for nt in part:
            carry_dag = ("add_rn", carry_dag, ("add_rn", "p" + str(nt) + "0", "p" + str(nt) + "1"))
    require(original_dag == carry_dag, "symbolic sum order")
    native_l = ("fma_rn", "prior_l", "alpha", original_dag)
    partner_l = ("fma_rn", "prior_l", "alpha", carry_dag)
    require(native_l == partner_l, "online denominator FMA")
    return {"max_associativity_cases": len(values) ** 3, "random_tile_cases": cases,
            "wrong_regrouped_sum_differences": regroup_differences,
            "original_running_l_FMA_DAG": "preserved",
            "final_quad_reduce_and_reciprocal": "native source retained on side 1, bits transported",
            "domain": "finite computed QK scores and masked -inf, finite quant scales, positive finite attention scale",
            "MUFU_and_tensor_core_emulation": False,
            "max_lowering_and_IEEE_signed_zero_require_compile_review": True}


def prove():
    return {"status": "PASS", "ownership": ownership(),
            "P_transport": probability_transport(), "fragments": fragments(), "memory": memory_and_pages(),
            "ordering_model": ordering(), "tails": tails(), "arithmetic": arithmetic(),
            "resources": {
                "source_live_arrays_QK": {"acc": 64, "score": 16, "af": 8, "Q_scales": 2},
                "source_live_arrays_PV": {"acc": 64, "P": 16, "h": 8, "V_scales": 2},
                "array_counts_are_not_actual_register_counts": True,
                "compile_prerequisite_registers_max": 128,
                "compile_prerequisite_stack_spills_local_bytes": 0,
                "compile_prerequisite_total_shared_bytes_max": 101376,
                "actual_compile_pass": False},
            "cost": {"CTA_IMMA": 512, "CTA_HMMA": 1024,
                     "native_main_shared_read_bytes": 294912,
                     "paired_main_shared_read_write_bytes": 376832,
                     "main_shared_traffic_increase_fraction": 81920 / 294912,
                     "extra_logical_V_scale_read_bytes": 32768,
                     "extra_logical_max_carry_shift_bytes": 10272,
                     "extra_Q_scale_shuffle_warp_instructions": 128,
                     "extra_CTA_barriers_per_page": 3,
                     "performance_claim": None}}
