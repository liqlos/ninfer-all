import json

import generate


def require(value, message):
    if not value:
        raise ValueError(message)


def add(left, right):
    return ("FP32_add", left, right)


def decode_metadata(data, u):
    if u < 4:
        return data[u] & 63, data[u + 4] & 63
    return ((data[u + 4] & 15) | ((data[u - 4] >> 6) << 4),
            (data[u + 4] >> 4) | ((data[u] >> 6) << 4))


def encode_metadata(scales, minima):
    return ([scales[j] | ((scales[j + 4] >> 4) << 6) for j in range(4)] +
            [minima[j] | ((minima[j + 4] >> 4) << 6) for j in range(4)] +
            [(scales[j + 4] & 15) | ((minima[j + 4] & 15) << 4) for j in range(4)])


def prove(root=generate.ROOT):
    _, receipt = generate.prepare(root)
    owners = {}
    for warp in range(4):
        for lane in range(32):
            gid, tig = lane >> 2, lane & 3
            for r in range(4):
                stripe = 2 * warp + (tig >> 1) + 8 * r
                for c in range(4):
                    row = gid + 8 * (c >> 1)
                    token = 2 * (tig & 1) + (c & 1)
                    key = row, token, stripe
                    require(key not in owners, "duplicate FP32 chain owner")
                    owners[key] = warp, lane, r, c
    require(len(owners) == 16 * 4 * 32, "missing FP32 chain owner")
    coordinate_checks = 0
    for stripe in range(256):
        iteration, chain = divmod(stripe, 32)
        r, sub = divmod(chain, 8)
        warp, p = divmod(sub, 2)
        block = r + 4 * iteration
        require((block, 2 * warp + p) == divmod(stripe, 8), "K block/sub mismatch")
        a_fragment = {}
        b_fragment = {}
        for lane in range(32):
            gid, tig = lane >> 2, lane & 3
            for reg in range(4):
                row = gid + 8 * (reg & 1)
                first_k = 4 * tig + 16 * (reg >> 1)
                for byte in range(4):
                    value = first_k + byte
                    address = 144 * block + 16 + 32 * warp + value
                    key = row, value
                    require(key not in a_fragment, "duplicate A fragment byte")
                    a_fragment[key] = (row, address, 4 * p)
            for reg in range(2):
                for byte in range(4):
                    value = 4 * tig + 16 * reg + byte
                    b_fragment[gid, value] = ((gid & 3) * 5120 + 32 * stripe + value)
        require(len(a_fragment) == 16 * 32 and len(b_fragment) == 8 * 32,
                "incomplete fragment")
        for lane in range(32):
            gid, tig = lane >> 2, lane & 3
            if (tig >> 1) != p:
                continue
            for c in range(4):
                row = gid + 8 * (c >> 1)
                n = 2 * tig + (c & 1)
                token = n & 3
                for value in range(32):
                    native_byte = 144 * (stripe // 8) + 16 + 32 * ((stripe % 8) // 2) + value
                    require(a_fragment[row, value] == (row, native_byte, 4 * (stripe & 1)),
                            "native Q4 nibble coordinate mismatch")
                    require(b_fragment[n, value] == token * 5120 + 32 * stripe + value,
                            "native Q8 activation coordinate mismatch")
                    coordinate_checks += 1
    for byte in range(256):
        require((byte & 15, (byte >> 4) & 15) == (byte % 16, byte // 16),
                "nibble-pair mismatch")
    for bit in range(97):
        scales = [63 if bit == 96 else 0 for _ in range(8)]
        minima = scales.copy()
        if bit < 96:
            target = scales if bit < 48 else minima
            target[(bit % 48) // 6] = 1 << (bit % 6)
        packed = encode_metadata(scales, minima)
        for u in range(8):
            require(decode_metadata(packed, u) == (scales[u], minima[u]), "scale/min bits mismatch")
    updates = {}
    for iteration in range(5):
        for (row, token, chain), (warp, lane, r, _) in owners.items():
            p = (lane & 3) >> 1
            stripe = 32 * iteration + 8 * r + 2 * warp + p
            term = ("native_dot_integer", row, token, stripe // 8, stripe % 8,
                    ("half_d_times_integer_scale", stripe % 8),
                    ("half_dmin_times_integer_min", stripe % 8),
                    ("original_half2_activation", token, stripe))
            updates.setdefault((row, token, chain), []).append(term)
    terminal_owners = {}
    for row in range(16):
        for token in range(4):
            native = []
            for chain in range(32):
                sequence = updates[row, token, chain]
                require(len(sequence) == 5, "K5120 must have five updates per chain")
                require([term[3] * 8 + term[4] for term in sequence] ==
                        [chain + 32 * t for t in range(5)], "FP32 update order changed")
                node = ("FP32_zero",)
                for term in sequence:
                    node = add(node, term)
                native.append(node)
            stages = native.copy()
            for xor in (16, 8, 4, 2, 1):
                stages = [add(stages[s], stages[s ^ xor]) for s in range(32)]
            terminal = []
            for s in range(8):
                xor16_0 = add(native[s], native[s + 16])
                xor16_1 = add(native[s + 8], native[s + 24])
                terminal.append(add(xor16_0, xor16_1))
                key = s, row * 4 + token
                require(key not in terminal_owners, "terminal shared write race")
                terminal_owners[key] = True
            xor2 = [add(add(terminal[p], terminal[p ^ 4]),
                        add(terminal[p ^ 2], terminal[p ^ 6])) for p in range(2)]
            require(add(xor2[0], xor2[1]) == stages[0], "literal XOR reduction tree changed")
    require(len(terminal_owners) * 4 == 2048, "terminal allocation mismatch")
    return {
        "status": "SYMBOLIC_PASS",
        "scope": "source-coordinate and ordered expression proof; no CUDA execution",
        "checked_global_K32_stripes": 256,
        "actual_valid_K32_stripes_at_K5120": 160,
        "stripes_160_to_255": "symbolic coordinate extension only; never accessed by kernel",
        "native_candidate_coordinate_checks": coordinate_checks,
        "chains": len(owners),
        "updates_per_chain_at_K5120": 5,
        "chain_updates_per_CTA": 16 * 4 * 160,
        "FP32_partials_per_thread": 16,
        "terminal_shared_bytes": 2048,
        "literal_XOR_order": [16, 8, 4, 2, 1],
        "scale_min_basis_cases": 97,
        "nibble_byte_cases": 256,
        "dot_abs_integer_bound": 32 * 15 * 128,
        "native_float_expression": receipt["float_expression"],
        "unproven": ["compiler resource allocation", "actual MMA fragment execution",
                     "compiled FP32 instruction order", "packed-word reuse in SASS",
                     "FP32/BF16 oracle qualification", "full Gemm and helperpack cost",
                     "fused gate/up integration", "performance"],
    }


if __name__ == "__main__":
    print(json.dumps(prove(), indent=2))
