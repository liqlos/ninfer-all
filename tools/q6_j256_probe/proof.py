import collections


def require(condition, message):
    if not condition:
        raise ValueError(message)


def run():
    outputs = {}
    fragment_checks = 0
    for warp in range(16):
        for lane in range(32):
            for j in range(16):
                for l in range(4):
                    row = 16 * (warp // 2) + 8 * (l // 2) + lane // 4
                    col = 16 * j + 8 * (warp % 2) + 2 * (lane % 4) + l % 2
                    require((row, col) not in outputs, "duplicate output writer")
                    outputs[row, col] = (warp, lane, j * 4 + l)
                    native_warp = 2 * (row // 32) + (col % 16) // 8
                    native_n = (row % 32) // 16
                    native_l = 2 * ((row % 16) // 8) + col % 2
                    native_lane = 4 * (row % 8) + (col % 8) // 2
                    native_j0 = (col % 128) // 16 * 16
                    require(native_lane == lane and native_l == l, "C fragment differs")
                    require(32 * (native_warp // 2) + 16 * native_n == 16 * (warp // 2),
                            "A minitile row differs")
                    require(128 * (col // 128) + native_j0 + 8 * (native_warp % 2) ==
                            16 * j + 8 * (warp % 2), "B minitile column differs")
                    for half in range(2):
                        for update in range(4):
                            for pair in range(2):
                                k0 = 128 * half + 32 * update + 16 * pair
                                scale = 8 * half + 2 * update + pair
                                require(k0 // 16 == scale, "native signed scale group differs")
                                require((32 * half + 8 * update) // 16 * 4 +
                                        (2 * update + pair) % 4 == scale,
                                        "packed scale byte differs")
                                fragment_checks += 1
    require(set(outputs) == {(i, j) for i in range(128) for j in range(256)},
            "output coverage differs")
    a_words = {(8 * reg + lane // 4, lane % 4): (lane, reg)
               for lane in range(32) for reg in range(2)}
    b_words = {(lane // 4, lane % 4): lane for lane in range(32)}
    require(len(a_words) == 64 and len(b_words) == 32, "MMA input fragment coverage differs")
    scalar_terms = 0
    for row in range(16):
        for col in range(8):
            for k in range(16):
                a_lane, a_reg = a_words[row, k // 4]
                b_lane = b_words[col, k // 4]
                require((a_lane // 4 + 8 * a_reg, 4 * (a_lane % 4) + k % 4) == (row, k),
                        "ldmatrix m8n8.x2 fragment differs")
                require((b_lane // 4, 4 * (b_lane % 4) + k % 4) == (col, k),
                        "generic B fragment differs")
                scalar_terms += 1
    writers = collections.Counter()
    for warp in range(16):
        for lane in range(32):
            tid = 32 * warp + lane
            if tid < 256:
                writers["ids", tid] += 1
            for i0 in range(0, 128, 16):
                row = i0 + warp
                kq0 = 2 * lane - lane % 16
                writers["x", row * 76 + kq0] += 1
                writers["x", row * 76 + kq0 + 16] += 1
            if tid < 128:
                writers["x", tid * 76 + 64] += 1
            row = warp * 8 + lane // 4
            writers["x", row * 76 + 65 + lane % 4] += 1
            for l0 in range(0, 256 * 36, 512):
                writers["y", l0 + tid] += 1
    require(all(count == 1 for count in writers.values()), "duplicate shared writer")
    required = {(kind, at) for kind, extent in (("ids", 256), ("y", 256 * 36))
                for at in range(extent)}
    required.update(("x", row * 76 + word) for row in range(128) for word in range(69))
    require(set(writers) == required, "missing shared producer or padding write")
    addresses = set(range(256))
    addresses.update(256 + at for kind, at in writers if kind == "y")
    addresses.update(256 + 256 * 36 + at for kind, at in writers if kind == "x")
    require(len(addresses) == len(writers) and max(addresses) < 76800 // 4,
            "shared planes overlap or exceed 76800 bytes")
    decode_checks = 0
    for lane in range(32):
        half = lane // 16
        for high_nibble in range(2):
            word = 2 * lane - lane % 16 + 16 * high_nibble
            for byte in range(4):
                k = 4 * word + byte
                local = k % 128
                ql_index = 64 * half + local % 32 + 32 * ((local // 32) % 2)
                qh_index = 32 * half + local % 32
                qh_shift = 2 * (local // 32)
                require(ql_index == 4 * lane + byte, "QL native byte address differs")
                require(qh_index == 4 * (8 * (lane // 16) + lane % 8) + byte,
                        "QH native byte address differs")
                require(qh_shift == ((lane & 8) >> 2) + 4 * high_nibble,
                        "QH native bit pair differs")
                for q in range(64):
                    ql_word = (q & 15) << (8 * byte + 4 * high_nibble)
                    qh_word = (q >> 4) << (8 * byte + qh_shift)
                    low = (ql_word >> (4 * high_nibble)) & 0x0F0F0F0F
                    shifted = qh_word >> ((lane & 8) >> 2)
                    high = (shifted << 4 if high_nibble == 0 else shifted) & 0x30303030
                    decoded = ((low | high) >> (8 * byte) & 255) - 32
                    require(-32 <= decoded <= 31 and decoded == q - 32,
                            "native signed Q6 code differs")
                    decode_checks += 1
    return {"status": "PASS_SYMBOLIC_OWNERSHIP", "outputs": len(outputs),
            "outputs_per_thread": 64, "fragment_group_checks": fragment_checks,
            "mma_scalar_terms": scalar_terms, "q6_bit_checks": decode_checks,
            "shared_unique_writers": len(writers), "dynamic_shared_bytes": 76800,
            "K256_blocks": 68,
            "CTA_threads": 512, "CTAs": 200, "native_backing_bytes_per_block": 210,
            "ordered_updates_per_K128": 4, "k16_MMA_per_update": 2,
            "logical_source_array_elements": 98, "GPU_numerics_proven": False}
