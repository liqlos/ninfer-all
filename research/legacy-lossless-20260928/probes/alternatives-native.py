import ctypes
import ctypes.util
import json
import mmap
import pathlib
import struct
import time
import zlib


ROOT = pathlib.Path('/mnt/c/AI/tuning/ninfer-lossless-20260928')
ANALYSIS = json.loads((ROOT / 'results/analysis.json').read_text())
MODEL = pathlib.Path('/mnt/c/' + ANALYSIS['model'][3:].replace('\\', '/'))
OUTPUT = ROOT / 'alternatives-codecs.json'
TILE_BLOCKS = 32
SAMPLE_BLOCKS = 512
SAMPLE_FRACTIONS = (0.0, 0.333, 0.666, 0.99)


def lib(name):
    path = ctypes.util.find_library(name)
    if not path:
        raise RuntimeError(f'cannot locate {name}')
    return ctypes.CDLL(path)


zstd = lib('zstd')
zstd.ZSTD_compressBound.argtypes = [ctypes.c_size_t]
zstd.ZSTD_compressBound.restype = ctypes.c_size_t
zstd.ZSTD_compress.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
zstd.ZSTD_compress.restype = ctypes.c_size_t
zstd.ZSTD_decompress.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t]
zstd.ZSTD_decompress.restype = ctypes.c_size_t
zstd.ZSTD_isError.argtypes = [ctypes.c_size_t]
zstd.ZSTD_isError.restype = ctypes.c_uint

lz4 = lib('lz4')
lz4.LZ4_compressBound.argtypes = [ctypes.c_int]
lz4.LZ4_compressBound.restype = ctypes.c_int
lz4.LZ4_compress_default.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
lz4.LZ4_compress_default.restype = ctypes.c_int
lz4.LZ4_decompress_safe.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
lz4.LZ4_decompress_safe.restype = ctypes.c_int


def zstd_compress(data, level):
    out = ctypes.create_string_buffer(zstd.ZSTD_compressBound(len(data)))
    src = ctypes.create_string_buffer(data)
    n = zstd.ZSTD_compress(out, len(out), src, len(data), level)
    if zstd.ZSTD_isError(n):
        raise RuntimeError('ZSTD_compress failed')
    return out.raw[:n]


def zstd_decompress(data, size):
    out = ctypes.create_string_buffer(size)
    src = ctypes.create_string_buffer(data)
    n = zstd.ZSTD_decompress(out, size, src, len(data))
    if zstd.ZSTD_isError(n) or n != size:
        raise RuntimeError('ZSTD_decompress failed')
    return out.raw[:n]


def lz4_compress(data):
    out = ctypes.create_string_buffer(lz4.LZ4_compressBound(len(data)))
    src = ctypes.create_string_buffer(data)
    n = lz4.LZ4_compress_default(src, out, len(data), len(out))
    if n <= 0:
        raise RuntimeError('LZ4_compress_default failed')
    return out.raw[:n]


def lz4_decompress(data, size):
    out = ctypes.create_string_buffer(size)
    src = ctypes.create_string_buffer(data)
    n = lz4.LZ4_decompress_safe(src, out, len(data), size)
    if n != size:
        raise RuntimeError('LZ4_decompress_safe failed')
    return out.raw[:n]


def tile_shuffle(tile, nblocks, stride):
    return b''.join(tile[row * stride + col:row * stride + col + 1] for col in range(stride) for row in range(nblocks))


def q4_nibble_plane(tile, nblocks):
    stride = 144
    meta = b''.join(tile[row * stride:row * stride + 16] for row in range(nblocks))
    payload = []
    for byte_column in range(16, stride):
        for row in range(nblocks):
            b = tile[row * stride + byte_column]
            payload.append(b & 15)
            payload.append(b >> 4)
    return meta + bytes(payload)


def huffman_bits(counts):
    import heapq
    heap = [int(n) for n in counts if n]
    heapq.heapify(heap)
    if len(heap) <= 1:
        return 0
    total = 0
    while len(heap) > 1:
        a = heapq.heappop(heap)
        b = heapq.heappop(heap)
        total += a + b
        heapq.heappush(heap, a + b)
    return total


def make_huffman_codes(counts):
    import heapq
    heap = []
    serial = 0
    for symbol, count in enumerate(counts):
        if count:
            heapq.heappush(heap, (int(count), serial, symbol))
            serial += 1
    if len(heap) == 1:
        return {heap[0][2]: '0'}
    while len(heap) > 1:
        wa, _, a = heapq.heappop(heap)
        wb, _, b = heapq.heappop(heap)
        heapq.heappush(heap, (wa + wb, serial, (a, b)))
        serial += 1
    lengths = {}
    def visit(node, depth):
        if isinstance(node, int):
            lengths[node] = max(depth, 1)
        else:
            visit(node[0], depth + 1)
            visit(node[1], depth + 1)
    visit(heap[0][2], 0)
    ordered = sorted(lengths.items(), key=lambda item: (item[1], item[0]))
    codes = {}
    code = 0
    previous = ordered[0][1]
    for i, (symbol, length) in enumerate(ordered):
        if i:
            code = (code + 1) << (length - previous)
        codes[symbol] = format(code, f'0{length}b')
        previous = length
    return codes


def huffman_encode(symbols, codes):
    bitbuf = 0
    nbits = 0
    out = bytearray()
    for symbol in symbols:
        code = codes[symbol]
        bitbuf = (bitbuf << len(code)) | int(code, 2)
        nbits += len(code)
        while nbits >= 8:
            nbits -= 8
            out.append((bitbuf >> nbits) & 255)
            bitbuf &= (1 << nbits) - 1
    if nbits:
        out.append((bitbuf << (8 - nbits)) & 255)
    return bytes(out), nbits


def huffman_decode(packed, symbol_count, codes, final_bits):
    reverse = {v: k for k, v in codes.items()}
    out = []
    code = ''
    total_bits = (len(packed) - 1) * 8 + (final_bits or 8)
    for bit_index in range(total_bits):
        code += str((packed[bit_index // 8] >> (7 - bit_index % 8)) & 1)
        symbol = reverse.get(code)
        if symbol is not None:
            out.append(symbol)
            code = ''
            if len(out) == symbol_count:
                break
    if len(out) != symbol_count or code:
        raise AssertionError('Huffman round trip decode failed')
    return out


def q4_symbols(tile, nblocks):
    symbols = []
    for row in range(nblocks):
        base = row * 144 + 16
        for b in tile[base:base + 128]:
            symbols.append(b & 15)
            symbols.append(b >> 4)
    return symbols


def q4_repack(symbols):
    if len(symbols) != 256:
        raise AssertionError('expected 256 Q4_K codes')
    packed = bytearray(128)
    for i in range(128):
        packed[i] = symbols[2 * i] | (symbols[2 * i + 1] << 4)
    return bytes(packed)


def q6_symbols(tile, nblocks):
    symbols = []
    for row in range(nblocks):
        base = row * 210
        ql = tile[base:base + 128]
        qh = tile[base + 128:base + 192]
        decoded = [0] * 256
        for j in range(0, 256, 128):
            qlo = j // 2
            qhi = j // 4
            for lane in range(32):
                high = qh[qhi + lane]
                a = ql[qlo + lane]
                b = ql[qlo + lane + 32]
                decoded[j + lane] = (a & 15) | ((high & 3) << 4)
                decoded[j + lane + 32] = (b & 15) | (((high >> 2) & 3) << 4)
                decoded[j + lane + 64] = (a >> 4) | (((high >> 4) & 3) << 4)
                decoded[j + lane + 96] = (b >> 4) | (((high >> 6) & 3) << 4)
        symbols.extend(decoded)
    return symbols


def q6_repack(symbols):
    ql = bytearray(128)
    qh = bytearray(64)
    if len(symbols) != 256:
        raise AssertionError('expected 256 Q6_K codes')
    for j in range(0, 256, 128):
        qlo = j // 2
        qhi = j // 4
        for lane in range(32):
            a = symbols[j + lane]
            b = symbols[j + lane + 32]
            c = symbols[j + lane + 64]
            d = symbols[j + lane + 96]
            ql[qlo + lane] = (a & 15) | ((c & 15) << 4)
            ql[qlo + lane + 32] = (b & 15) | ((d & 15) << 4)
            qh[qhi + lane] = ((a >> 4) & 3) | (((b >> 4) & 3) << 2) | (((c >> 4) & 3) << 4) | (((d >> 4) & 3) << 6)
    return bytes(ql + qh)


def rice_bits(hist):
    vals = range(len(hist))
    best = None
    for k in range(8):
        bits = sum(int(c) * ((v >> k) + 1 + k) for v, c in zip(vals, hist))
        if best is None or bits < best[0]:
            best = (bits, k)
    return {'bits': best[0], 'k': best[1]}


def nibble_stats(tile, fmt):
    stride = 144 if fmt == 'gguf_q4_k' else 210
    codes = []
    for p in range(16, 144) if stride == 144 else range(128):
        for b in range(0, len(tile), stride):
            x = tile[b + p]
            codes.append(x & 15)
            codes.append(x >> 4)
    hist = [0] * 16
    for c in codes:
        hist[c] += 1
    return hist


def exact_codec(candidate, decompressor, expected):
    rebuilt = decompressor(candidate, len(expected))
    if rebuilt != expected:
        raise AssertionError('round trip mismatch')


def main():
    with MODEL.open('rb') as f:
        magic, directory_len, _ = struct.unpack('<8sQ16s', f.read(32))
        if magic != b'NINFER\0\3':
            raise RuntimeError(f'not a NINFER artifact: {magic!r}')
        directory = json.loads(f.read(directory_len))
        payload_start = (32 + directory_len + 4095) // 4096 * 4096
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        tensors = {t['id']: t for t in directory['objects'] if t['kind'] == 'tensor'}
        selected = ['weight/000000', 'weight/000001', 'weight/000060', 'weight/000746']
        results = {'model': str(MODEL), 'payload_offset_derived': payload_start, 'tile_gguf_blocks': TILE_BLOCKS, 'sample_gguf_blocks_per_region': SAMPLE_BLOCKS, 'library_versions': {}, 'tensors': []}
        results['library_versions'] = {'zstd_version': int(zstd.ZSTD_versionNumber()), 'lz4_version': int(lz4.LZ4_versionNumber())}
        for tid in selected:
            tensor = tensors[tid]
            fmt = tensor['format']
            stride = 144 if fmt == 'gguf_q4_k' else 210
            assert fmt in ('gguf_q4_k', 'gguf_q6_k')
            total_blocks = tensor['bytes'] // stride
            tile_count = SAMPLE_BLOCKS // TILE_BLOCKS
            all_tiles = {name: [] for name in ('raw_zstd_1', 'raw_zstd_3', 'raw_zstd_9', 'raw_lz4', 'raw_zlib_1', 'raw_zlib_6', 'shuffled_zstd_1', 'shuffled_zstd_3', 'shuffled_lz4', 'shuffled_zlib_1')}
            aggregate = {name: {'bytes': 0, 'exact_tiles': 0, 'compress_seconds': 0.0, 'decompress_seconds': 0.0} for name in all_tiles}
            code_hist = [0] * 16
            sampled_raw = 0
            quant_hist = [0] * (16 if fmt == 'gguf_q4_k' else 64)
            quant_huffman_tiles = []
            for frac in SAMPLE_FRACTIONS:
                base = min(int(total_blocks * frac), total_blocks - SAMPLE_BLOCKS)
                base -= base % TILE_BLOCKS
                view = memoryview(mm)[payload_start + tensor['offset'] + base * stride:payload_start + tensor['offset'] + (base + SAMPLE_BLOCKS) * stride]
                for ti in range(tile_count):
                    tile = bytes(view[ti * TILE_BLOCKS * stride:(ti + 1) * TILE_BLOCKS * stride])
                    shuffled = tile_shuffle(tile, TILE_BLOCKS, stride)
                    sampled_raw += len(tile)
                    hist = nibble_stats(tile, fmt)
                    code_hist = [a + b for a, b in zip(code_hist, hist)]
                    if fmt == 'gguf_q4_k':
                        symbols = q4_symbols(tile, TILE_BLOCKS)
                    else:
                        symbols = q6_symbols(tile, TILE_BLOCKS)
                    quant_huffman_tiles.append((tile, symbols))
                    for symbol in symbols:
                        quant_hist[symbol] += 1
                    variants = {
                        'raw_zstd_1': (tile, lambda b: zstd_compress(b, 1), zstd_decompress),
                        'raw_zstd_3': (tile, lambda b: zstd_compress(b, 3), zstd_decompress),
                        'raw_zstd_9': (tile, lambda b: zstd_compress(b, 9), zstd_decompress),
                        'raw_lz4': (tile, lz4_compress, lz4_decompress),
                        'raw_zlib_1': (tile, lambda b: zlib.compress(b, 1), lambda b, n: zlib.decompress(b)),
                        'raw_zlib_6': (tile, lambda b: zlib.compress(b, 6), lambda b, n: zlib.decompress(b)),
                        'shuffled_zstd_1': (shuffled, lambda b: zstd_compress(b, 1), zstd_decompress),
                        'shuffled_zstd_3': (shuffled, lambda b: zstd_compress(b, 3), zstd_decompress),
                        'shuffled_lz4': (shuffled, lz4_compress, lz4_decompress),
                        'shuffled_zlib_1': (shuffled, lambda b: zlib.compress(b, 1), lambda b, n: zlib.decompress(b)),
                    }
                    for name, (input_bytes, compressor, decompressor) in variants.items():
                        t0 = time.perf_counter()
                        packed = compressor(input_bytes)
                        aggregate[name]['compress_seconds'] += time.perf_counter() - t0
                        t0 = time.perf_counter()
                        rebuilt = decompressor(packed, len(input_bytes))
                        aggregate[name]['decompress_seconds'] += time.perf_counter() - t0
                        if rebuilt != input_bytes:
                            raise AssertionError(f'{name} roundtrip mismatch for {tid} tile {ti}')
                        aggregate[name]['bytes'] += min(len(packed), len(input_bytes)) + 8
                        aggregate[name]['exact_tiles'] += 1
                del view
            huff_payload_bits = huffman_bits(code_hist)
            nibbles = sum(code_hist)
            rice = rice_bits(code_hist)
            item = {'id': tid, 'format': fmt, 'roles': tensor.get('roles', []), 'tensor_bytes': tensor['bytes'], 'stride': stride, 'sample_raw_bytes': sampled_raw, 'sample_tiles': SAMPLE_BLOCKS // TILE_BLOCKS * len(SAMPLE_FRACTIONS), 'tile_bytes': TILE_BLOCKS * stride, 'metrics': {}}
            for name, stat in aggregate.items():
                item['metrics'][name] = {**stat, 'ratio_including_8_byte_tile_header_and_raw_fallback': stat['bytes'] / sampled_raw, 'savings_percent_including_header_and_raw_fallback': 100 * (1 - stat['bytes'] / sampled_raw)}
            item['histogram16'] = code_hist
            item['nibble_entropy_bits_symbol'] = -sum((x / nibbles) * __import__('math').log2(x / nibbles) for x in code_hist if x)
            item['static_huffman_nibble_estimate'] = {'payload_bytes': (huff_payload_bits + 7) // 8, 'table_bytes_per_tile_assumed': 16, 'tile_count': item['sample_tiles'], 'ratio_with_table': ((huff_payload_bits + 7) // 8 + 16 * item['sample_tiles']) / (nibbles / 2), 'warning': 'estimate only; canonical code lengths packed 4 bits per symbol, no GPU decoder measured'}
            item['rice_golomb_best_unary_nibble_estimate'] = {'k': rice['k'], 'payload_bits': rice['bits'], 'ratio_vs_4bit_codes': rice['bits'] / (nibbles * 4), 'warning': 'direct unsigned 0..15 mapping; excludes byte-stream metadata and framing'}
            if fmt in ('gguf_q4_k', 'gguf_q6_k'):
                codes = make_huffman_codes(quant_hist)
                codebook_bytes = len(quant_hist)
                stored = 0
                payload_only = 0
                raw_fallback_tiles = 0
                for tile, symbols in quant_huffman_tiles:
                    encoded, final_bits = huffman_encode(symbols, codes)
                    decoded = huffman_decode(encoded, len(symbols), codes, final_bits)
                    if decoded != symbols:
                        raise AssertionError(f'Huffman symbol mismatch for {tid}')
                    if fmt == 'gguf_q6_k':
                        for block in range(TILE_BLOCKS):
                            rebuilt_codes = decoded[block * 256:(block + 1) * 256]
                            rebuilt = q6_repack(rebuilt_codes)
                            base = block * 210
                            if rebuilt != tile[base:base + 192]:
                                raise AssertionError(f'Q6_K layout roundtrip mismatch for {tid} tile block {block}')
                    else:
                        for block in range(TILE_BLOCKS):
                            rebuilt = q4_repack(decoded[block * 256:(block + 1) * 256])
                            base = block * 144
                            if rebuilt != tile[base + 16:base + 144]:
                                raise AssertionError(f'Q4_K layout roundtrip mismatch for {tid} tile block {block}')
                    uncompressed = len(tile)
                    metadata_bytes = (16 if fmt == 'gguf_q4_k' else 18) * TILE_BLOCKS
                    packed_total = metadata_bytes + ((len(encoded) + 15) // 16) * 16 + 4
                    if packed_total >= uncompressed:
                        stored += uncompressed
                        raw_fallback_tiles += 1
                    else:
                        stored += packed_total
                        payload_only += len(encoded)
                table_and_offsets = codebook_bytes + item['sample_tiles'] * 4
                item['independent_tile_huffman'] = {'symbol_alphabet': len(quant_hist), 'global_canonical_code_lengths': [len(codes.get(i, '')) for i in range(len(quant_hist))], 'table_bytes_per_tensor_assumed': codebook_bytes, 'offset_bytes_per_tile_assumed': 4, 'tile_payload_alignment': 16, 'exact_symbol_roundtrip_tiles': len(quant_huffman_tiles), 'decoded_quant_codes': sum(quant_hist), 'encoded_payload_bytes_before_alignment': payload_only, 'raw_fallback_tiles': raw_fallback_tiles, 'stored_bytes_including_unchanged_scales_framing_alignment_and_raw_fallback': stored + table_and_offsets, 'ratio_to_sample_raw_bytes': (stored + table_and_offsets) / sampled_raw, 'savings_percent': 100 * (1 - (stored + table_and_offsets) / sampled_raw), 'note': 'CPU-only format feasibility; metadata bytes are copied unchanged; tile-local Huffman streams share one static canonical codebook per tensor'}
            results['tensors'].append(item)
        (ROOT / 'alternatives-codecs.json').write_text(json.dumps(results, indent=2))
        print(json.dumps({'model': str(MODEL), 'payload_offset_derived': payload_start, 'tensors': [{ 'id': t['id'], 'format': t['format'], 'metrics': {k: round(v['ratio_including_8_byte_tile_header_and_raw_fallback'], 5) for k, v in t['metrics'].items()}, 'huffman_ratio_estimate': round(t['static_huffman_nibble_estimate']['ratio_with_table'], 5), 'rice_ratio': round(t['rice_golomb_best_unary_nibble_estimate']['ratio_vs_4bit_codes'], 5)} for t in results['tensors']]}, indent=2))


if __name__ == '__main__':
    main()
