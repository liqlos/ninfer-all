import argparse
import json
import pathlib
import struct
import subprocess


ROOT = pathlib.Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
MODEL = pathlib.Path(r'C:\AI\staging\qwen38-jonathan-ninfer\qwen38-jonathan-q4km-proposal.ninfer')


def linux(path):
    path = pathlib.Path(path).resolve()
    return '/mnt/' + path.drive[0].lower() + '/' + '/'.join(path.parts[1:])


def execute(args, output, required=True):
    p = subprocess.run(['wsl.exe', '-d', 'Ubuntu-24.04', '-u', 'root', '--', *map(str, args)], capture_output=True)
    (RESULTS / output).write_bytes(p.stdout)
    (RESULTS / (output + '.stderr')).write_bytes(p.stderr)
    if required and p.returncode:
        raise RuntimeError(f'{output}: exit {p.returncode}; see stderr')
    return p


def tensor(model, identity):
    with model.open('rb') as f:
        magic, length, aid = struct.unpack('<8sQ16s', f.read(32))
        if magic != b'NINFER\0\3':
            raise ValueError('NInfer v3 required')
        directory = json.loads(f.read(length))
    item = next(t for t in directory['objects'] if t['id'] == identity)
    n, k = item['shape']
    if item['format'] != 'gguf_q4_k' or k != 5120 or n % 32 or item['bytes'] != n * (k // 256) * 144:
        raise ValueError('Word-rANS prototype requires GGUF Q4_K, K5120 and complete 32-row tiles')
    offset = (32 + length + 4095) // 4096 * 4096 + item['offset']
    return item, offset, n, k, aid.hex()


def invocation(model, offset, n, k, label, mode, artifact):
    return [linux(ROOT / 'word-rans-probe'), linux(model), offset, n, k, label, mode, linux(artifact)]


def prepare_selftest():
    RESULTS.mkdir(exist_ok=True)
    n = 96
    k = 5120
    row_bytes = k // 256 * 144
    fixture = bytearray(n * row_bytes)
    for row in range(n):
        tile = row // 32
        local_row = row % 32
        for block in range(k // 256):
            base = row * row_bytes + block * 144
            fixture[base:base + 4] = b'\0\x3c\0\0'
            fixture[base + 4:base + 16] = bytes([1] * 12)
            for byte in range(128):
                i = local_row * 20 * 128 + block * 128 + byte
                if tile == 0:
                    value = i & 255
                elif tile == 1:
                    value = 0
                else:
                    value = 255 if i % 97 == 0 else (1 if i % 11 == 0 else 0)
                fixture[base + 16 + byte] = value
    raw = RESULTS / 'word-rans-fixture.raw'
    artifact = RESULTS / 'word-rans-fixture.nlwran'
    raw.write_bytes(fixture)
    pack_output = RESULTS / 'word-rans-fixture-pack.json'
    execute(invocation(raw, 0, n, k, 'word-rans-fixture', 'pack', artifact), pack_output.name)
    packed = json.loads(pack_output.read_text())
    if not packed['cpu_all_bytes_exact'] or packed['raw_tiles'] < 1 or packed['coded_tiles'] < 1:
        raise RuntimeError('wordwise fixture must contain exact raw fallback and compressed tiles')
    original = artifact.read_bytes()
    payload_base = 32 + 512 + (n // 32 + 1) * 8
    index = [struct.unpack_from('<Q', original, 32 + 512 + i * 8)[0] for i in range(n // 32 + 1)]
    compressed_tile = next(i for i in range(n // 32) if not (index[i] >> 63))
    tile_start = index[compressed_tile]
    if tile_start >= len(original) - payload_base:
        raise RuntimeError('fixture compressed tile index is invalid')
    cases = {
        'version': (6, b'\x02'),
        'shape': (8, struct.pack('<I', 32)),
        'frequency': (32, b'\0\0'),
        'offset': (32 + 512 + 8, struct.pack('<Q', len(original) + 1)),
        'payload': (payload_base + tile_start, bytes([original[payload_base + tile_start] ^ 0x7f])),
    }
    rejected = {}
    for name, (at, replacement) in cases.items():
        bad = bytearray(original)
        bad[at:at + len(replacement)] = replacement
        path = RESULTS / ('word-rans-fixture-bad-' + name + '.nlwran')
        path.write_bytes(bad)
        result = execute(invocation(raw, 0, n, k, 'word-rans-fixture', 'bench', path), 'word-rans-reject-' + name + '.txt', False)
        rejected[name] = result.returncode
        if result.returncode == 0:
            raise RuntimeError('corrupt wordwise artifact was accepted: ' + name)
        path.unlink()
    path = RESULTS / 'word-rans-fixture-truncated.nlwran'
    path.write_bytes(original[:-1])
    result = execute(invocation(raw, 0, n, k, 'word-rans-fixture', 'bench', path), 'word-rans-reject-truncated.txt', False)
    rejected['truncated'] = result.returncode
    if result.returncode == 0:
        raise RuntimeError('truncated wordwise artifact was accepted')
    path.unlink()
    (RESULTS / 'word-rans-fixture-selftest-prepared.json').write_text(json.dumps({'n': n, 'k': k, 'coded_tiles': packed['coded_tiles'], 'raw_tiles': packed['raw_tiles'], 'cpu_all_bytes_exact': True, 'rejected': rejected}, indent=2))
    return raw, artifact, n, k


def selftest():
    raw, artifact, n, k = prepare_selftest()
    result_path = RESULTS / 'word-rans-fixture-gpu.jsonl'
    execute(invocation(raw, 0, n, k, 'word-rans-fixture', 'bench', artifact), result_path.name)
    rows = [json.loads(line) for line in result_path.read_text().splitlines()]
    if [row['T'] for row in rows] != [1, 2, 3, 4, 8]:
        raise RuntimeError('wordwise GPU selftest missing T coverage')
    if not all(row['fp32_output_bit_exact'] and row['bf16_output_bit_exact'] for row in rows):
        raise RuntimeError('wordwise GPU operator output mismatch')
    if not rows[0]['gpu_all_payload_bytes_exact']:
        raise RuntimeError('wordwise GPU payload reconstruction was not verified')
    (RESULTS / 'word-rans-fixture-selftest.json').write_text(json.dumps({'all_byte_alphabet': True, 'zero_stream': True, 'skew_stream': True, 'raw_fallback': True, 'compressed_tiles': True, 'cpu_all_bytes_exact': True, 'gpu_all_bytes_exact': True, 'fp32_bf16_exact': True, 'T': [1, 2, 3, 4, 8], 'graphs': True, 'corruptions_rejected': True}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['build', 'pack', 'bench', 'prepare-selftest', 'selftest'])
    parser.add_argument('--model', type=pathlib.Path, default=MODEL)
    parser.add_argument('--tensor', default='weight/000060')
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    if args.action == 'build':
        includes = ['source/src', 'source/third_party/ggml-quants/src', 'source/third_party/ggml-quants/include']
        execute(['/root/miniconda3/envs/trellis2/bin/nvcc', '-O3', '-std=c++17', '-arch=sm_86', '-allow-unsupported-compiler', '--extended-lambda', '-Xptxas=-v', *['-I' + linux(ROOT / p) for p in includes], linux(ROOT / 'word_rans_probe.cu'), '-o', linux(ROOT / 'word-rans-probe')], 'word-rans-build.txt')
        print(f'build: completed; results in {RESULTS}')
        return
    if args.action == 'prepare-selftest':
        prepare_selftest()
        print(f'prepare-selftest: completed; CPU results in {RESULTS}')
        return
    if args.action == 'selftest':
        selftest()
        print(f'selftest: completed; results in {RESULTS}')
        return
    item, offset, n, k, aid = tensor(args.model, args.tensor)
    label = args.tensor.replace('/', '-')
    artifact = RESULTS / ('word-rans-' + label + '.nlwran')
    action = 'pack' if args.action == 'pack' else 'bench'
    execute(invocation(args.model, offset, n, k, label, action, artifact), 'word-rans-' + label + '-' + action + ('.json' if action == 'pack' else '.jsonl'))
    manifest = {'source': str(args.model.resolve()), 'artifact_id': aid, 'tensor': item, 'absolute_payload_offset': offset, 'experimental_layout': 'gguf_q4_k_32row_wordrans_v1', 'artifact': str(artifact)}
    (RESULTS / ('word-rans-' + label + '-manifest.json')).write_text(json.dumps(manifest, indent=2))
    print(f'{action}: completed; results in {RESULTS}')


if __name__ == '__main__':
    main()
