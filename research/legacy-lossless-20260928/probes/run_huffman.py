import argparse
import json
import pathlib
import struct
import subprocess


ROOT = pathlib.Path(__file__).resolve().parent
RESULTS = ROOT / 'results' / 'huffman'
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
    if item['format'] != 'gguf_q4_k' or k != 5120 or n % 8 or item['bytes'] != n * (k // 256) * 144:
        raise ValueError('This prototype requires GGUF Q4_K, K5120 and complete eight-row tiles')
    offset = (32 + length + 4095) // 4096 * 4096 + item['offset']
    return item, offset, n, k, aid.hex()


def invocation(model, offset, n, k, label, mode, artifact):
    return [linux(ROOT / 'huffman-probe'), linux(model), offset, n, k, label, mode, linux(artifact)]


def selftest(model):
    item, offset, n, k, aid = tensor(model, 'weight/000060')
    tilebytes = 8 * (k // 256) * 144
    with model.open('rb') as f:
        f.seek(offset)
        original = f.read(tilebytes)
    fixture = bytearray(original * 3)
    for tile in range(3):
        for block in range(tilebytes // 144):
            start = tile * tilebytes + block * 144 + 16
            for byte in range(128):
                fixture[start + byte] = (block * 128 + byte) % 256 if tile == 0 else (0 if tile == 1 else (17 if byte % 11 else 255))
    raw = RESULTS / 'fixture.raw'
    artifact = RESULTS / 'fixture.nlgq4'
    raw.write_bytes(fixture)
    execute(invocation(raw, 0, 24, k, 'fixture', 'pack', artifact), 'fixture-pack.json')
    execute(invocation(raw, 0, 24, k, 'fixture', 'bench', artifact), 'fixture-gpu.jsonl')
    rows = [r for r in map(json.loads, (RESULTS / 'fixture-gpu.jsonl').read_text().splitlines()) if 'samples_ms' in r]
    assert all(r['raw_tiles'] >= 1 and r['fp32_output_bit_exact'] and r['bf16_output_bit_exact'] for r in rows)
    data = artifact.read_bytes()
    cases = {'version': (6, b'\x02'), 'shape': (8, struct.pack('<I', 32)), 'frequency': (32, b'\0\0'), 'payload': (len(data)-100, bytes([data[-100] ^ 127]))}
    outcomes = {}
    for label, (at, replacement) in cases.items():
        bad = bytearray(data)
        bad[at:at+len(replacement)] = replacement
        path = RESULTS / ('fixture-bad-' + label + '.nlhq4')
        path.write_bytes(bad)
        p = execute(invocation(raw, 0, 24, k, 'fixture', 'bench', path), 'reject-' + label + '.txt', False)
        assert p.returncode != 0, label
        outcomes[label] = p.returncode
        path.unlink()
    path = RESULTS / 'fixture-truncated.nlgq4'
    path.write_bytes(data[:-1])
    p = execute(invocation(raw, 0, 24, k, 'fixture', 'bench', path), 'reject-truncated.txt', False)
    assert p.returncode != 0
    outcomes['truncated'] = p.returncode
    path.unlink()
    (RESULTS / 'selftest.json').write_text(json.dumps({'all_nibble_values': True, 'raw_fallback': True, 'compressed_tiles': True, 'compact_stream_tails': True, 'cpu_gpu_bytes_exact': True, 'fp32_bf16_exact': True, 'T': [1,2,3,4,8], 'rejected': outcomes}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['build', 'pack', 'bench', 'selftest'])
    parser.add_argument('--model', type=pathlib.Path, default=MODEL)
    parser.add_argument('--tensor', default='weight/000060')
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True, parents=True)
    if args.action == 'build':
        includes = ['source/src', 'source/third_party/ggml-quants/src', 'source/third_party/ggml-quants/include']
        execute(['/root/miniconda3/envs/trellis2/bin/nvcc', '-O3', '-std=c++17', '-arch=sm_86', '-allow-unsupported-compiler', '--extended-lambda', '-Xptxas=-v', *['-I'+linux(ROOT/p) for p in includes], linux(ROOT/'huffman_probe.cu'), '-o', linux(ROOT/'huffman-probe')], 'build.txt')
    elif args.action == 'selftest':
        selftest(args.model)
    else:
        item, offset, n, k, aid = tensor(args.model, args.tensor)
        label = 'huffman-' + args.tensor.replace('/', '-')
        artifact = RESULTS / (label + '.nlhq4')
        execute(invocation(args.model, offset, n, k, label, args.action, artifact), label + '-' + args.action + ('.json' if args.action == 'pack' else '.jsonl'))
        manifest = {'source': str(args.model.resolve()), 'artifact_id': aid, 'tensor': item, 'absolute_payload_offset': offset, 'experimental_layout': 'gguf_q4_k_cta8_warp_rans_v1', 'artifact': str(artifact)}
        (RESULTS / (label+'-manifest.json')).write_text(json.dumps(manifest, indent=2))
    print(f'{args.action}: completed; results in {RESULTS}')


if __name__ == '__main__':
    main()
