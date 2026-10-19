import argparse
import json
import pathlib
import struct
import subprocess


ROOT = pathlib.Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
MODEL = pathlib.Path(r'C:\AI\staging\qwen38-jonathan-ninfer\qwen38-jonathan-q4km-proposal.ninfer')
TENSORS = ('weight/000001', 'weight/000060')


def linux(path):
    path = pathlib.Path(path).resolve()
    return '/mnt/' + path.drive[0].lower() + '/' + '/'.join(path.parts[1:])


def tensor_directory(model):
    with model.open('rb') as stream:
        magic, length, artifact_id = struct.unpack('<8sQ16s', stream.read(32))
        if magic != b'NINFER\0\3':
            raise ValueError('NInfer v3 required')
        directory = json.loads(stream.read(length))
    payload = (32 + length + 4095) // 4096 * 4096
    items = {item['id']: item for item in directory['objects']}
    return artifact_id.hex(), payload, items


def execute(args, out_path, required=True):
    result = subprocess.run(['wsl.exe', '-d', 'Ubuntu-24.04', '-u', 'root', '--', *map(str, args)],
                            capture_output=True)
    out_path.write_bytes(result.stdout)
    out_path.with_suffix(out_path.suffix + '.stderr').write_bytes(result.stderr)
    if required and result.returncode:
        raise RuntimeError(f'probe failed with exit {result.returncode}; see {out_path}.stderr')
    return result


def build():
    includes = ['source/src', 'source/third_party/ggml-quants/src',
                'source/third_party/ggml-quants/include']
    command = ['/root/miniconda3/envs/trellis2/bin/nvcc', '-O3', '-std=c++17', '-arch=sm_86',
               '-allow-unsupported-compiler', '-Xptxas=-v',
               *['-I' + linux(ROOT / include) for include in includes],
               linux(ROOT / 'scales_probe.cu'), '-o', linux(RESULTS / 'scales-probe')]
    result = execute(command, RESULTS / 'scales-build.txt')
    if result.returncode:
        raise RuntimeError('nvcc build failed; see results/scales-build.txt.stderr')
    print(json.dumps({'binary': str(RESULTS / 'scales-probe'), 'gpu_executed': False}))


def run(args):
    binary = RESULTS / 'scales-probe'
    if not binary.is_file():
        raise SystemExit('run `python run_scales.py build` first')
    artifact_id, payload_base, items = tensor_directory(args.model)
    results = []
    for identity in ((args.tensor,) if args.tensor else TENSORS):
        item = items[identity]
        n, k = item['shape']
        if item['format'] != 'gguf_q4_k' or k != 5120 or n % 8 or item['bytes'] != n * (k // 256) * 144:
            raise ValueError(f'{identity} is not a complete K5120 GGUF Q4_K tensor')
        offset = payload_base + item['offset']
        suffix = 'bench' if args.action == 'bench' else 'scan'
        output_path = RESULTS / f'scales-{identity.replace("/", "-")}-{suffix}.jsonl'
        command = [linux(binary), linux(args.model), offset, n, k, identity, suffix]
        result = execute(command, output_path)
        lines = [json.loads(line) for line in result.stdout.decode().splitlines() if line.strip()]
        for row in lines:
            row.update({'tensor': identity, 'source': str(args.model.resolve()),
                        'artifact_id': artifact_id, 'absolute_payload_offset': offset})
        if args.action == 'scan':
            if len(lines) != 1 or not lines[0].get('cpu_all_payload_bytes_exact'):
                raise RuntimeError(f'CPU round trip failed for {identity}')
            result_path = RESULTS / f'scales-{identity.replace("/", "-")}.json'
            result_path.write_text(json.dumps(lines[0], indent=2) + '\n')
        else:
            if len(lines) != 5 or not all(row.get('fp32_output_bit_exact') and row.get('bf16_output_bit_exact')
                                          for row in lines):
                raise RuntimeError(f'GPU arithmetic comparison failed for {identity}')
        results.extend(lines)
    report = {'format': 'q4k-max63-elision-64block-v2',
              'eligibility': 'one 63 in each eight-scale and eight-min vector; other blocks keep exact 96-bit metadata',
              'groups': '64 blocks, uint64 eligibility bitmap, uint32 16-byte-aligned metadata offset',
              'addressing': 'localBlock*96 - 6*popcount(eligiblePrefix)',
              'metadata_bits_eligible': 90,
              'metadata_bits_fallback': 96,
              'cpu_full_payload_roundtrip': args.action == 'scan',
              'gpu_run': args.action == 'bench',
              'tensors': results}
    summary_path = RESULTS / f'scales-{args.action}-summary.json'
    summary_path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('build', 'scan', 'bench'))
    parser.add_argument('--model', type=pathlib.Path, default=MODEL)
    parser.add_argument('--tensor', choices=TENSORS)
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    if args.action == 'build':
        build()
    else:
        run(args)


if __name__ == '__main__':
    main()
