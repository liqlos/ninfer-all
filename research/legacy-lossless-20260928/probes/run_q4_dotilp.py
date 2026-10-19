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


def build(toolchain):
    includes = ['source/src', 'source/third_party/ggml-quants/src',
                'source/third_party/ggml-quants/include']
    binary = RESULTS / ('q4-dotilp-probe-cu13' if toolchain == 'cu13' else 'q4-dotilp-probe')
    if toolchain == 'cu13':
        command = ['docker', 'run', '--rm', '-v', f'{linux(ROOT)}:/experiment',
                   '--entrypoint', '/usr/local/cuda/bin/nvcc', 'ninfer-lossless-toolchain:cu13',
                   '-O3', '-DNDEBUG', '-std=c++20', '-arch=sm_86', '-lineinfo', '-DNINFER_SM8X_COMPAT=1', '-Xptxas=-v',
                   *['-I/experiment/' + include for include in includes],
                   '/experiment/q4_dotilp_probe.cu', '-o', f'/experiment/results/{binary.name}']
    else:
        command = ['/root/miniconda3/envs/trellis2/bin/nvcc', '-O3', '-std=c++17', '-arch=sm_86',
                   '-allow-unsupported-compiler', '-Xptxas=-v',
                   *['-I' + linux(ROOT / include) for include in includes],
                   linux(ROOT / 'q4_dotilp_probe.cu'), '-o', linux(binary)]
    result = execute(command, RESULTS / f'q4-dotilp-build-{toolchain}.txt')
    if result.returncode:
        raise RuntimeError(f'nvcc build failed; see results/q4-dotilp-build-{toolchain}.txt.stderr')
    print(json.dumps({'binary': str(binary), 'toolchain': toolchain, 'gpu_executed': False}))


def run(args):
    binary = RESULTS / ('q4-dotilp-probe-cu13' if args.toolchain == 'cu13' else 'q4-dotilp-probe')
    if not binary.is_file():
        raise SystemExit('run `python run_q4_dotilp.py build --toolchain cu13` first')
    artifact_id, payload_base, items = tensor_directory(args.model)
    results = []
    for identity in ((args.tensor,) if args.tensor else TENSORS):
        item = items[identity]
        n, k = item['shape']
        if item['format'] != 'gguf_q4_k' or k != 5120 or n % 8 or item['bytes'] != n * (k // 256) * 144:
            raise ValueError(f'{identity} is not a complete K5120 GGUF Q4_K tensor')
        offset = payload_base + item['offset']
        suffix = 'bench' if args.action == 'bench' else 'scan'
        output_path = RESULTS / f'q4-dotilp-{identity.replace("/", "-")}-{suffix}.jsonl'
        command = [linux(binary), linux(args.model), offset, n, k, identity, suffix]
        if args.activation:
            command.append(linux(args.activation))
        result = execute(command, output_path)
        lines = [json.loads(line) for line in result.stdout.decode().splitlines() if line.strip()]
        for row in lines:
            row.update({'tensor': identity, 'source': str(args.model.resolve()),
                        'artifact_id': artifact_id, 'absolute_payload_offset': offset})
        if args.action == 'scan':
            if (len(lines) != 1 or not lines[0].get('cpu_all_payload_bytes_exact') or
                    not lines[0].get('cpu_fixed_width_halves_exact') or not lines[0].get('cpu_dot_ilp_exact') or
                    lines[0].get('cpu_dot_q4xq8_random_trials') != 100000):
                raise RuntimeError(f'CPU round trip failed for {identity}')
            result_path = RESULTS / f'q4-dotilp-{identity.replace("/", "-")}.json'
            result_path.write_text(json.dumps(lines[0], indent=2) + '\n')
        else:
            if len(lines) != (1 if args.activation else 5) or not all(row.get('fp32_output_bit_exact') and row.get('bf16_output_bit_exact')
                                          for row in lines):
                raise RuntimeError(f'GPU arithmetic comparison failed for {identity}')
        results.extend(lines)
    report = {'format': 'q4k-fixed-fast-dotilp-v1',
              'encoding': 'per tensor compile-time fixed d/dmin exponent bases and widths; signless mantissa10 plus exponent delta; code extraction uses byte loads and a fixed-width crossing predicate',
              'planes': 'Q4 data unchanged; 12B/block scales/minima unchanged; separate fixed-stride d and dmin bitstreams',
              'cpu_roundtrip': 'all source Q4_K block bytes reconstructed exactly before any GPU launch',
              'cpu_full_payload_roundtrip': args.action == 'scan',
              'gpu_run': args.action == 'bench',
              'activation': str(args.activation) if args.activation else None,
              'tensors': results}
    summary_path = RESULTS / f'q4-dotilp-{args.action}-summary.json'
    summary_path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('build', 'scan', 'bench'))
    parser.add_argument('--model', type=pathlib.Path, default=MODEL)
    parser.add_argument('--tensor', choices=TENSORS)
    parser.add_argument('--activation', type=pathlib.Path)
    parser.add_argument('--toolchain', choices=('cu12', 'cu13'), default='cu13')
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    if args.action == 'build':
        build(args.toolchain)
    else:
        run(args)


if __name__ == '__main__':
    main()
