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
    return artifact_id.hex(), payload, {item['id']: item for item in directory['objects']}


def execute(command, output):
    result = subprocess.run(['wsl.exe', '-d', 'Ubuntu-24.04', '-u', 'root', '--', *map(str, command)], capture_output=True)
    output.write_bytes(result.stdout)
    output.with_suffix(output.suffix + '.stderr').write_bytes(result.stderr)
    if result.returncode:
        raise RuntimeError(f'command failed with exit {result.returncode}; see {output}.stderr')
    return result.stdout.decode()


def build():
    includes = ['source/src', 'source/third_party/ggml-quants/src', 'source/third_party/ggml-quants/include']
    binary = RESULTS / 'q4-fastwidepairmeta-probe-cu13'
    command = ['docker', 'run', '--rm', '-v', f'{linux(ROOT)}:/experiment', '--entrypoint', '/usr/local/cuda/bin/nvcc',
               'ninfer-lossless-toolchain:cu13', '-O3', '-DNDEBUG', '-std=c++20', '-arch=sm_86', '-lineinfo',
               '-DNINFER_SM8X_COMPAT=1', '-Xptxas=-v', *['-I/experiment/' + item for item in includes],
               '/experiment/q4_fastwidepairmeta_probe.cu', '-o', f'/experiment/results/{binary.name}']
    execute(command, RESULTS / 'q4-fastwidepairmeta-build-cu13.txt')
    print(json.dumps({'binary': str(binary), 'gpu_executed': False}))


def scan(args):
    binary = RESULTS / 'q4-fastwidepairmeta-probe-cu13'
    if not binary.is_file():
        raise SystemExit('run build first')
    gpu = args.action == 'bench'
    if gpu and not args.activation:
        raise ValueError('bench requires a captured activation')
    artifact_id, payload_base, items = tensor_directory(args.model)
    rows = []
    for identity in ((args.tensor,) if args.tensor else TENSORS):
        item = items[identity]
        n, k = item['shape']
        if item['format'] != 'gguf_q4_k' or k != 5120 or n % 8 or item['bytes'] != n * (k // 256) * 144:
            raise ValueError(f'{identity} is not a complete K5120 GGUF Q4_K tensor')
        offset = payload_base + item['offset']
        suffix = '-bench-' + args.activation.stem if gpu else ''
        output_path = RESULTS / f'q4-fastwidepairmeta-{identity.replace("/", "-")}{suffix}.jsonl'
        command = [linux(binary), linux(args.model), offset, n, k, identity, args.action]
        if gpu:
            command.append(linux(args.activation))
        output = execute(command, output_path)
        parsed = [json.loads(line) for line in output.splitlines() if line.strip()]
        required = ('fp32_output_bit_exact', 'bf16_output_bit_exact', 'gpu_all_payload_bytes_exact') if gpu else ('cpu_all_payload_bytes_exact', 'cpu_fixed_width_halves_exact', 'cpu_old_pairmeta_exact', 'cpu_wide_pairmeta_exact')
        if len(parsed) != 1 or not all(parsed[0].get(key) for key in required):
            raise RuntimeError(f'CPU full tensor verification failed for {identity}')
        row = parsed[0]
        row.update({'tensor': identity, 'source': str(args.model.resolve()), 'artifact_id': artifact_id,
                    'absolute_payload_offset': offset, 'gpu_executed': gpu})
        (RESULTS / f'q4-fastwidepairmeta-{identity.replace("/", "-")}{suffix}.json').write_text(json.dumps(row, indent=2) + '\n')
        rows.append(row)
    report = {'format': 'q4-fast-fields-wide-pair-metadata-probe-v1',
              'candidate': 'same fixed-width d/dmin streams and fused Q4 arithmetic; compare original metadata, packed 12-bit scale/min pairs, and 12-byte-aligned word/funnel-shift decoder with 4-byte tail padding',
              'cpu_full_tensor_oracle': not gpu, 'gpu_executed': gpu, 'tensors': rows}
    report_name = 'q4-fastwidepairmeta-bench-' + args.activation.stem + '-summary.json' if gpu else 'q4-fastwidepairmeta-summary.json'
    (RESULTS / report_name).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('build', 'scan', 'bench'))
    parser.add_argument('--model', type=pathlib.Path, default=MODEL)
    parser.add_argument('--tensor', choices=TENSORS)
    parser.add_argument('--activation', type=pathlib.Path)
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    if args.action == 'build':
        build()
    else:
        scan(args)


if __name__ == '__main__':
    main()
