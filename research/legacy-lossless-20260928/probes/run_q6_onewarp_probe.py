import argparse
import json
import pathlib
import struct
import subprocess


ROOT = pathlib.Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
MODEL = pathlib.Path(r'C:\AI\staging\qwen38-jonathan-ninfer\qwen38-jonathan-q4km-proposal.ninfer')
IMAGE = 'ninfer-lossless-q6-oracle:sm86'


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


def build():
    includes = ['/src/src', '/src/third_party/ggml-quants/src', '/src/third_party/ggml-quants/include']
    archives = ['/out/libs/libninfer_artifact.a', '/out/libs/libninfer_ops.a',
                '/out/libs/libninfer_core.a', '/out/libs/libninfer_ggml_quants.a']
    command = ['docker', 'run', '--rm', '-v', linux(ROOT) + ':/experiment', '--entrypoint',
               '/usr/local/cuda/bin/nvcc', IMAGE, '-O3', '-DNDEBUG', '-std=c++20', '-arch=sm_86',
               '-lineinfo', '-DNINFER_SM8X_COMPAT=1', '-Xptxas=-v', *['-I' + p for p in includes],
               '/experiment/q6_onewarp_probe.cu', '-Xlinker', '--start-group', *archives,
               '-Xlinker', '--end-group', '-lcublas', '-lcudart', '-ldl', '-lpthread',
               '-o', '/experiment/results/q6-onewarp-probe']
    result = subprocess.run(['wsl.exe', '-d', 'Ubuntu-24.04', '-u', 'root', '--', *command], capture_output=True)
    (RESULTS / 'q6-onewarp-build.txt').write_bytes(result.stdout)
    (RESULTS / 'q6-onewarp-build.txt.stderr').write_bytes(result.stderr)
    if result.returncode:
        raise RuntimeError('CUDA 13.1 build failed; see results/q6-onewarp-build.txt.stderr')
    print(json.dumps({'binary': str(RESULTS / 'q6-onewarp-probe'), 'gpu_executed': False}))


def run(args):
    binary = RESULTS / 'q6-onewarp-probe'
    if not binary.is_file():
        raise SystemExit('run `python run_q6_onewarp_probe.py build` first')
    artifact_id, payload_base, items = tensor_directory(args.model)
    item = items.get(args.tensor)
    if item is None or item['format'] != 'gguf_q6_k' or len(item['shape']) != 2:
        raise ValueError(f'{args.tensor} is not a rank-2 Q6_K tensor')
    n, k = item['shape']
    if k % 1024 or item['bytes'] != n * (k // 256) * 210:
        raise ValueError('tensor geometry is not supported')
    activation = RESULTS / 'activations' / f'q14-k{k}-t4-f0.bin'
    if not activation.is_file():
        raise ValueError(f'missing activation snapshot {activation.name}')
    offset = payload_base + item['offset']
    args_list = ['-v', linux(ROOT) + ':/experiment', '-v', linux(args.model) + ':/model.ninfer:ro',
                 '--entrypoint', '/experiment/results/q6-onewarp-probe', IMAGE,
                 '/model.ninfer', str(offset), str(n), str(k), args.tensor,
                 '/experiment/' + activation.relative_to(ROOT).as_posix()]
    result = subprocess.run(['wsl.exe', '-d', 'Ubuntu-24.04', '-u', 'root', '--', 'docker', 'run', '--rm',
                             '--gpus', 'all', *args_list], capture_output=True)
    path = RESULTS / f'q6-onewarp-{args.tensor.replace("/", "-")}-t4.jsonl'
    path.write_bytes(result.stdout)
    path.with_suffix(path.suffix + '.stderr').write_bytes(result.stderr)
    if result.returncode:
        raise RuntimeError(f'probe failed with exit {result.returncode}; see {path}.stderr')
    records = [json.loads(line) for line in result.stdout.decode().splitlines() if line.strip()]
    cases, summary = records[:-1], records[-1]
    if len(cases) != 10 or not all(case.get('fp32_bit_exact') and case.get('bf16_bit_exact') for case in cases):
        raise RuntimeError('one-warp candidate mismatch or missing case; see probe receipt')
    summary.update({'tensor': args.tensor, 'N': n, 'K': k, 'T': 4,
                    'row_bytes': k // 256 * 210, 'source': str(args.model.resolve()),
                    'artifact_id': artifact_id, 'absolute_payload_offset': offset,
                    'activation_snapshot': str(activation.resolve()), 'gpu_executed': True,
                    'cases': cases,
                    'candidate_hypothesis': 'one output row per warp reduces accumulator live state from 2*T to T; late activation load keeps one eight-word slice live across output-column work'})
    path.write_text(json.dumps(summary) + '\n')
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('build', 'run'))
    parser.add_argument('--model', type=pathlib.Path, default=MODEL)
    parser.add_argument('--tensor', default='weight/000143')
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    if args.action == 'build':
        build()
    else:
        run(args)


if __name__ == '__main__':
    main()
