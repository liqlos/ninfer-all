import argparse
import json
import pathlib
import struct
import subprocess


ROOT = pathlib.Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
MODEL = pathlib.Path(r'C:\AI\staging\qwen38-jonathan-ninfer\qwen38-jonathan-q4km-proposal.ninfer')
TENSOR = 'weight/000131'
COLUMNS = (9, 12, 16, 32, 64, 128, 1024)


def linux(path):
    path = pathlib.Path(path).resolve()
    return '/mnt/' + path.drive[0].lower() + '/' + '/'.join(path.parts[1:])


def execute(command, output_path, gpu=False):
    docker = ['docker', 'run', '--rm']
    if gpu:
        docker.append('--gpus')
        docker.append('all')
        model_parent = str(pathlib.PurePosixPath(str(command[1])).parent)
        docker.extend(['-v', model_parent + ':' + model_parent + ':ro'])
    docker.extend(['-v', linux(ROOT) + ':/experiment', '--entrypoint',
                    '/experiment/results/q6-mmq-probe-cu13', 'ninfer-lossless-toolchain:cu13'])
    result = subprocess.run(['wsl.exe', '-d', 'Ubuntu-24.04', '-u', 'root', '--', *docker, *map(str, command)],
                            capture_output=True)
    output_path.write_bytes(result.stdout)
    output_path.with_suffix(output_path.suffix + '.stderr').write_bytes(result.stderr)
    if result.returncode:
        raise RuntimeError(f'probe failed with exit {result.returncode}; see {output_path}.stderr')
    return result.stdout.decode()


def build():
    includes = ['source/src', 'source/third_party/ggml-quants/src',
                'source/third_party/ggml-quants/include']
    command = ['docker', 'run', '--rm', '-v', linux(ROOT) + ':/experiment',
               '--entrypoint', '/usr/local/cuda/bin/nvcc', 'ninfer-lossless-toolchain:cu13',
               '-O3', '-DNDEBUG', '-std=c++20', '-arch=sm_86', '-lineinfo', '-DNINFER_SM8X_COMPAT=1',
               '-Xptxas=-v', *['-I/experiment/' + include for include in includes],
               '/experiment/q6_mmq_probe.cu', '-o', '/experiment/results/q6-mmq-probe-cu13']
    result = subprocess.run(['wsl.exe', '-d', 'Ubuntu-24.04', '-u', 'root', '--', *command], capture_output=True)
    (RESULTS / 'q6mmq-build-cu13.txt').write_bytes(result.stdout)
    (RESULTS / 'q6mmq-build-cu13.txt.stderr').write_bytes(result.stderr)
    if result.returncode:
        raise RuntimeError('CUDA 13.1 build failed; see results/q6mmq-build-cu13.txt.stderr')
    print(json.dumps({'binary': str(RESULTS / 'q6-mmq-probe-cu13'), 'gpu_executed': False}))


def artifact(model):
    with model.open('rb') as stream:
        magic, length, artifact_id = struct.unpack('<8sQ16s', stream.read(32))
        if magic != b'NINFER\0\3':
            raise ValueError('NInfer v3 required')
        directory = json.loads(stream.read(length))
    payload = (32 + length + 4095) // 4096 * 4096
    items = {item['id']: item for item in directory['objects']}
    return artifact_id.hex(), payload, items


def run_plan(args):
    binary = RESULTS / 'q6-mmq-probe-cu13'
    if not binary.is_file():
        raise SystemExit('run `python run_q6_mmq_probe.py build` first')
    output_path = RESULTS / 'q6mmq-plan-head.jsonl'
    records = []
    for columns in COLUMNS:
        raw = execute(['plan', 248320, 5120, columns], output_path.with_name(f'q6mmq-plan-t{columns}.jsonl'))
        records.append(json.loads(raw.strip().splitlines()[-1]))
    report = {'target': TENSOR, 'N': 248320, 'K': 5120, 'columns': list(COLUMNS),
              'scratch_cap_bytes': 96 * 1024 * 1024, 'device_facts_source': 'fixed CPU plan facts for RTX 3090 sm_86, 82 SMs, 101376 optin shared bytes',
              'partition_function': 'source/src/ops/linear/gguf/ggml_mmq_q6_k.cu:q6_matrix_partition',
              'launch_selection': 'source/src/ops/linear/gguf/ggml_bridge_mmq.cuh:select_matrix_launch',
              'gpu_executed': False, 'plans': records}
    output_path.write_text('\n'.join(json.dumps(record) for record in records) + '\n')
    (RESULTS / 'q6mmq-plan-summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


def run_bench(args):
    binary = RESULTS / 'q6-mmq-probe-cu13'
    if not binary.is_file():
        raise SystemExit('run `python run_q6_mmq_probe.py build` first')
    artifact_id, payload_base, items = artifact(args.model)
    item = items[TENSOR]
    n, k = item['shape']
    if item['format'] != 'gguf_q6_k' or k % 1024 or n % 128 or item['bytes'] != n * (k // 256) * 210:
        raise ValueError(f'{TENSOR} is not a full row-tile Q6_K tensor')
    offset = payload_base + item['offset']
    command = ['bench', linux(args.model), offset, n, k, args.columns]
    output_path = RESULTS / f'q6mmq-bench-t{args.columns}.jsonl'
    output = execute(command, output_path, gpu=True)
    row = json.loads(output.strip().splitlines()[-1])
    if not row.get('whole_fp32_bit_exact') or not row.get('whole_bf16_bit_exact') or not row.get('whole_and_chunk_graph_replay_exact'):
        raise RuntimeError('whole/chunk MMQ bit-exactness gate failed')
    row.update({'tensor': TENSOR, 'source': str(args.model.resolve()), 'artifact_id': artifact_id,
                'absolute_payload_offset': offset, 'gpu_executed': True})
    output_path.write_text(json.dumps(row) + '\n')
    print(json.dumps(row, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('build', 'plan', 'bench'))
    parser.add_argument('--model', type=pathlib.Path, default=MODEL)
    parser.add_argument('--columns', type=int, choices=COLUMNS, default=9)
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    if args.action == 'build':
        build()
    elif args.action == 'plan':
        run_plan(args)
    else:
        run_bench(args)


if __name__ == '__main__':
    main()
