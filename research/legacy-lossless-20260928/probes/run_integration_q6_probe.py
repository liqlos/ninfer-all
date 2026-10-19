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
    items = {item['id']: item for item in directory['objects']}
    return artifact_id.hex(), payload, items


def docker(command, output_path, gpu=False):
    args = ['docker', 'run', '--rm']
    if gpu:
        args.extend(['--gpus', 'all'])
    args.extend(command)
    result = subprocess.run(['wsl.exe', '-d', 'Ubuntu-24.04', '-u', 'root', '--', *args],
                            capture_output=True)
    output_path.write_bytes(result.stdout)
    output_path.with_suffix(output_path.suffix + '.stderr').write_bytes(result.stderr)
    if result.returncode:
        raise RuntimeError(f'command failed with exit {result.returncode}; see {output_path}.stderr')
    return result.stdout.decode()


def build(args):
    includes = ['/src/src', '/src/third_party/ggml-quants/src', '/src/third_party/ggml-quants/include']
    archives = ['/out/libs/libninfer_artifact.a', '/out/libs/libninfer_ops.a',
                '/out/libs/libninfer_core.a', '/out/libs/libninfer_ggml_quants.a']
    command = ['docker', 'run', '--rm', '-v', linux(ROOT) + ':/experiment', '--entrypoint',
               '/usr/local/cuda/bin/nvcc', IMAGE, '-O3', '-DNDEBUG', '-std=c++20', '-arch=sm_86',
               '-lineinfo', '-DNINFER_SM8X_COMPAT=1', '-Xptxas=-v',
               *['-I' + path for path in includes], '/experiment/integration_q6_probe.cu',
               '-Xlinker', '--start-group', *archives, '-Xlinker', '--end-group',
               '-lcublas', '-lcudart', '-ldl', '-lpthread', '-o', '/experiment/results/integration-q6-probe']
    result = subprocess.run(['wsl.exe', '-d', 'Ubuntu-24.04', '-u', 'root', '--', *command], capture_output=True)
    (RESULTS / 'integration-q6-build.txt').write_bytes(result.stdout)
    (RESULTS / 'integration-q6-build.txt.stderr').write_bytes(result.stderr)
    if result.returncode:
        raise RuntimeError('CUDA 13.1 link failed; see results/integration-q6-build.txt.stderr')
    print(json.dumps({'binary': str(RESULTS / 'integration-q6-probe'), 'gpu_executed': False}))


def run(args):
    binary = RESULTS / 'integration-q6-probe'
    if not binary.is_file():
        raise SystemExit('run `python run_integration_q6_probe.py build` first')
    artifact_id, payload_base, items = tensor_directory(args.model)
    identities = (args.tensor,) if args.tensor else ('weight/000131', 'weight/000143')
    results = []
    for identity in identities:
        item = items.get(identity)
        if item is None:
            raise ValueError(f'{identity} is not in the model directory')
        if item['format'] != 'gguf_q6_k' or len(item['shape']) != 2:
            raise ValueError(f'{identity} is not a rank-2 GGUF Q6_K tensor')
        n, k = item['shape']
        row_bytes = k // 256 * 210
        if k % 1024 or n * row_bytes != item['bytes']:
            raise ValueError(f'{identity} has unsupported Q6_K geometry')
        activation = RESULTS / 'activations' / f'q14-k{k}-t{args.t}-f0.bin'
        if not activation.is_file():
            raise ValueError(f'missing captured Q6 activation {activation.name}')
        slice_rows = min(args.slice_rows, n)
        slice_begin = (n // 2 // 64) * 64 if args.slice_begin is None else args.slice_begin
        if slice_begin < 0 or slice_begin + slice_rows > n:
            raise ValueError(f'slice [{slice_begin}, {slice_begin + slice_rows}) exceeds {identity} rows')
        offset = payload_base + item['offset']
        log = RESULTS / f'integration-q6-{identity.replace("/", "-")}-t{args.t}.jsonl'
        command = ['-v', linux(ROOT) + ':/experiment', '-v', linux(args.model) + ':/model.ninfer:ro',
                   '--entrypoint', '/experiment/results/integration-q6-probe', IMAGE,
                   '/model.ninfer', str(offset), str(n), str(k), identity,
                   '/experiment/' + activation.relative_to(ROOT).as_posix(), str(slice_begin), str(slice_rows)]
        output = docker(command, log, gpu=True)
        records = [json.loads(line) for line in output.splitlines() if line.strip()]
        cases, summary = records[:-1], records[-1]
        if len(cases) != 8 or not all(case.get('bit_exact') for case in cases):
            raise RuntimeError(f'{identity} integrated vector oracle failed; see {log}')
        summary.update({'tensor': identity, 'format': item['format'], 'N': n, 'K': k,
                        'row_bytes': row_bytes, 'columns': args.t, 'source': str(args.model.resolve()),
                        'artifact_id': artifact_id, 'absolute_payload_offset': offset,
                        'activation_snapshot': str(activation.resolve()),
                        'oracle': 'actual artifact describe_q6_fields/encode_q6_fields + core native_weight + ops vector_q6_fields against ops vector_product',
                        'gpu_executed': True, 'cases': cases})
        log.write_text(json.dumps(summary) + '\n')
        results.append(summary)
    report = {'gpu_executed': True, 'toolchain_image': IMAGE,
              'activation_T': args.t, 'tensors': results}
    report_path = RESULTS / f'integration-q6-t{args.t}-summary.json'
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('build', 'run'))
    parser.add_argument('--model', type=pathlib.Path, default=MODEL)
    parser.add_argument('--tensor')
    parser.add_argument('--t', type=int, choices=(1, 3, 4), default=1)
    parser.add_argument('--slice-rows', type=int, default=256)
    parser.add_argument('--slice-begin', type=int)
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    if args.action == 'build':
        build(args)
    else:
        run(args)


if __name__ == '__main__':
    main()
