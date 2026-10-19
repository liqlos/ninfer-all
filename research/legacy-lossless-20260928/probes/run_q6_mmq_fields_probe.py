import argparse
import json
import pathlib
import struct
import subprocess


ROOT = pathlib.Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
MODEL = pathlib.Path(r'C:\AI\staging\qwen38-jonathan-ninfer\qwen38-jonathan-q4km-proposal.ninfer')
IMAGE = 'ninfer-lossless-q6-oracle:sm86'
TENSORS = ('weight/000131', 'weight/000143')


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


def build(variant):
    includes = ['/experiment/source/src', '/experiment/source/include',
                '/experiment/source/third_party',
                '/experiment/source/third_party/ggml-quants/src',
                '/experiment/source/third_party/ggml-quants/include']
    if variant in ('b4', 'static-b4'):
        includes.insert(0, '/experiment/q6_mmq_wide/include')
        includes.append('/experiment/source/third_party/ggml-quants/src/ggml-cuda')
    sources = ['/experiment/q6_mmq_fields_probe.cu',
               '/experiment/source/src/artifact/q6_fields.cpp',
               '/experiment/source/src/core/weight_view.cpp',
               '/experiment/source/src/core/tensor.cpp',
               '/experiment/source/src/core/dtype.cpp',
               '/experiment/source/src/core/q6_fields.cu',
               '/experiment/source/src/ops/linear/gguf/ggml_mmq_q6_k.cu']
    archives = ['/out/libs/libninfer_artifact.a', '/out/libs/libninfer_ops.a',
                '/out/libs/libninfer_core.a', '/out/libs/libninfer_ggml_quants.a']
    command = ['docker', 'run', '--rm', '-v', linux(ROOT) + ':/experiment', '--entrypoint',
               '/usr/local/cuda/bin/nvcc', IMAGE, '-O3', '-DNDEBUG', '-std=c++20', '-arch=sm_86',
               '-lineinfo', '-DNINFER_SM8X_COMPAT=1', *(['-DNINFER_Q6_STATIC_B4=1'] if variant == 'static-b4' else []), '-Xptxas=-v',
               *['-I' + path for path in includes],
               *sources,
               '-Xlinker', '--start-group', *archives, '-Xlinker', '--end-group',
               '-lcublas', '-lcudart', '-ldl', '-lpthread',
               '-o', f'/experiment/results/q6-mmq-fields-probe-cu13-{variant}']
    result = subprocess.run(['wsl.exe', '-d', 'Ubuntu-24.04', '-u', 'root', '--', *command], capture_output=True)
    (RESULTS / f'q6-mmq-fields-{variant}-build.txt').write_bytes(result.stdout)
    (RESULTS / f'q6-mmq-fields-{variant}-build.txt.stderr').write_bytes(result.stderr)
    if result.returncode:
        raise RuntimeError(f'CUDA 13.1 link failed; see results/q6-mmq-fields-{variant}-build.txt.stderr')
    print(json.dumps({'binary': str(RESULTS / f'q6-mmq-fields-probe-cu13-{variant}'),
                      'load_variant': variant, 'gpu_executed': False}))


def run(args):
    binary = RESULTS / f'q6-mmq-fields-probe-cu13-{args.variant}'
    if not binary.is_file():
        raise SystemExit('run `python run_q6_mmq_fields_probe.py build` first')
    artifact_id, payload_base, items = tensor_directory(args.model)
    identities = (args.tensor,) if args.tensor else TENSORS
    summaries = []
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
        if args.slice_rows and (args.slice_begin < 0 or args.slice_begin + args.slice_rows > n):
            raise ValueError(f'slice exceeds {identity} rows')
        offset = payload_base + item['offset']
        log = RESULTS / f'q6-mmq-fields-{identity.replace("/", "-")}-t{args.t.replace(",", "-")}-{args.variant}.jsonl'
        command = ['docker', 'run', '--rm', '--gpus', 'all', '-v', linux(ROOT) + ':/experiment',
                   '-v', linux(args.model) + ':/model.ninfer:ro', '--entrypoint', '/bin/sh', IMAGE,
                   '/experiment/results/run-q6-mmq-fields-probe.sh',
                   f'/experiment/results/{binary.name}', '/model.ninfer',
                   str(offset), str(n), str(k), identity, args.t,
                   f'{args.slice_begin}:{args.slice_rows}']
        result = subprocess.run(['wsl.exe', '-d', 'Ubuntu-24.04', '-u', 'root', '--', *command], capture_output=True)
        log.write_bytes(result.stdout)
        log.with_suffix(log.suffix + '.stderr').write_bytes(result.stderr)
        if result.returncode:
            raise RuntimeError(f'{identity} Q6 MMQ fields oracle failed; see {log}.stderr')
        records = [json.loads(line) for line in result.stdout.decode().splitlines() if line.strip()]
        if not records or not all(record.get('fp32_exact', True) and record.get('bf16_exact', True)
                                  for record in records[:-1]):
            raise RuntimeError(f'{identity} Q6 MMQ fields comparison failed; see {log}')
        summary = {'tensor': identity, 'N': n, 'K': k, 'format': item['format'],
                   'raw_row_bytes': row_bytes, 'source': str(args.model.resolve()),
                   'artifact_id': artifact_id, 'absolute_payload_offset': offset,
                   'columns': args.t, 'slice_begin': args.slice_begin,
                   'slice_rows': args.slice_rows, 'toolchain_image': IMAGE, 'load_variant': args.variant,
                   'gpu_executed': True, 'cases': records}
        summaries.append(summary)
    report = {'gpu_executed': True, 'toolchain_image': IMAGE, 'matrices': summaries}
    report_path = RESULTS / f'q6-mmq-fields-t{args.t.replace(",", "-")}-{args.variant}-summary.json'
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('build', 'build-b4', 'build-static-b4', 'run'))
    parser.add_argument('--variant', choices=('current-abi', 'b4', 'static-b4'), default='current-abi')
    parser.add_argument('--model', type=pathlib.Path, default=MODEL)
    parser.add_argument('--tensor')
    parser.add_argument('--t', default='9,12,16,1024')
    parser.add_argument('--slice-begin', type=int, default=256)
    parser.add_argument('--slice-rows', type=int, default=512)
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    if args.action == 'build':
        build('current-abi')
    elif args.action == 'build-b4':
        build('b4')
    elif args.action == 'build-static-b4':
        build('static-b4')
    else:
        run(args)


if __name__ == '__main__':
    main()
