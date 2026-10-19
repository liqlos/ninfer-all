import argparse
import json
import pathlib
import struct
import subprocess


ROOT = pathlib.Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
MODEL = pathlib.Path(r'C:\AI\staging\qwen38-jonathan-ninfer\qwen38-jonathan-q4km-proposal.ninfer')
VENDOR = pathlib.Path(r'C:\AI\staging\ninfer-all-pin\third_party\ggml-quants')
IMAGE = 'ninfer-lossless-q6-oracle:sm86'
PROBE = RESULTS / 'q6-integrated-operator-probe'


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


def run_docker(command, output_path):
    result = subprocess.run(['wsl.exe', '-d', 'Ubuntu-24.04', '-u', 'root', '--', *command],
                            capture_output=True)
    output_path.write_bytes(result.stdout)
    output_path.with_suffix(output_path.suffix + '.stderr').write_bytes(result.stderr)
    if result.returncode:
        raise RuntimeError(f'command failed with exit {result.returncode}; see {output_path}.stderr')
    return result.stdout.decode()


def build():
    includes = ['/experiment/source/src', '/vendor/src', '/vendor/include',
                '/src/src', '/src/third_party/ggml-quants/src',
                '/src/third_party/ggml-quants/include', '/src/third_party']
    sources = ['/experiment/q6_integrated_operator_probe.cu',
               '/experiment/source/src/artifact/q6_fields.cpp',
               '/experiment/source/src/core/weight_view.cpp',
               '/experiment/source/src/core/tensor.cpp',
               '/experiment/source/src/core/dtype.cpp',
               '/experiment/source/src/core/q6_fields.cu']
    archives = ['/out/libs/libninfer_artifact.a', '/out/libs/libninfer_ops.a',
                '/out/libs/libninfer_core.a', '/out/libs/libninfer_ggml_quants.a']
    command = ['docker', 'run', '--rm', '-v', linux(ROOT) + ':/experiment',
               '-v', linux(VENDOR) + ':/vendor:ro', '--entrypoint',
               '/usr/local/cuda/bin/nvcc', IMAGE, '-O3', '-DNDEBUG', '-std=c++20',
               '-arch=sm_86', '-lineinfo', '-DNINFER_SM8X_COMPAT=1',
               '-Xptxas=-v', *['-I' + path for path in includes], *sources,
               '-Xlinker', '--start-group', *archives, '-Xlinker', '--end-group',
               '-lcublas', '-lcudart', '-ldl', '-lpthread', '-o',
               '/experiment/results/q6-integrated-operator-probe']
    output = run_docker(command, RESULTS / 'q6-integrated-operator-build.txt')
    build_log = (RESULTS / 'q6-integrated-operator-build.txt.stderr').read_text(errors='replace')
    registers = {}
    current_width = None
    for line in build_log.splitlines():
        for width in (1, 3, 4):
            if f'kernelILi{width}ELb0ELb1' in line:
                current_width = width
        if 'Used ' in line and ' registers' in line:
            if current_width is not None:
                registers[f'T{current_width}'] = int(line.split('Used ', 1)[1].split(' registers', 1)[0])
                current_width = None
    report = {'build_succeeded': True, 'gpu_executed': False, 'binary': str(PROBE),
              'toolchain_image': IMAGE, 'vendor_headers': str(VENDOR),
              'candidate_registers_per_thread': registers, 'build_stdout': output[-2000:]}
    (RESULTS / 'q6-integrated-operator-build.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


def run(args):
    artifact_id, payload_base, items = tensor_directory(args.model)
    item = items.get(args.tensor)
    if item is None or item['format'] != 'gguf_q6_k' or item['shape'] != [5120, 17408]:
        raise ValueError(f'{args.tensor} is not the supported [5120,17408] GGUF Q6_K tensor')
    activation = RESULTS / 'activations' / f'q14-k17408-t{args.t}-f0.bin'
    if not activation.is_file():
        raise ValueError(f'missing activation capture {activation.name}')
    slice_rows = min(args.slice_rows, item['shape'][0])
    slice_begin = ((item['shape'][0] - slice_rows) // 2 // 64) * 64 if args.slice_begin is None else args.slice_begin
    if slice_begin < 0 or slice_begin + slice_rows > item['shape'][0]:
        raise ValueError('requested slice exceeds tensor rows')
    tensor_offset = payload_base + item['offset']
    log = RESULTS / f'q6-integrated-{args.tensor.replace("/", "-")}-t{args.t}.jsonl'
    command = ['docker', 'run', '--rm', '--gpus', 'all', '-v', linux(ROOT) + ':/experiment',
               '-v', linux(args.model) + ':/model.ninfer:ro', '--entrypoint',
               '/bin/sh', IMAGE, '/experiment/results/q6-integrated-launch.sh',
               '/model.ninfer', str(tensor_offset), str(item['shape'][0]), str(item['shape'][1]),
               args.tensor, '/experiment/' + activation.relative_to(ROOT).as_posix(),
               str(slice_begin), str(slice_rows)]
    output = run_docker(command, log)
    records = [json.loads(line) for line in output.splitlines() if line.strip()]
    summary = records[-1]
    summary.update({'tensor': args.tensor, 'format': item['format'], 'artifact_id': artifact_id,
                    'absolute_payload_offset': tensor_offset, 'activation_columns': args.t,
                    'activation_snapshot': str(activation.resolve()),
                    'oracle': 'actual q6 artifact encoder + current native_weight + direct q6_fields.cuh launch versus original vector_product + core unpack_q6_fields',
                    'gpu_executed': True, 'records': records[:-1]})
    report = RESULTS / f'q6-integrated-{args.tensor.replace("/", "-")}-t{args.t}-summary.json'
    report.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('build', 'run'))
    parser.add_argument('--model', type=pathlib.Path, default=MODEL)
    parser.add_argument('--tensor', default='weight/000143')
    parser.add_argument('--t', type=int, choices=(4,), default=4)
    parser.add_argument('--slice-rows', type=int, default=4096)
    parser.add_argument('--slice-begin', type=int)
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    if args.action == 'build':
        build()
    else:
        run(args)


if __name__ == '__main__':
    main()
