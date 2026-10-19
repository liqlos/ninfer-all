import argparse
import hashlib
import json
import pathlib
import re
import struct
import subprocess
import threading
import time
import urllib.request


ROOT = pathlib.Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
PRIVATE = ROOT / 'q4_mmq_private'
MODEL = pathlib.Path(r'C:\AI\staging\qwen38-jonathan-ninfer\qwen38-jonathan-q4km-proposal.ninfer')
IMAGE = 'ninfer-lossless-q6-oracle:sm86'
IMAGE_ID = 'sha256:11d92d17b1c5065c02f15c16a152d52cab638f7f3461550648f242de1113a52c'
TENSORS = ('weight/000060', 'weight/000224')


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


def require_gpu_unowned():
    with urllib.request.urlopen('http://127.0.0.1:8096/v1/status', timeout=15) as response:
        status = json.load(response)
    if (status.get('leases') or status.get('agents') or status.get('instances') or
            status.get('inference', {}).get('active_requests') or
            status.get('external_growth_pressure', {}).get('active')):
        raise RuntimeError('GPU ownership is unavailable')


def image_identity():
    result = subprocess.run(['wsl.exe', '-d', 'Ubuntu-24.04', '-u', 'root', '--', 'bash', '-lc',
                             f"docker image inspect {IMAGE} --format '{{{{.Id}}}}'"],
                            capture_output=True, text=True)
    if result.returncode or result.stdout.strip() != IMAGE_ID:
        raise RuntimeError(f'expected pinned toolchain image {IMAGE_ID}, got {result.stdout.strip()}')
    return result.stdout.strip()


def build():
    includes = ['/experiment/q4_mmq_private', '/experiment/q4_mmq_private/include',
                '/experiment/source/src', '/experiment/source/include',
                '/experiment/source/src/ops/linear/gguf',
                '/experiment/source/third_party',
                '/experiment/source/third_party/ggml-quants/src',
                '/experiment/source/third_party/ggml-quants/src/ggml-cuda',
                '/experiment/source/third_party/ggml-quants/include']
    sources = ['/experiment/q4_mmq_private/q4_mmq_fields_probe.cu',
               '/experiment/source/src/artifact/q4_fields.cpp',
               '/experiment/source/src/core/weight_view.cpp',
               '/experiment/source/src/core/tensor.cpp',
               '/experiment/source/src/core/dtype.cpp',
               '/experiment/source/src/core/q4_fields.cu',
               '/experiment/q4_mmq_private/ggml_mmq_q4_fields_private.cu']
    archives = ['/out/libs/libninfer_artifact.a', '/out/libs/libninfer_ops.a',
                '/out/libs/libninfer_core.a', '/out/libs/libninfer_ggml_quants.a']
    command = ['docker', 'run', '--rm', '-v', linux(ROOT) + ':/experiment', '--entrypoint',
               '/usr/local/cuda/bin/nvcc', IMAGE, '-O3', '-DNDEBUG', '-std=c++20', '-arch=sm_86',
               '-lineinfo', '-DNINFER_SM8X_COMPAT=1', '-Xptxas=-v',
               *['-I' + path for path in includes], *sources,
               '-Xlinker', '--start-group', *archives, '-Xlinker', '--end-group',
               '-lcublas', '-lcudart', '-ldl', '-lpthread',
               '-o', '/experiment/results/q4-mmq-private-probe-cu13']
    result = subprocess.run(['wsl.exe', '-d', 'Ubuntu-24.04', '-u', 'root', '--', *command], capture_output=True)
    (RESULTS / 'q4-mmq-private-build.txt').write_bytes(result.stdout)
    (RESULTS / 'q4-mmq-private-build.txt.stderr').write_bytes(result.stderr)
    if result.returncode:
        raise RuntimeError('CUDA 13.1 private Q4 MMQ build failed; inspect q4-mmq-private-build.txt.stderr')
    print(json.dumps({'binary': str(RESULTS / 'q4-mmq-private-probe-cu13'), 'gpu_executed': False}))


def run(args):
    binary = RESULTS / 'q4-mmq-private-probe-cu13'
    if not binary.is_file():
        raise SystemExit('run `python run_q4_mmq_private.py build` first')
    artifact_id, payload_base, items = tensor_directory(args.model)
    if not args.tag or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,63}', args.tag):
        raise SystemExit('--tag is required and must contain 1-64 safe filename characters')
    expected_columns = [int(value) for value in args.t.split(',')]
    if not expected_columns or any(value <= 0 for value in expected_columns) or len(set(expected_columns)) != len(expected_columns):
        raise SystemExit('--t must be a comma-separated list of unique positive integers')
    pinned_image_id = image_identity()
    require_gpu_unowned()
    identities = tuple(args.tensor or TENSORS)
    report_path = RESULTS / f'q4-mmq-private-{args.tag}-t{args.t.replace(",", "-")}-summary.json'
    prospective_logs = [RESULTS / f'q4-mmq-private-{args.tag}-{identity.replace("/", "-")}-t{args.t.replace(",", "-")}.jsonl'
                        for identity in identities]
    collisions = [path for path in [report_path, *prospective_logs] if path.exists()]
    collisions += [path.with_suffix('.telemetry.json') for path in prospective_logs if path.with_suffix('.telemetry.json').exists()]
    collisions += [path.with_suffix(path.suffix + '.stderr') for path in prospective_logs if path.with_suffix(path.suffix + '.stderr').exists()]
    if collisions:
        raise FileExistsError(f'preserving prior tagged receipts; choose a unique --tag: {collisions[0]}')
    summaries = []
    for identity in identities:
        item = items.get(identity)
        if item is None:
            raise ValueError(f'{identity} is not in the model directory')
        if item['format'] != 'gguf_q4_k' or len(item['shape']) != 2:
            raise ValueError(f'{identity} is not a rank-2 GGUF Q4_K tensor')
        n, k = item['shape']
        if (n, k) not in ((16384, 5120), (5120, 17408)):
            raise ValueError(f'{identity} has unsupported Q4 field geometry {item["shape"]}')
        if n * (k // 256 * 144) != item['bytes']:
            raise ValueError(f'{identity} has inconsistent Q4_K geometry')
        offset = payload_base + item['offset']
        log = RESULTS / f'q4-mmq-private-{args.tag}-{identity.replace("/", "-")}-t{args.t.replace(",", "-")}.jsonl'
        telemetry_path = log.with_suffix('.telemetry.json')
        stderr_path = log.with_suffix(log.suffix + '.stderr')
        if log.exists() or telemetry_path.exists() or stderr_path.exists():
            raise FileExistsError(f'{log} or its companion receipt already exists; choose a unique --tag')
        activation_mount = []
        activation_path = '-'
        if args.activation:
            activation_mount = ['-v', linux(args.activation) + ':/model-activation.bin:ro']
            activation_path = '/model-activation.bin'
        command = ['docker', 'run', '--rm', '--gpus', 'all', '-v', linux(ROOT) + ':/experiment',
                   '-v', linux(args.model) + ':/model.ninfer:ro', *activation_mount,
                   '--entrypoint', '/bin/sh', IMAGE,
                   '/experiment/results/run-q4-mmq-private.sh',
                   f'/experiment/results/{binary.name}', '/model.ninfer', str(offset), str(n), str(k),
                   identity, args.t, args.activation_mode, activation_path]
        stop = threading.Event()
        samples = []

        def monitor():
            while not stop.is_set():
                epoch_ns = time.time_ns()
                observed = subprocess.run(['wsl.exe', '-d', 'Ubuntu-24.04', '-u', 'root', '--', 'nvidia-smi',
                    '--query-gpu=timestamp,clocks.sm,clocks.mem,temperature.gpu,power.draw,memory.used,utilization.gpu',
                    '--format=csv,noheader,nounits'], capture_output=True, text=True)
                samples.append({'epoch_ns': epoch_ns, 'gpu_csv': observed.stdout.strip(), 'exit_code': observed.returncode})
                stop.wait(0.25)

        metadata = {'command': command, 'oracle_tag': args.tag, 'toolchain_image': IMAGE,
                    'toolchain_image_id': pinned_image_id, 'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
                    'runner_sha256': hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
                    'private_loader_sha256': hashlib.sha256((PRIVATE / 'include' / 'ggml-cuda' / 'mmq-load-tiles.cuh').read_bytes()).hexdigest(),
                    'start_epoch_ns': time.time_ns(), 'power_or_clock_controls_changed': False}
        observer = threading.Thread(target=monitor)
        observer.start()
        try:
            result = subprocess.run(['wsl.exe', '-d', 'Ubuntu-24.04', '-u', 'root', '--', *command], capture_output=True)
        finally:
            stop.set()
            observer.join()
        log.write_bytes(result.stdout)
        stderr_path.write_bytes(result.stderr)
        metadata.update(end_epoch_ns=time.time_ns(), exit_code=result.returncode, samples=samples)
        telemetry_path.write_text(json.dumps(metadata, indent=2) + '\n')
        if result.returncode:
            raise RuntimeError(f'{identity} private Q4 MMQ oracle failed; inspect {log}.stderr')
        records = [json.loads(line) for line in result.stdout.decode().splitlines() if line.strip()]
        required_case_keys = {'tensor', 'N', 'K', 'T', 'view', 'row_origin', 'rows', 'activation',
                              'J', 'I', 'threads', 'shared_bytes', 'stream_k', 'blocks', 'fixup',
                              'fp32_exact', 'bf16_exact', 'fp32_canary', 'bf16_canary',
                              'raw_graph_ms', 'unpack_raw_graph_ms', 'direct_graph_ms'}
        if len(records) != len(args.t.split(',')) * 2 + 1:
            raise RuntimeError(f'{identity} private Q4 MMQ emitted {len(records)} records, expected {len(args.t.split(",")) * 2 + 1}; inspect {log}')
        cases = records[:-1]
        footer = records[-1]
        valid_cases = all(required_case_keys <= record.keys() and
                          record['tensor'] == identity and record['N'] == n and record['K'] == k and
                          record['T'] in expected_columns and
                          record['activation'] == args.activation_mode and
                          record['view'] in ('root', 'odd-origin-slice') and
                          record['row_origin'] == (0 if record['view'] == 'root' else 1) and
                          (record['rows'] == n if record['view'] == 'root' else record['rows'] > 0) and
                          record['fp32_exact'] is True and record['bf16_exact'] is True and
                          record['fp32_canary'] is True and record['bf16_canary'] is True and
                          all(isinstance(record[key], list) and len(record[key]) == 5
                              for key in ('raw_graph_ms', 'unpack_raw_graph_ms', 'direct_graph_ms'))
                          for record in cases)
        complete_views = valid_cases and all({record['view'] for record in cases if record['T'] == columns} ==
                                             {'root', 'odd-origin-slice'} for columns in expected_columns)
        footer_valid = footer.get('tensor') == identity and footer.get('N') == n and footer.get('K') == k and footer.get('full_byte_reconstruction') is True
        if not valid_cases or not complete_views or not footer_valid:
            raise RuntimeError(f'{identity} private Q4 MMQ comparison failed; inspect {log}')
        summary = {'tensor': identity, 'N': n, 'K': k, 'format': item['format'],
                   'raw_row_bytes': k // 256 * 144, 'source': str(args.model.resolve()),
                   'artifact_id': artifact_id, 'absolute_payload_offset': offset,
                   'columns': args.t, 'activation_mode': args.activation_mode,
                   'activation_file': str(args.activation.resolve()) if args.activation else None,
                   'toolchain_image': IMAGE, 'toolchain_image_id': pinned_image_id,
                   'oracle_tag': args.tag, 'gpu_executed': True, 'cases': records}
        summaries.append(summary)
    report = {'gpu_executed': True, 'toolchain_image': IMAGE, 'toolchain_image_id': pinned_image_id,
              'oracle_tag': args.tag, 'matrices': summaries}
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('build', 'run'))
    parser.add_argument('--model', type=pathlib.Path, default=MODEL)
    parser.add_argument('--tensor', action='append')
    parser.add_argument('--t', default='9,12,1024')
    parser.add_argument('--tag')
    parser.add_argument('--activation-mode', choices=('synthetic', 'real'), default='synthetic')
    parser.add_argument('--activation', type=pathlib.Path)
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    if args.activation_mode == 'real' and args.activation is None:
        raise SystemExit('--activation is required with --activation-mode real')
    if args.action == 'build':
        build()
    else:
        run(args)


if __name__ == '__main__':
    main()
