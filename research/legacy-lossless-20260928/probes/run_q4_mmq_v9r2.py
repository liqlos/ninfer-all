import argparse
import hashlib
import json
import pathlib
import struct
import subprocess
import threading
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
SOURCE = ROOT / 'source'
MODEL = pathlib.Path(r'C:\AI\staging\qwen38-jonathan-ninfer\qwen38-jonathan-q4km-proposal.ninfer')
BINARY = RESULTS / 'q4-mmq-integrated-v9r2-probe-cu13'
IMAGE_DEFAULT = 'ninfer-lossless-q4-prefill-v7-oracle:sm86'


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


def image_identity(image, expected=None):
    result = subprocess.run(['wsl.exe', '-d', 'Ubuntu-24.04', '-u', 'root', '--', 'bash', '-lc',
                             f"docker image inspect {image} --format '{{{{.Id}}}}'"],
                            capture_output=True, text=True)
    identity = result.stdout.strip()
    if result.returncode or not identity.startswith('sha256:'):
        raise RuntimeError(f'cannot identify oracle image {image}: {identity}')
    if expected and identity != expected:
        raise RuntimeError(f'expected image {expected}, got {identity}')
    return identity


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes():
    files = [pathlib.Path(__file__), ROOT / 'integration_q4_mmq_v9r2_probe.cu', ROOT / 'q4_fused_v9_artifact.cpp', ROOT / 'q4_fused_v9_mmq.cu']
    for base in (SOURCE / 'src', SOURCE / 'include', SOURCE / 'third_party/ggml-quants'):
        files.extend(p for p in base.rglob('*') if p.is_file() and p.suffix in ('.h', '.hpp', '.cuh', '.cu', '.cpp', '.c'))
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(set(files))}


def build(args):
    if BINARY.exists():
        raise FileExistsError(f'preserving existing binary: {BINARY}')
    image_id = image_identity(args.image, args.image_id)
    build_log = RESULTS / 'q4-mmq-integrated-v9r2-build.txt'
    build_stderr = RESULTS / 'q4-mmq-integrated-v9r2-build.txt.stderr'
    receipt_path = RESULTS / 'q4-mmq-integrated-v9r2-build.json'
    collisions = [path for path in (BINARY, build_log, build_stderr, receipt_path) if path.exists()]
    if collisions:
        raise FileExistsError(f'preserving existing build artifact: {collisions[0]}')
    includes = ['/experiment/source/src', '/experiment/source/include',
                '/experiment/source/src/ops/linear/gguf', '/experiment/source/third_party',
                '/experiment/source/third_party/ggml-quants/src',
                '/experiment/source/third_party/ggml-quants/src/ggml-cuda',
                '/experiment/source/third_party/ggml-quants/include']
    sources = ['/experiment/integration_q4_mmq_v9r2_probe.cu',
               '/experiment/q4_fused_v9_artifact.cpp',
               '/experiment/q4_fused_v9_mmq.cu',
               '/experiment/source/src/core/weight_view.cpp',
               '/experiment/source/src/core/tensor.cpp',
               '/experiment/source/src/core/dtype.cpp',
               '/experiment/source/src/core/q4_fields.cu']
    archives = ['/out/libs/libninfer_artifact.a', '/out/libs/libninfer_ops.a',
                '/out/libs/libninfer_core.a', '/out/libs/libninfer_ggml_quants.a']
    command = ['docker', 'run', '--rm', '-v', linux(ROOT) + ':/experiment', '--entrypoint',
               '/usr/local/cuda/bin/nvcc', args.image, '-O3', '-DNDEBUG', '-std=c++20', '-arch=sm_86',
               '-lineinfo', '-DNINFER_SM8X_COMPAT=1', '-Xptxas=-v',
               *['-I' + path for path in includes], *sources,
               '-Xlinker', '--start-group', *archives, '-Xlinker', '--end-group',
               '-lcublas', '-lcudart', '-ldl', '-lpthread',
               '-o', '/experiment/results/' + BINARY.name]
    before = source_hashes()
    result = subprocess.run(['wsl.exe', '-d', 'Ubuntu-24.04', '-u', 'root', '--', *command], capture_output=True)
    build_log.write_bytes(result.stdout)
    build_stderr.write_bytes(result.stderr)
    if result.returncode:
        raise RuntimeError('public Q4 fused MLP MMQ oracle build failed; inspect q4-mmq-integrated-v9r2-build.txt.stderr')
    if source_hashes() != before:
        raise RuntimeError('Source changed during oracle build')
    receipt = {'source_sha256': before, 'binary': str(BINARY), 'binary_sha256': sha256(BINARY),
               'toolchain_image': args.image, 'toolchain_image_id': image_id,
               'runner_sha256': sha256(pathlib.Path(__file__)),
               'probe_sha256': sha256(ROOT / 'integration_q4_mmq_v9r2_probe.cu'),
               'artifact_source_sha256': sha256(ROOT / 'q4_fused_v9_artifact.cpp'),
               'mmq_source_sha256': sha256(ROOT / 'q4_fused_v9_mmq.cu'),
               'public_host_sha256': sha256(SOURCE / 'src/ops/linear/gguf/ggml_mmq_q4_k.cu'),
               'public_header_sha256': sha256(SOURCE / 'src/ops/linear/gguf/ggml_bridge_q4_mmq.h'),
               'mmq_sha256': sha256(SOURCE / 'third_party/ggml-quants/src/ggml-cuda/mmq.cuh'),
               'mmq_loader_sha256': sha256(SOURCE / 'third_party/ggml-quants/src/ggml-cuda/mmq-load-tiles.cuh'),
               'gpu_executed': False}
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


def run(args):
    if not BINARY.is_file():
        raise SystemExit('run `python run_q4_mmq_v9r2.py build` first')
    if not args.tag or not __import__('re').fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]{0,63}', args.tag):
        raise SystemExit('--tag is required and must contain 1-64 safe filename characters')
    columns = [int(value) for value in args.t.split(',')]
    if not columns or any(value <= 0 for value in columns) or len(set(columns)) != len(columns):
        raise SystemExit('--t must be a comma-separated list of unique positive integers')
    if args.activation_mode != 'real' or args.activation is None or columns != [1280]:
        raise SystemExit('v9r2 requires the captured real BF16 K5120/T1280 activation')
    expected_activation_bytes = 5120 * 1280 * 2
    if args.activation.stat().st_size != expected_activation_bytes:
        raise ValueError(f'expected {expected_activation_bytes} activation bytes')
    image_id = image_identity(args.image, args.image_id)
    build_receipt = json.loads((RESULTS / 'q4-mmq-integrated-v9r2-build.json').read_text(encoding='utf-8'))
    if build_receipt['binary_sha256'] != sha256(BINARY) or build_receipt['source_sha256'] != source_hashes() or build_receipt['toolchain_image_id'] != image_id:
        raise RuntimeError('Oracle binary, sources or image differ from frozen build')
    require_gpu_unowned()
    artifact_id, payload_base, items = tensor_directory(args.model)
    identities = tuple(args.tensor or ('weight/000001',))
    reports = []
    for identity in identities:
        item = items.get(identity)
        if item is None:
            raise ValueError(f'{identity} is not in the model directory')
        if item['format'] != 'gguf_q4_k' or len(item['shape']) != 2:
            raise ValueError(f'{identity} is not a rank-2 GGUF Q4_K tensor')
        n, k = item['shape']
        if (n, k) != (34816, 5120):
            raise ValueError(f'{identity} has unsupported Q4 field geometry {item["shape"]}')
        raw_row_bytes = k // 256 * 144
        if n * raw_row_bytes != item['bytes']:
            raise ValueError(f'{identity} has inconsistent Q4_K geometry')
        offset = payload_base + item['offset']
        stem = f'q4-mmq-integrated-v9r2-{args.tag}-{identity.replace("/", "-")}-t{args.t.replace(",", "-")}'
        log = RESULTS / (stem + '.jsonl')
        stderr = RESULTS / (stem + '.jsonl.stderr')
        telemetry = RESULTS / (stem + '.telemetry.json')
        summary_path = RESULTS / f'q4-mmq-integrated-v9r2-{args.tag}-t{args.t.replace(",", "-")}-summary.json'
        collisions = [path for path in (log, stderr, telemetry, summary_path) if path.exists()]
        if collisions:
            raise FileExistsError(f'preserving existing receipt {collisions[0]}')
        activation_mount = []
        activation_path = '-'
        if args.activation:
            activation_mount = ['-v', linux(args.activation) + ':/model-activation.bin:ro']
            activation_path = '/model-activation.bin'
        command = ['docker', 'run', '--rm', '--gpus', 'all', '-v', linux(ROOT) + ':/experiment',
                   '-v', linux(args.model) + ':/model.ninfer:ro', *activation_mount,
                   '--entrypoint', '/bin/sh', args.image,
                   '/experiment/results/run-q4-mmq-integrated-v9r2.sh',
                   f'/experiment/results/{BINARY.name}', '/model.ninfer', str(offset), str(n), str(k),
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

        metadata = {'command': command, 'oracle_tag': args.tag, 'activation_mode': args.activation_mode,
                    'toolchain_image': args.image, 'toolchain_image_id': image_id,
                    'binary_sha256': sha256(BINARY), 'runner_sha256': sha256(pathlib.Path(__file__)),
                    'probe_sha256': sha256(ROOT / 'integration_q4_mmq_v9r2_probe.cu'),
                    'artifact_source_sha256': sha256(ROOT / 'q4_fused_v9_artifact.cpp'),
                    'mmq_source_sha256': sha256(ROOT / 'q4_fused_v9_mmq.cu'),
                    'public_host_sha256': sha256(SOURCE / 'src/ops/linear/gguf/ggml_mmq_q4_k.cu'),
                    'mmq_sha256': sha256(SOURCE / 'third_party/ggml-quants/src/ggml-cuda/mmq.cuh'),
                    'mmq_loader_sha256': sha256(SOURCE / 'third_party/ggml-quants/src/ggml-cuda/mmq-load-tiles.cuh'),
                    'start_epoch_ns': time.time_ns(), 'power_or_clock_controls_changed': False}
        observer = threading.Thread(target=monitor)
        observer.start()
        try:
            result = subprocess.run(['wsl.exe', '-d', 'Ubuntu-24.04', '-u', 'root', '--', *command], capture_output=True)
        finally:
            stop.set()
            observer.join()
        log.write_bytes(result.stdout)
        stderr.write_bytes(result.stderr)
        metadata.update(end_epoch_ns=time.time_ns(), exit_code=result.returncode, samples=samples)
        telemetry.write_text(json.dumps(metadata, indent=2) + '\n')
        if result.returncode:
            raise RuntimeError(f'{identity} public Q4 MMQ oracle failed; inspect {log}.stderr')
        records = [json.loads(line) for line in result.stdout.decode().splitlines() if line.strip()]
        expected_views = {'root', 'odd-origin-slice'}
        expected_views |= {'mlp-gate-split', 'mlp-up-split'}
        if len(records) != len(columns) * len(expected_views) + 1:
            raise RuntimeError(f'{identity} emitted {len(records)} records; inspect {log}')
        cases, footer = records[:-1], records[-1]
        required = {'tensor', 'N', 'K', 'T', 'view', 'row_origin', 'rows', 'activation', 'J', 'I',
                    'threads', 'shared_bytes', 'stream_k', 'blocks', 'fixup', 'fp32_exact', 'bf16_exact',
                    'fp32_canary', 'bf16_canary', 'graph_fp32_exact', 'graph_fp32_canary',
                    'raw_graph_ms', 'unpack_raw_graph_ms', 'public_direct_graph_ms'}
        valid = all(required <= case.keys() and case['tensor'] == identity and case['N'] == n and
                    case['K'] == k and case['T'] in columns and case['activation'] == args.activation_mode and
                    case['view'] in expected_views and case['row_origin'] ==
                    {'root': 0, 'odd-origin-slice': 1, 'mlp-gate-split': 0,
                     'mlp-up-split': 17408}[case['view']] and
                    (case['rows'] == n if case['view'] == 'root' else
                     case['rows'] == 17408 if case['view'] in ('mlp-gate-split', 'mlp-up-split') else case['rows'] > 0) and
                    all(case[key] is True for key in ('fp32_exact', 'bf16_exact', 'fp32_canary',
                                                       'bf16_canary', 'graph_fp32_exact', 'graph_fp32_canary')) and
                    all(isinstance(case[key], list) and len(case[key]) == 5 for key in
                        ('raw_graph_ms', 'unpack_raw_graph_ms', 'public_direct_graph_ms')) for case in cases)
        complete = valid and all({case['view'] for case in cases if case['T'] == value} == expected_views
                                 for value in columns)
        footer_valid = footer.get('tensor') == identity and footer.get('N') == n and footer.get('K') == k and footer.get('full_byte_reconstruction') is True
        if not complete or not footer_valid:
            raise RuntimeError(f'{identity} public Q4 MMQ comparison failed; inspect {log}')
        reports.append({'tensor': identity, 'N': n, 'K': k, 'format': item['format'],
                        'raw_row_bytes': raw_row_bytes, 'source': str(args.model.resolve()),
                        'artifact_id': artifact_id, 'absolute_payload_offset': offset,
                        'columns': args.t, 'activation_mode': args.activation_mode,
                        'activation_file': str(args.activation.resolve()) if args.activation else None,
                        'activation_sha256': sha256(args.activation),
                        'toolchain_image': args.image, 'toolchain_image_id': image_id,
                        'oracle_tag': args.tag, 'gpu_executed': True, 'cases': records})
    report = {'gpu_executed': True, 'toolchain_image': args.image, 'toolchain_image_id': image_id,
              'oracle_tag': args.tag, 'activation_mode': args.activation_mode, 'matrices': reports}
    out = RESULTS / f'q4-mmq-integrated-v9r2-{args.tag}-t{args.t.replace(",", "-")}-summary.json'
    if out.exists():
        raise FileExistsError(f'preserving existing summary {out}')
    out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('build', 'run'))
    parser.add_argument('--model', type=pathlib.Path, default=MODEL)
    parser.add_argument('--image', default=IMAGE_DEFAULT)
    parser.add_argument('--image-id')
    parser.add_argument('--tensor', action='append')
    parser.add_argument('--t', default='1280')
    parser.add_argument('--tag')
    parser.add_argument('--activation-mode', choices=('synthetic', 'real'), default='real')
    parser.add_argument('--activation', type=pathlib.Path,
                        default=RESULTS / 'matrix-activations-mmq-real-v6' / 'bf16-k5120-t1280.bin')
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    if args.activation_mode == 'real' and args.activation is None:
        raise SystemExit('--activation is required with --activation-mode real')
    if args.action == 'build':
        build(args)
    else:
        run(args)


if __name__ == '__main__':
    main()
