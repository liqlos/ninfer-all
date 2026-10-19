import argparse
import hashlib
import json
import pathlib
import struct
import subprocess


ROOT = pathlib.Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
MODEL = pathlib.Path(r'C:\AI\staging\qwen38-jonathan-ninfer\qwen38-jonathan-q4km-proposal.ninfer')
VENDOR = pathlib.Path(r'C:\AI\staging\ninfer-all-pin\third_party\ggml-quants')
IMAGE = 'ninfer-lossless-q6-oracle:sm86'
BINARY = RESULTS / 'q4q6-paired-q4-probe-v2'
TENSOR = 'weight/000060'
TENSOR_FORMAT = 'gguf_q4_k'


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


def run_docker(command, receipt, gpu=False):
    if receipt.exists() or receipt.with_suffix(receipt.suffix + '.stderr').exists():
        raise FileExistsError(f'preserving existing receipt {receipt}')
    result = subprocess.run(['wsl.exe', '-d', 'Ubuntu-24.04', '-u', 'root', '--', *command], capture_output=True)
    receipt.write_bytes(result.stdout)
    receipt.with_suffix(receipt.suffix + '.stderr').write_bytes(result.stderr)
    if result.returncode:
        raise RuntimeError(f'command failed with exit {result.returncode}; see {receipt}.stderr')
    return result.stdout.decode()


def source_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes():
    files = [pathlib.Path(__file__), ROOT / 'q4q6_paired_probe.cu', ROOT / 'q4q6_paired_fields.cuh', ROOT / 'q6_regbalance_fields.cuh']
    for base in (ROOT / 'source/src', ROOT / 'source/include', VENDOR):
        files.extend(path for path in base.rglob('*') if path.is_file() and path.suffix in ('.h', '.hpp', '.cuh', '.cu', '.cpp', '.c'))
    return {str(path.resolve()): source_hash(path) for path in sorted(set(files))}


def image_identity():
    result = subprocess.run(['wsl.exe', '-d', 'Ubuntu-24.04', '-u', 'root', '--', 'docker', 'image', 'inspect', IMAGE, '--format', '{{.Id}}'], capture_output=True, text=True)
    identity = result.stdout.strip()
    if result.returncode or not identity.startswith('sha256:'):
        raise RuntimeError('cannot identify the oracle image')
    return identity


def build():
    if BINARY.exists() or BINARY.with_suffix('.build.json').exists():
        raise FileExistsError('preserving existing paired Q4 binary or build receipt')
    includes = ['/experiment/source/src', '/vendor/src', '/vendor/include',
                '/src/src', '/src/third_party/ggml-quants/src',
                '/src/third_party/ggml-quants/include', '/src/third_party']
    sources = ['/experiment/q4q6_paired_probe.cu',
               '/experiment/source/src/artifact/q4_fields.cpp',
               '/experiment/source/src/core/weight_view.cpp',
               '/experiment/source/src/core/tensor.cpp',
               '/experiment/source/src/core/dtype.cpp',
               '/experiment/source/src/core/q4_fields.cu']
    archives = ['/out/libs/libninfer_artifact.a', '/out/libs/libninfer_ops.a',
                '/out/libs/libninfer_core.a', '/out/libs/libninfer_ggml_quants.a']
    build_receipt = RESULTS / 'q4q6-paired-q4-build-v2.txt'
    command = ['docker', 'run', '--rm', '-v', linux(ROOT) + ':/experiment',
               '-v', linux(VENDOR) + ':/vendor:ro', '--entrypoint',
               '/usr/local/cuda/bin/nvcc', IMAGE, '-O3', '-DNDEBUG', '-std=c++20',
               '-arch=sm_86', '-lineinfo', '-DNINFER_SM8X_COMPAT=1',
               '-Xptxas=-v', '-DPAIRED_Q4=1', *['-I' + path for path in includes], *sources,
               '-Xlinker', '--start-group', *archives, '-Xlinker', '--end-group',
               '-lcublas', '-lcudart', '-ldl', '-lpthread', '-o', '/experiment/' + BINARY.relative_to(ROOT).as_posix()]
    frozen_sources = source_hashes()
    frozen_image = image_identity()
    output = run_docker(command, build_receipt)
    if source_hashes() != frozen_sources or image_identity() != frozen_image:
        raise RuntimeError('sources or image changed during build')
    metadata = {
        'kind': 'q4',
        'binary': str(BINARY),
        'binary_sha256': source_hash(BINARY),
        'probe_source_sha256': source_hash(ROOT / 'q4q6_paired_probe.cu'),
        'paired_fields_source_sha256': source_hash(ROOT / 'q4q6_paired_fields.cuh'),
        'q4_fields_source_sha256': source_hash(ROOT / 'source/src/ops/linear/gguf/q4_fields.cuh'),
        'q6_regbalance_source_sha256': source_hash(ROOT / 'q6_regbalance_fields.cuh'),
        'q6_fields_source_sha256': source_hash(ROOT / 'source/src/ops/linear/gguf/q6_fields.cuh'),
        'toolchain_image': IMAGE,
        'toolchain_image_id': frozen_image,
        'source_sha256': frozen_sources,
        'gpu_executed': False,
        'build_command': command,
        'build_stdout_tail': output[-2000:],
    }
    BINARY.with_suffix('.build.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2))


def activation_path(kind, path):
    if path:
        return path
    return RESULTS / 'activations' / ('q12-k5120-t4-f0.bin' if kind == 'q4' else 'q14-k5120-t4-f0.bin')


def run(args):
    if not args.tag or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in args.tag):
        raise ValueError('Use a unique safe --tag')
    if args.tensor != TENSOR:
        raise ValueError('only the verified Q4 parent weight/000060 is supported')
    kind, expected_format = 'q4', TENSOR_FORMAT
    artifact_id, payload_base, items = tensor_directory(args.model)
    item = items.get(args.tensor)
    if item is None or item['format'] != expected_format or item['shape'] != [16384, 5120]:
        raise ValueError(f'{args.tensor} is not the supported [16384,5120] {expected_format} parent')
    binary = BINARY
    build_json = BINARY.with_suffix('.build.json')
    if not binary.is_file() or not build_json.is_file():
        raise SystemExit('run `python run_q4q6_paired.py build` first')
    metadata = json.loads(build_json.read_text())
    if metadata['source_sha256'] != source_hashes() or metadata['toolchain_image_id'] != image_identity():
        raise RuntimeError('compiled sources or oracle image differ from frozen build')
    if metadata['probe_source_sha256'] != source_hash(ROOT / 'q4q6_paired_probe.cu') or \
       metadata['paired_fields_source_sha256'] != source_hash(ROOT / 'q4q6_paired_fields.cuh') or \
       metadata['binary_sha256'] != source_hash(BINARY):
        raise RuntimeError('source or binary differs from frozen build receipt')
    if args.action == 'bench' and args.t != 4:
        raise ValueError('paired split probe is T4 only')
    activation = activation_path(kind, args.activation)
    if args.action == 'bench':
        if not activation.is_file():
            raise ValueError(f'missing activation snapshot {activation}')
        activation_bytes = activation.read_bytes()
        if len(activation_bytes) < 32:
            raise ValueError('activation snapshot is shorter than its header')
        header = struct.unpack('<8I', activation_bytes[:32])
        expected_payload = 5120 * 4 + (5120 // 32) * 4 * 4
        if header[:6] != (0x31534156, 1, 12, 5120, 4, 0) or header[6] != expected_payload or \
           header[7] != 0 or len(activation_bytes) != 32 + expected_payload:
            raise ValueError('activation snapshot header or payload does not match the captured T4 Q4 input')
    begin = (16384 // 2 - args.slice_rows // 2) // 64 * 64 if args.slice_begin is None else args.slice_begin
    if args.slice_rows <= 0 or begin < 0 or begin + args.slice_rows > 16384:
        raise ValueError('requested verification slice exceeds the parent')
    tensor_offset = payload_base + item['offset']
    suffix = f'{kind}-{args.action}-{args.tag}'
    receipt = RESULTS / f'q4q6-paired-{suffix}.jsonl'
    summary_path = RESULTS / f'q4q6-paired-{suffix}-summary.json'
    if summary_path.exists():
        raise FileExistsError('preserving existing paired summary')
    command = ['docker', 'run', '--rm']
    if args.action == 'bench':
        command.append('--gpus')
        command.append('all')
    command.extend(['-v', linux(ROOT) + ':/experiment', '-v', linux(args.model) + ':/model.ninfer:ro',
                    '--entrypoint', '/experiment/' + binary.relative_to(ROOT).as_posix(), IMAGE,
                    '/model.ninfer', str(tensor_offset), '16384', '5120', args.tensor,
                    '/experiment/' + activation.relative_to(ROOT).as_posix(), str(begin),
                    str(args.slice_rows), args.action])
    output = run_docker(command, receipt, gpu=args.action == 'bench')
    records = [json.loads(line) for line in output.splitlines() if line.strip()]
    summary = {'action': args.action, 'tensor': args.tensor, 'format': expected_format,
               'binary_sha256': metadata['binary_sha256'], 'source_sha256': metadata['source_sha256'],
               'toolchain_image_id': metadata['toolchain_image_id'],
               'artifact_id': artifact_id, 'absolute_payload_offset': tensor_offset,
               'activation_snapshot': str(activation.resolve()) if args.action == 'bench' else None,
               'activation_columns': 4 if args.action == 'bench' else None,
               'working_set_copies': 3 if args.action == 'bench' else None,
               'cpu_scan_scope': 'artifact encoder geometry and encoded size only; full byte reconstruction is gated on GPU bench',
               'baseline_pair_capture': 'two sibling vector_product calls inside each single captured graph and event interval',
               'accepted_fields_control': 'two accepted compressed-field sibling launches inside each single captured graph and event interval',
               'candidate': 'one full-parent paired-output kernel stores to independent top and bottom VecStore buffers',
               'full_payload_gpu_unpack_exact': any(record.get('kind') == 'full_raw_unpack' and record.get('byte_exact') for record in records),
               'records': records, 'gpu_executed': args.action == 'bench'}
    if args.action == 'scan':
        scan_record = next((record for record in records if record.get('kind') == 'cpu_scan'), None)
        if not scan_record or not scan_record.get('encoder_succeeded') or scan_record.get('gpu_executed'):
            raise RuntimeError('CPU encoder scan failed')
    else:
        pair_records = [record for record in records if record.get('kind') == 'paired_split']
        required = ('raw_vs_fields_direct_exact', 'fields_vs_candidate_direct_exact',
                    'raw_vs_candidate_direct_exact', 'all_arms_graph_exact', 'all_output_canaries_exact')
        if {record.get('dtype') for record in pair_records} != {'fp32', 'bf16'} or any(record.get(key) is not True for record in pair_records for key in required):
            raise RuntimeError('paired direct/graph/canary exactness gate failed')
        if not summary['full_payload_gpu_unpack_exact'] or not any(record.get('matrix') == args.tensor and record.get('all_checks_exact') is True for record in records):
            raise RuntimeError('full payload unpack or integrated root/view oracle is missing or false')
    summary_path.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('build', 'scan', 'bench'))
    parser.add_argument('--model', type=pathlib.Path, default=MODEL)
    parser.add_argument('--tensor', default='weight/000060')
    parser.add_argument('--activation', type=pathlib.Path)
    parser.add_argument('--t', type=int, choices=(4,), default=4)
    parser.add_argument('--slice-rows', type=int, default=4096)
    parser.add_argument('--slice-begin', type=int)
    parser.add_argument('--tag', default='')
    parser.add_argument('--toolchain', choices=('cu13',), default='cu13')
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    if args.action == 'build':
        build()
    else:
        run(args)


if __name__ == '__main__':
    main()
