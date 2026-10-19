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
PROBE = RESULTS / 'q4-rooted-address-probe'


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


def synthetic_activation(columns):
    source_path = RESULTS / 'activations' / 'q12-k5120-t4-f0.bin'
    source = source_path.read_bytes()
    if len(source) < 32:
        raise ValueError('saved T4 Q4 activation is shorter than its header')
    magic, version, qtype, k, source_columns, fused, payload_bytes, reserved = struct.unpack_from('<8I', source)
    if (magic, version, qtype, k, source_columns, fused, payload_bytes, reserved) != (
            0x31534156, 1, 12, 5120, 4, 0, len(source) - 32, 0):
        raise ValueError('saved T4 activation header or payload is incompatible')
    q_bytes = k * source_columns
    ds_column_bytes = k // 32 * 4
    payload = bytearray(k * columns + ds_column_bytes * columns)
    mapping = [column % source_columns for column in range(columns)]
    for column, source_column in enumerate(mapping):
        payload[column * k:(column + 1) * k] = source[32 + source_column * k:32 + (source_column + 1) * k]
        source_ds = 32 + q_bytes + source_column * ds_column_bytes
        target_ds = k * columns + column * ds_column_bytes
        payload[target_ds:target_ds + ds_column_bytes] = source[source_ds:source_ds + ds_column_bytes]
    header = struct.pack('<8I', magic, version, qtype, k, columns, 0, len(payload), 0)
    target_path = RESULTS / 'activations' / f'q12-k5120-t{columns}-f0.synthetic-from-t4-cycle.bin'
    target_path.write_bytes(header + payload)
    checked = target_path.read_bytes()
    if checked[:32] != header or checked[32:] != payload:
        raise ValueError('synthetic activation fixture failed byte verification')
    for column, source_column in enumerate(mapping):
        if checked[32 + column * k:32 + (column + 1) * k] != source[32 + source_column * k:32 + (source_column + 1) * k]:
            raise ValueError('synthetic Q8 activation column differs from its source column')
        source_ds = 32 + q_bytes + source_column * ds_column_bytes
        target_ds = 32 + k * columns + column * ds_column_bytes
        if checked[target_ds:target_ds + ds_column_bytes] != source[source_ds:source_ds + ds_column_bytes]:
            raise ValueError('synthetic half2 activation scale/sum column differs from its source column')
    return target_path, mapping, source_path


def validate_host_bounds():
    def accepted(k, rows, row_begin):
        return k == 5120 and 0 < rows <= 16384 and 0 <= row_begin <= 16384 and row_begin + rows <= 16384

    cases = [
        ('root', 5120, 16384, 0, True),
        ('slice_2048', 5120, 2048, 6144, True),
        ('slice_4096', 5120, 4096, 8192, True),
        ('upper_edge', 5120, 2048, 14336, True),
        ('wrong_k', 5121, 2048, 0, False),
        ('empty_rows', 5120, 0, 0, False),
        ('oversize_rows', 5120, 16385, 0, False),
        ('origin_overflow', 5120, 1, 16385, False),
        ('range_overflow', 5120, 2048, 14337, False),
    ]
    records = []
    for name, k, rows, origin, expected in cases:
        result = accepted(k, rows, origin)
        if result != expected:
            raise ValueError(f'host row/origin boundary case {name} failed')
        records.append({'case': name, 'K': k, 'rows': rows, 'row_begin': origin,
                        'accepted': result, 'expected': expected})
    source = (ROOT / 'source/src/ops/linear/gguf/q4_fields.cuh').read_text()
    if 'args.k != 5120' not in source or 'args.rows > 16384' not in source or 'row_begin + std::uint64_t(args.rows) > 16384' not in source:
        raise ValueError('Q4 host launch bounds differ from checked CPU fixtures')
    report = {'cpu_only': True, 'gpu_executed': False,
              'oracle': 'launch host guard K=5120, rows in 1..16384, row_begin in 0..16384, row_begin+rows<=16384',
              'cases': records}
    (RESULTS / 'q4-integrated-host-bounds.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def prepare_synthetic_activations():
    records = []
    for columns in (2, 5, 6, 7, 8):
        path, mapping, source = synthetic_activation(columns)
        records.append({'columns': columns, 'path': str(path.resolve()),
                        'kind': 'synthetic-column-copy', 'source_capture': str(source.resolve()),
                        'source_columns': mapping, 'gpu_executed': False})
    report = {'cpu_only': True, 'gpu_executed': False,
              'description': 'Q8 bytes and paired-half scale/sum bytes copied intact from saved T4 columns; columns cycle modulo four',
              'fixtures': records, 'host_bounds': validate_host_bounds()}
    (RESULTS / 'q4-integrated-synthetic-activation-fixtures.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


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
    sources = ['/experiment/q4_rooted_address_probe.cu',
               '/experiment/source/src/artifact/q4_fields.cpp',
               '/experiment/source/src/core/weight_view.cpp',
               '/experiment/source/src/core/tensor.cpp',
               '/experiment/source/src/core/dtype.cpp',
               '/experiment/source/src/core/q4_fields.cu']
    archives = ['/out/libs/libninfer_artifact.a', '/out/libs/libninfer_ops.a',
                '/out/libs/libninfer_core.a', '/out/libs/libninfer_ggml_quants.a']
    command = ['docker', 'run', '--rm', '-v', linux(ROOT) + ':/experiment',
               '-v', linux(VENDOR) + ':/vendor:ro', '--entrypoint',
               '/usr/local/cuda/bin/nvcc', IMAGE, '-O3', '-DNDEBUG', '-std=c++20',
               '-arch=sm_86', '-lineinfo', '-DNINFER_SM8X_COMPAT=1',
               '-Xptxas=-v', *['-I' + path for path in includes], *sources,
               '-Xlinker', '--start-group', *archives, '-Xlinker', '--end-group',
               '-lcublas', '-lcudart', '-ldl', '-lpthread', '-o',
               '/experiment/results/q4-rooted-address-probe']
    output = run_docker(command, RESULTS / 'q4-rooted-address-build.txt')
    build_log = (RESULTS / 'q4-rooted-address-build.txt.stderr').read_text(errors='replace')
    registers = {'current': {}, 'rooted': {}}
    current_width = None
    current_variant = None
    for line in build_log.splitlines():
        for width in range(1, 9):
            if f'kernelILi{width}ELj12ELj0ELj13ELj2' in line and 'q4_rooted_fields' not in line:
                current_width = width
                current_variant = 'current'
            elif f'kernelILi{width}ELj12ELj0ELj13ELj2' in line and 'q4_rooted_fields' in line:
                current_width = width
                current_variant = 'rooted'
        if 'Used ' in line and ' registers' in line:
            if current_width is not None and current_variant is not None:
                registers[current_variant][f'T{current_width}'] = int(line.split('Used ', 1)[1].split(' registers', 1)[0])
                current_width = None
                current_variant = None
    report = {'build_succeeded': True, 'gpu_executed': False, 'binary': str(PROBE),
              'toolchain_image': IMAGE, 'vendor_headers': str(VENDOR),
              'registers_per_thread': registers, 'build_stdout': output[-2000:]}
    (RESULTS / 'q4-rooted-address-build.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


def validate_rooted_addresses():
    k = 5120
    blocks_per_row = k // 256
    regions = [('root', 0, 16384), ('gdn_up', 0, 10240), ('gdn_down', 10240, 6144)]
    records = []
    for name, origin, rows in regions:
        checks = 0
        for row in (0, rows - 1, rows // 2):
            for iteration in range(k // 1024):
                for lane in range(32):
                    local_block = row * blocks_per_row + 4 * iteration + lane // 8
                    rebased_code_address = origin * blocks_per_row * 128 + local_block * 128
                    rebased_scale_address = origin * blocks_per_row * 12 + local_block * 12
                    global_block = (origin + row) * blocks_per_row + 4 * iteration + lane // 8
                    if rebased_code_address != global_block * 128 or rebased_scale_address != global_block * 12:
                        raise ValueError(f'{name} rooted address mismatch')
                    if global_block >= 16384 * blocks_per_row or global_block > 0xFFFFFFFF:
                        raise ValueError(f'{name} global block exceeds root or uint32 bounds')
                    checks += 1
        records.append({'view': name, 'row_begin': origin, 'rows': rows,
                        'code_row_stride': blocks_per_row * 128,
                        'metadata_row_stride': blocks_per_row * 12,
                        'checked_addresses': checks, 'exact_root_equivalence': True})
    source = (ROOT / 'q4_rooted_fields_private.cuh').read_text()
    if 'args.weight -= row_begin * row_code_bytes' not in source or 'metadata.scales -= row_begin * row_scale_bytes' not in source:
        raise ValueError('private rooted pointer rebasing differs from checked CPU oracle')
    if 'global_block = (std::uint32_t(row_begin) + std::uint32_t(row))' not in source:
        raise ValueError('private rooted global block expression differs from checked CPU oracle')
    report = {'cpu_only': True, 'gpu_executed': False,
              'equivalence': 'view-relative code/meta pointers minus row_begin strides equal root pointers indexed by global_block; d/dmin stay at tensor roots',
              'regions': records, 'root_max_block': 16384 * blocks_per_row - 1,
              'uint32_safe': 16384 * blocks_per_row - 1 <= 0xFFFFFFFF}
    (RESULTS / 'q4-rooted-address-cpu-validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


def run(args):
    artifact_id, payload_base, items = tensor_directory(args.model)
    item = items.get(args.tensor)
    if item is None or item['format'] != 'gguf_q4_k' or item['shape'] != [16384, 5120]:
        raise ValueError(f'{args.tensor} is not a supported [16384,5120] GGUF Q4_K tensor')
    if args.t not in (1, 3, 4):
        raise ValueError('rooted address oracle is limited to captured T1, T3, and T4')
    activation_kind = 'captured'
    activation = RESULTS / 'activations' / f'q12-k5120-t{args.t}-f0.bin'
    if not activation.is_file():
        raise ValueError(f'missing activation capture {activation.name}')
    slice_rows = min(args.slice_rows, item['shape'][0])
    slice_begin = (item['shape'][0] // 2 // 64) * 64 if args.slice_begin is None else args.slice_begin
    if slice_begin < 0 or slice_begin + slice_rows > item['shape'][0]:
        raise ValueError('requested slice exceeds tensor rows')
    tensor_offset = payload_base + item['offset']
    log = RESULTS / f'q4-rooted-address-{args.tensor.replace("/", "-")}-t{args.t}.jsonl'
    command = ['docker', 'run', '--rm', '--gpus', 'all', '-v', linux(ROOT) + ':/experiment',
               '-v', linux(args.model) + ':/model.ninfer:ro', '--entrypoint',
               '/bin/sh', IMAGE, '/experiment/results/q4-rooted-address-launch.sh',
               '/model.ninfer', str(tensor_offset), str(item['shape'][0]), str(item['shape'][1]),
               args.tensor, '/experiment/' + activation.relative_to(ROOT).as_posix(),
               str(slice_begin), str(slice_rows)]
    output = run_docker(command, log)
    records = [json.loads(line) for line in output.splitlines() if line.strip()]
    summary = records[-1]
    summary.update({'tensor': args.tensor, 'format': item['format'], 'artifact_id': artifact_id,
                    'absolute_payload_offset': tensor_offset, 'activation_columns': args.t,
                    'activation_snapshot': str(activation.resolve()),
                    'activation_kind': activation_kind,
                    'activation_source_capture': None,
                    'synthetic_source_columns': None,
                    'oracle': 'actual artifact encoder and native_weight, comparing original vector_product, current q4_fields.cuh and private rooted-address variant with core unpack oracle',
                    'gpu_executed': True, 'records': records[:-1]})
    report = RESULTS / f'q4-rooted-address-{args.tensor.replace("/", "-")}-t{args.t}-summary.json'
    report.write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('build', 'cpu-validate', 'run'))
    parser.add_argument('--model', type=pathlib.Path, default=MODEL)
    parser.add_argument('--tensor', default='weight/000060')
    parser.add_argument('--t', type=int, choices=(1, 3, 4), default=1)
    parser.add_argument('--slice-rows', type=int, default=4096)
    parser.add_argument('--slice-begin', type=int)
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    if args.action == 'build':
        build()
    elif args.action == 'cpu-validate':
        validate_rooted_addresses()
    else:
        run(args)


if __name__ == '__main__':
    main()

