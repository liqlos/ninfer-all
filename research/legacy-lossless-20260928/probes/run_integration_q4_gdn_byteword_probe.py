import argparse
import hashlib
import json
import pathlib
import struct
import subprocess


ROOT = pathlib.Path(__file__).resolve().parent
SOURCE = ROOT / 'source'
RESULTS = ROOT / 'results'
ACTIVATIONS = RESULTS / 'activations'
MODEL = pathlib.Path(r'C:\AI\staging\qwen38-jonathan-ninfer\qwen38-jonathan-q4km-proposal.ninfer')
IMAGE = 'ninfer-lossless-word-v6-oracle:sm86'
PROBE = RESULTS / 'integration-q4-gdn-byteword-probe'
ORACLE = RESULTS / 'q4-fields-word-cpu-oracle-v6'


def linux(path):
    path = pathlib.Path(path).resolve()
    return '/mnt/' + path.drive[0].lower() + '/' + '/'.join(path.parts[1:])


def execute(command, stdout_path, stderr_path=None):
    result = subprocess.run(['wsl.exe', '-d', 'Ubuntu-24.04', '-u', 'root', '--', *map(str, command)],
                            capture_output=True)
    stdout_path.write_bytes(result.stdout)
    if stderr_path is None:
        stderr_path = stdout_path.with_suffix(stdout_path.suffix + '.stderr')
    stderr_path.write_bytes(result.stderr)
    if result.returncode:
        raise RuntimeError(f'command failed with exit {result.returncode}; see {stderr_path}')
    return result.stdout


def tensor_directory(model):
    with model.open('rb') as stream:
        magic, length, artifact_id = struct.unpack('<8sQ16s', stream.read(32))
        if magic != b'NINFER\0\3':
            raise ValueError('NInfer v3 artifact required')
        directory = json.loads(stream.read(length))
    payload = (32 + length + 4095) // 4096 * 4096
    items = {item['id']: item for item in directory['objects']}
    return artifact_id.hex(), payload, items


def build_cpu_oracle():
    command = ['/usr/bin/g++', '-O3', '-DNDEBUG', '-std=c++20', '-I' + linux(SOURCE / 'src'),
               '-I' + linux(SOURCE / 'third_party'), linux(ROOT / 'q4_fields_word_cpu_oracle.cpp'),
               linux(SOURCE / 'src/core/weight_view.cpp'), linux(SOURCE / 'src/core/tensor.cpp'),
               linux(SOURCE / 'src/core/dtype.cpp'), linux(SOURCE / 'src/artifact/q4_fields.cpp'),
               '-o', linux(ORACLE)]
    execute(command, RESULTS / 'q4-fields-word-v6-cpu-build.stdout', RESULTS / 'q4-fields-word-v6-cpu-build.stderr')
    print(json.dumps({'binary': str(ORACLE), 'cpu_only': True, 'gpu_executed': False}))


def cpu_scan(args):
    if not ORACLE.is_file():
        build_cpu_oracle()
    artifact_id, payload_base, items = tensor_directory(args.model)
    selected = []
    for identity, item in items.items():
        shape = item.get('shape')
        if item.get('format') != 'gguf_q4_k' or len(shape) != 2:
            continue
        if shape == [16384, 5120]:
            selected.append((identity, item, 'scan'))
        elif shape == [5120, 17408]:
            selected.append((identity, item, 'scan'))
    logs = []
    counts = {'gdn_eligible': 0, 'gdn_fallback': 0, 'down_eligible': 0, 'down_fallback': 0}
    for identity, item, eligibility in selected:
        offset = payload_base + item['offset']
        target = RESULTS / f'q4-fields-word-v6-cpu-{identity.replace("/", "-")}.json'
        output = execute([linux(ORACLE), linux(args.model), offset, item['shape'][0], item['shape'][1], identity, eligibility], target)
        record = json.loads(output.decode().strip())
        record.update({'artifact_id': artifact_id, 'source': str(args.model.resolve()),
                       'absolute_payload_offset': offset, 'format': item['format'],
                       'eligibility_expected': eligibility, 'cpu_only': True, 'gpu_executed': False})
        target.write_text(json.dumps(record, indent=2) + '\n')
        if record.get('eligible'):
            if item['shape'] == [16384, 5120]:
                counts['gdn_eligible'] += 1
            else:
                counts['down_eligible'] += 1
        else:
            counts['gdn_fallback' if item['shape'] == [16384, 5120] else 'down_fallback'] += 1
        logs.append(record)
    expected = {'gdn_eligible': 19, 'gdn_fallback': 5, 'down_eligible': 24, 'down_fallback': 8}
    if args.assert_counts and counts != expected:
        raise RuntimeError(f'Q4 fields candidate counts {counts} differ from expected {expected}')
    report = {'cpu_only': True, 'gpu_executed': False, 'artifact_id': artifact_id,
              'source': str(args.model.resolve()), 'counts': counts,
              'expected_counts': expected if args.assert_counts else None, 'tensors': logs}
    path = RESULTS / 'q4-fields-word-v6-cpu-scan.json'
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


def image_build():
    execute(['docker', 'build', '-f', linux(ROOT / 'Dockerfile.q4-word-oracle'),
             '-t', IMAGE, linux(ROOT)], RESULTS / 'q4-word-v6-image-build.stdout',
            RESULTS / 'q4-word-v6-image-build.stderr')
    print(json.dumps({'image': IMAGE, 'built': True, 'gpu_executed': False}))


def build_probe():
    includes = ['/experiment/source/src', '/experiment/source/third_party/ggml-quants/src',
                '/experiment/source/third_party/ggml-quants/include', '/src/src',
                '/src/third_party/ggml-quants/src', '/src/third_party/ggml-quants/include']
    archives = ['/out/libs/libninfer_artifact.a', '/out/libs/libninfer_ops.a',
                '/out/libs/libninfer_core.a', '/out/libs/libninfer_ggml_quants.a']
    command = ['docker', 'run', '--rm', '-v', linux(ROOT) + ':/experiment', '--entrypoint',
               '/usr/local/cuda/bin/nvcc', IMAGE, '-O3', '-DNDEBUG', '-std=c++20', '-arch=sm_86',
               '-lineinfo', '-DNINFER_SM8X_COMPAT=1', '-Xptxas=-v',
               *['-I' + path for path in includes], '/experiment/integration_q4_gdn_byteword_probe.cu',
               '-Xlinker', '--start-group', *archives, '-Xlinker', '--end-group',
               '-lcublas', '-lcudart', '-ldl', '-lpthread', '-o', '/experiment/results/integration-q4-gdn-byteword-probe']
    execute(command, RESULTS / 'integration-q4-gdn-byteword-build.stdout', RESULTS / 'integration-q4-gdn-byteword-build.stderr')
    print(json.dumps({'binary': str(PROBE), 'toolchain_image': IMAGE, 'gpu_executed': False,
                      'variants': ['v6_public_word_default_grid', 'byte_reconstructed_cap6', 'word_cap6']}))


def sha256(path):
    digest = hashlib.sha256()
    with pathlib.Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def record_build():
    image_id = execute(['docker', 'image', 'inspect', '--format', '{{.Id}}', IMAGE],
                       RESULTS / 'q4-word-v6-image-inspect.stdout',
                       RESULTS / 'q4-word-v6-image-inspect.stderr').decode().strip()
    source_build_image_id = execute(['docker', 'image', 'inspect', '--format', '{{.Id}}',
                                     'ninfer-lossless-word-v6-build:sm86'],
                                    RESULTS / 'q4-word-v6-build-image-inspect.stdout',
                                    RESULTS / 'q4-word-v6-build-image-inspect.stderr').decode().strip()
    runtime_image_id = execute(['docker', 'image', 'inspect', '--format', '{{.Id}}',
                                'ninfer-lossless-word-v6:sm86'],
                               RESULTS / 'q4-word-v6-runtime-image-inspect.stdout',
                               RESULTS / 'q4-word-v6-runtime-image-inspect.stderr').decode().strip()
    record = {'gpu_executed': False, 'image': IMAGE, 'image_id': image_id,
              'source_build_image': 'ninfer-lossless-word-v6-build:sm86',
              'source_build_image_id': source_build_image_id,
              'runtime_image': 'ninfer-lossless-word-v6:sm86', 'runtime_image_id': runtime_image_id,
              'binary': str(PROBE.resolve()), 'binary_sha256': sha256(PROBE),
              'probe_source': str((ROOT / 'integration_q4_word_probe.cu').resolve()),
              'probe_source_sha256': sha256(ROOT / 'integration_q4_word_probe.cu'),
              'runner': str(pathlib.Path(__file__).resolve()), 'runner_sha256': sha256(__file__),
              'oracle_source_sha256': sha256(ROOT / 'q4_fields_word_cpu_oracle.cpp'),
              'oracle_binary_sha256': sha256(ORACLE),
              'dockerfile_sha256': sha256(ROOT / 'Dockerfile.q4-word-oracle'),
              'linked_archives': ['/out/libs/libninfer_artifact.a', '/out/libs/libninfer_ops.a',
                                  '/out/libs/libninfer_core.a', '/out/libs/libninfer_ggml_quants.a'],
              'candidate_api': 'public ninfer::ops::gguf::vector_q4_fields',
              'baseline_api': 'ninfer::ops::gguf::vector_product Q4_K',
              'cpu_scan': str((RESULTS / 'q4-fields-word-v6-cpu-scan.json').resolve()),
              'cpu_scan_sha256': sha256(RESULTS / 'q4-fields-word-v6-cpu-scan.json')}
    path = RESULTS / 'integration-q4-gdn-byteword-build.json'
    path.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


def activation_for(k, t):
    candidate = ACTIVATIONS / f'q12-k{k}-t{t}-f0.bin'
    if candidate.is_file():
        return candidate, 'captured-q4-tag'
    source = ACTIVATIONS / f'q14-k{k}-t{t}-f0.bin'
    if source.is_file():
        body = source.read_bytes()
        fields = list(struct.unpack_from('<8I', body))
        if fields[:6] != [0x31534156, 1, 14, k, t, 0] or fields[6] != len(body) - 32 or fields[7] != 0:
            raise ValueError(f'{source.name} has an invalid Q14 activation header')
        fields[2] = 12
        retagged = ACTIVATIONS / f'q12-k{k}-t{t}-f0.word-v6-retagged-from-q14.bin'
        retagged.write_bytes(struct.pack('<8I', *fields) + body[32:])
        return retagged, 'captured-q14-payload-retagged-q4-header'
    if t not in (1, 8):
        raise ValueError(f'missing activation capture K={k} T={t}')
    source = activation_for(k, 4)[0]
    data = source.read_bytes()
    header = struct.unpack_from('<8I', data)
    if header[:6] != (0x31534156, 1, 12, k, 4, 0) or header[6] != len(data) - 32 or header[7] != 0:
        raise ValueError('T4 source activation header is incompatible')
    q_bytes = k * 4
    scale_bytes = (k // 32) * 4
    payload = bytearray(k * t + scale_bytes * t)
    mapping = [column % 4 for column in range(t)]
    for column, source_column in enumerate(mapping):
        payload[column * k:(column + 1) * k] = data[32 + source_column * k:32 + (source_column + 1) * k]
        src = 32 + q_bytes + source_column * scale_bytes
        dst = k * t + column * scale_bytes
        payload[dst:dst + scale_bytes] = data[src:src + scale_bytes]
    target = ACTIVATIONS / f'q12-k{k}-t{t}-f0.word-v6-synthetic-from-t4-cycle.bin'
    target.write_bytes(struct.pack('<8I', 0x31534156, 1, 12, k, t, 0, len(payload), 0) + payload)
    return target, 'synthetic-column-copy-from-t4-cycle'


def run(args):
    if not PROBE.is_file():
        raise SystemExit('build the v6-linked probe first')
    artifact_id, payload_base, items = tensor_directory(args.model)
    identities = args.tensor or ['weight/000060', 'weight/000224']
    t_values = args.t or [1, 3, 4, 8]
    reports = []
    for identity in identities:
        item = items.get(identity)
        expected_shape = [16384, 5120] if identity == 'weight/000060' else [5120, 17408]
        if item is None or item.get('format') != 'gguf_q4_k' or item.get('shape') != expected_shape:
            raise ValueError(f'{identity} does not match an expected Q4_K test root')
        n, k = item['shape']
        if n == 16384:
            slice_begin, slice_rows, nested_begin, nested_rows = 10240, 6144, 1, 1023
        else:
            slice_begin, slice_rows, nested_begin, nested_rows = 1024, 3072, 1, 1023
        for t in t_values:
            activation, activation_kind = activation_for(k, t)
            offset = payload_base + item['offset']
            suffix = f'{identity.replace("/", "-")}-t{t}'
            stdout = RESULTS / f'integration-q4-gdn-byteword-{suffix}.jsonl'
            if args.reuse_existing and stdout.is_file():
                output = stdout.read_bytes()
            else:
                docker_command = ['docker', 'run', '--rm', '--gpus', 'all', '-v', linux(ROOT) + ':/experiment',
                                  '-v', linux(args.model) + ':/model.ninfer:ro', '--entrypoint',
                                  '/experiment/results/' + PROBE.name, IMAGE, '/model.ninfer', str(offset), str(n), str(k), identity,
                                  '/experiment/' + activation.relative_to(ROOT).as_posix(), str(slice_begin), str(slice_rows),
                                  str(nested_begin), str(nested_rows)]
                output = execute(docker_command, stdout, stdout.with_suffix(stdout.suffix + '.stderr'))
            records = [json.loads(line) for line in output.decode().splitlines() if line.strip()]
            summary = records[-1]
            cases = records[:-1]
            unpack = [case for case in cases if case.get('kind') == 'full_raw_unpack']
            operators = [case for case in cases if 'bit_exact' in case]
            if (not summary.get('all_checks_exact') or not unpack or not operators or
                    not all(case.get('byte_exact') for case in unpack) or
                    not all(case.get('bit_exact') and case.get('canaries_intact') for case in operators)):
                raise RuntimeError(f'{identity} T={t} integration oracle failed; see {stdout}')
            record = {'tensor': identity, 'format': item['format'], 'shape': item['shape'],
                      'artifact_id': artifact_id, 'absolute_payload_offset': offset,
                      'activation_columns': t, 'activation_snapshot': str(activation.resolve()),
                      'activation_kind': activation_kind, 'source': str(args.model.resolve()),
                      'oracle': 'v6 linked artifact encoder, core native_weight/unpack, public ops vector_q4_fields versus raw vector_product',
                      'full_root_and_sliced_views': True, 'gpu_executed': True,
                      'summary': summary, 'cases': cases}
            report_path = RESULTS / f'integration-q4-gdn-byteword-{suffix}-summary.json'
            report_path.write_text(json.dumps(record, indent=2) + '\n')
            reports.append(record)
    report = {'gpu_executed': True, 'toolchain_image': IMAGE, 'runs': reports}
    out = RESULTS / 'integration-q4-gdn-byteword-run-summary.json'
    out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('build-cpu-oracle', 'cpu-scan', 'image-build', 'build-probe', 'record-build', 'run'))
    parser.add_argument('--model', type=pathlib.Path, default=MODEL)
    parser.add_argument('--tensor', action='append')
    parser.add_argument('--t', type=int, action='append', choices=(1, 3, 4, 8))
    parser.add_argument('--assert-counts', action='store_true')
    parser.add_argument('--reuse-existing', action='store_true')
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    ACTIVATIONS.mkdir(exist_ok=True)
    if args.action == 'build-cpu-oracle':
        build_cpu_oracle()
    elif args.action == 'cpu-scan':
        cpu_scan(args)
    elif args.action == 'image-build':
        image_build()
    elif args.action == 'build-probe':
        build_probe()
    elif args.action == 'record-build':
        record_build()
    else:
        run(args)


if __name__ == '__main__':
    main()
