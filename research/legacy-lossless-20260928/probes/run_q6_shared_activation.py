import argparse
import hashlib
import json
import pathlib
import struct
import subprocess


ROOT = pathlib.Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
MODEL = pathlib.Path(r'C:\AI\staging\qwen38-jonathan-ninfer\qwen38-jonathan-q4km-proposal.ninfer')
PINNED_SOURCE = pathlib.Path(r'C:\AI\staging\ninfer-all-pin')
TENSORS = ('weight/000131',)


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


def execute(args, out_path, required=True):
    result = subprocess.run(['wsl.exe', '-d', 'Ubuntu-24.04', '-u', 'root', '--', *map(str, args)],
                            capture_output=True)
    out_path.write_bytes(result.stdout)
    out_path.with_suffix(out_path.suffix + '.stderr').write_bytes(result.stderr)
    if required and result.returncode:
        raise RuntimeError(f'probe failed with exit {result.returncode}; see {out_path}.stderr')
    return result


def build(toolchain):
    includes = ['src', 'third_party/ggml-quants/src', 'third_party/ggml-quants/include']
    if toolchain == 'cu13':
        command = ['docker', 'run', '--rm', '-v', linux(ROOT) + ':/experiment',
                   '-v', '/mnt/c/AI/staging/ninfer-all-pin:/production:ro',
                   '--entrypoint', '/usr/local/cuda/bin/nvcc', 'ninfer-lossless-toolchain:cu13',
                   '-O3', '-DNDEBUG', '-std=c++20', '-arch=sm_86', '-lineinfo',
                   '-DNINFER_SM8X_COMPAT=1', '-Xptxas=-v',
                   '-I/experiment/source/src', *['-I/production/' + include for include in includes],
                   '/experiment/q6_shared_activation_probe.cu', '-o', '/experiment/results/q6-shared-activation-probe-cu13']
        binary = RESULTS / 'q6-shared-activation-probe-cu13'
        receipt = RESULTS / 'q6-shared-activation-build-cu13.txt'
    else:
        command = ['/root/miniconda3/envs/trellis2/bin/nvcc', '-O3', '-std=c++17', '-arch=sm_86',
                   '-allow-unsupported-compiler', '-Xptxas=-v',
                   *['-I' + linux(PINNED_SOURCE / include) for include in includes],
                   linux(ROOT / 'q6_shared_activation_probe.cu'), '-o', linux(RESULTS / 'q6-shared-activation-probe-cu12')]
        binary = RESULTS / 'q6-shared-activation-probe-cu12'
        receipt = RESULTS / 'q6-shared-activation-build-cu12.txt'
    result = execute(command, receipt)
    if result.returncode:
        raise RuntimeError(f'nvcc build failed; see {receipt}.stderr')
    metadata = {'q6_fields_source_sha256': hashlib.sha256((ROOT / 'source/src/ops/linear/gguf/q6_fields.cuh').read_bytes()).hexdigest(),
                'shared_activation_source_sha256': hashlib.sha256((ROOT / 'q6_shared_activation_fields.cuh').read_bytes()).hexdigest(),
                'probe_source_sha256': hashlib.sha256((ROOT / 'q6_shared_activation_probe.cu').read_bytes()).hexdigest(),
                'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest()}
    binary.with_suffix('.build.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps({'binary': str(binary), 'toolchain': toolchain, 'gpu_executed': False}))


def run(args):
    binary = RESULTS / ('q6-shared-activation-probe-cu13' if args.toolchain == 'cu13' else 'q6-shared-activation-probe-cu12')
    if not binary.is_file():
        raise SystemExit('run `python run_q6_shared_activation.py build` first')
    build_metadata = json.loads(binary.with_suffix('.build.json').read_text())
    if hashlib.sha256(binary.read_bytes()).hexdigest() != build_metadata['binary_sha256']:
        raise RuntimeError('probe binary differs from its build receipt')
    if args.action == 'bench' and (not args.activation or args.t != 1):
        raise ValueError('shared activation head benchmark requires a captured --activation and --t 1')
    artifact_id, payload_base, items = tensor_directory(args.model)
    identities = (args.tensor,) if args.tensor else TENSORS
    all_rows = []
    for identity in identities:
        item = items.get(identity)
        if item is None:
            raise ValueError(f'{identity} is not present in the NInfer directory')
        n, k = item['shape']
        if identity != 'weight/000131' or item['format'] != 'gguf_q6_k' or k != 5120 or n % 2 or item['bytes'] != n * (k // 256) * 210:
            raise ValueError(f'{identity} is not a complete four-block-group Q6_K tensor')
        offset = payload_base + item['offset']
        suffix = args.action if not args.t else f'{args.action}-t{args.t}'
        prefix = 'q6-shared-activation-cu13' if args.toolchain == 'cu13' else 'q6-shared-activation-cu12'
        output_path = RESULTS / f'{prefix}-{identity.replace("/", "-")}-{suffix}.jsonl'
        command = [linux(binary), linux(args.model), offset, n, k, identity, args.action]
        if args.action == 'bench' and (args.activation or args.t):
            command.extend([linux(args.activation) if args.activation else '', args.t or ''])
        result = execute(command, output_path)
        lines = [json.loads(line) for line in result.stdout.decode().splitlines() if line.strip()]
        expected_lines = 1 if args.action == 'scan' or args.t else 8
        if len(lines) != expected_lines:
            raise RuntimeError(f'expected {expected_lines} probe records for {identity}, found {len(lines)}')
        for row in lines:
            row.update({'tensor': identity, 'source': str(args.model.resolve()),
                        'artifact_id': artifact_id, 'absolute_payload_offset': offset})
        if args.action == 'scan' and not all(row.get('raw_planar_cpu_full_roundtrip_exact') and
                                             row.get('transposed_cpu_full_roundtrip_exact') and
                                             row.get('compressed_d_cpu_full_roundtrip_exact') for row in lines):
            raise RuntimeError(f'CPU full-payload round trip failed for {identity}')
        if args.action == 'scan' and not all(row.get('compressed_planar_dwindow_bytes_including_headers', 0) < row['raw_bytes']
                                             for row in lines):
            raise RuntimeError(f'compressed d-window candidate fails full-tensor storage gate for {identity}')
        if args.action == 'scan' and not all(row.get('d_code_width_bits') == 12 and row.get('d_exp_min') == 0 and
                                             row.get('d_exp_bits') == 1 and row.get('d_sign_variable') == 1 for row in lines):
            raise RuntimeError('full head tensor is not strict width12/base0/expbits1/signvar')
        if args.action == 'bench' and not all(row.get('fp32_output_bit_exact') and row.get('bf16_output_bit_exact')
                                              and row.get('gpu_full_payload_reconstruction_exact')
                                              and row.get('post_graph_fp32_bit_exact')
                                              and row.get('head_subview_graph_f32_bf16_exact_q6_fields_dynamic_width12_and_shared_activation')
                                              for row in lines):
            raise RuntimeError(f'GPU arithmetic comparison failed for {identity}')
        all_rows.extend(lines)
    geometry = []
    for row in all_rows:
        geometry.append({'tensor': row['matrix'], 'N': row['N'], 'K': row['K'],
                         'blocks_per_row': row['K'] // 256,
                         'row_bytes': row['K'] // 256 * 210,
                         'raw_bytes': row['N'] * (row['K'] // 256) * 210})
    q6_fields_path = ROOT / 'source' / 'src' / 'ops' / 'linear' / 'gguf' / 'q6_fields.cuh'
    q6_fields_sha256 = build_metadata['q6_fields_source_sha256']
    report = {'format': 'q6k-shared-activation-head-t1-v1',
              'hypothesis': 'stage the unchanged Q8 T1 activation bytes and half2 scales once per CTA in shared memory, then reuse them across the existing grid-stride output row groups; keep Q6 decoder, dot expression, and XOR reduction order unchanged',
              'raw_control': 'production ggml Q6_K raw kernel, dynamic K',
              'current_control': 'current q6_fields runtime dispatch using strict width12/base0/expbits1/sign-variable metadata',
              'candidate': 'private copy of current q6_fields FastWidth12 kernel with cooperative T1 activation copy to shared memory once per CTA and one barrier before its row loop',
              'q6_fields_source_sha256': q6_fields_sha256,
              'q6_fields_current_source_matches_build': hashlib.sha256(q6_fields_path.read_bytes()).hexdigest() == q6_fields_sha256,
              'shared_activation_source_sha256': build_metadata['shared_activation_source_sha256'],
              'shared_activation_source_matches_build': hashlib.sha256((ROOT / 'q6_shared_activation_fields.cuh').read_bytes()).hexdigest() == build_metadata['shared_activation_source_sha256'],
              'probe_build': build_metadata,
              'toolchain': args.toolchain,
              'activation_stage': {'T': 1, 'K': 5120, 'q8_bytes': 5120, 'half2_records': 160, 'half2_bytes': 640,
                                   'shared_bytes_per_cta': 5760, 'copy': 'uint4 Q8 words and half2 records bitwise unchanged'},
              'synthetic_T_coverage': [],
              'payload_layout': {'header_bytes': 32,
                                 'header_u32': ['magic 0x31503651', 'version 1', 'GGML_TYPE_Q6_K', 'N', 'K',
                                                'tile bytes 832', 'd code width bits',
                                                'exp base | exp bits << 5 | sign-variable << 10 | sign-value << 11'],
                                 'planar_offsets': {'ql': 0, 'qh': 512, 'scales': 768},
                                 'd_plane': 'immediately after all row-major 832-byte tiles; flattened row-major blocks, little bit order, d planes not row padded'},
              'cpu_roundtrip': args.action == 'scan',
              'gpu_run': args.action == 'bench',
              'tensor_geometry': geometry,
              'activation': str(args.activation.resolve()) if args.activation else 'synthetic',
              'tensors': all_rows}
    summary_path = RESULTS / f'q6-shared-activation-{args.action}-summary.json'
    summary_path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('build', 'scan', 'bench'))
    parser.add_argument('--model', type=pathlib.Path, default=MODEL)
    parser.add_argument('--tensor')
    parser.add_argument('--activation', type=pathlib.Path)
    parser.add_argument('--t', type=int, choices=(1,))
    parser.add_argument('--toolchain', choices=('cu12', 'cu13'), default='cu13')
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    if args.action == 'build':
        build(args.toolchain)
    else:
        run(args)


if __name__ == '__main__':
    main()
