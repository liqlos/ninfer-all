import argparse
import json
import pathlib
import struct
import subprocess


ROOT = pathlib.Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
MODEL = pathlib.Path(r'C:\AI\staging\qwen38-jonathan-ninfer\qwen38-jonathan-q4km-proposal.ninfer')
PINNED_SOURCE = pathlib.Path(r'C:\AI\staging\ninfer-all-pin')
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
                   *['-I/production/' + include for include in includes],
                   '/experiment/q6_vectorload_probe.cu', '-o', '/experiment/results/q6-vectorload-probe-cu13-r4']
        binary = RESULTS / 'q6-vectorload-probe-cu13-r4'
        receipt = RESULTS / 'q6-vectorload-build-cu13-r4.txt'
    else:
        command = ['/root/miniconda3/envs/trellis2/bin/nvcc', '-O3', '-std=c++17', '-arch=sm_86',
                   '-allow-unsupported-compiler', '-Xptxas=-v',
                   *['-I' + linux(PINNED_SOURCE / include) for include in includes],
                   linux(ROOT / 'q6_vectorload_probe.cu'), '-o', linux(RESULTS / 'q6-vectorload-probe-cu12')]
        binary = RESULTS / 'q6-vectorload-probe-cu12'
        receipt = RESULTS / 'q6-vectorload-build-cu12.txt'
    result = execute(command, receipt)
    if result.returncode:
        raise RuntimeError(f'nvcc build failed; see {receipt}.stderr')
    print(json.dumps({'binary': str(binary), 'toolchain': toolchain, 'gpu_executed': False}))


def run(args):
    binary = RESULTS / ('q6-vectorload-probe-cu13-r4' if args.toolchain == 'cu13' else 'q6-vectorload-probe-cu12')
    if not binary.is_file():
        raise SystemExit('run `python run_q6_vectorload.py build` first')
    if args.activation and not args.t:
        raise ValueError('--activation requires --t 1 or --t 4')
    artifact_id, payload_base, items = tensor_directory(args.model)
    identities = (args.tensor,) if args.tensor else TENSORS
    all_rows = []
    for identity in identities:
        item = items.get(identity)
        if item is None:
            raise ValueError(f'{identity} is not present in the NInfer directory')
        n, k = item['shape']
        if item['format'] != 'gguf_q6_k' or k % 1024 or n % 2 or item['bytes'] != n * (k // 256) * 210:
            raise ValueError(f'{identity} is not a complete four-block-group Q6_K tensor')
        offset = payload_base + item['offset']
        suffix = args.action if not args.t else f'{args.action}-t{args.t}'
        prefix = 'q6-vectorload-cu13' if args.toolchain == 'cu13' else 'q6-vectorload-cu12'
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
        if args.action == 'bench' and not all(row.get('fp32_output_bit_exact') and row.get('bf16_output_bit_exact')
                                              and row.get('gpu_full_payload_reconstruction_exact')
                                              for row in lines):
            raise RuntimeError(f'GPU arithmetic comparison failed for {identity}')
        all_rows.extend(lines)
    geometry = []
    for row in all_rows:
        geometry.append({'tensor': row['matrix'], 'N': row['N'], 'K': row['K'],
                         'blocks_per_row': row['K'] // 256,
                         'row_bytes': row['K'] // 256 * 210,
                         'raw_bytes': row['N'] * (row['K'] // 256) * 210})
    report = {'format': 'q6k-planar-vectorload-v1-subview-t1-to-t8',
              'hypothesis': 'load each aligned four-word QL and QH group as uint4, decode that four-word batch immediately, then load the next batch; retain exact Q6 dot/reduction order and compare both raw-planar and unchanged global-d compressed controls',
              'raw_rearranged_control': 'four-block field planes, preserving each original component byte order',
              'candidate': 'aligned 832-byte compressed-plane QL/QH fields loaded as two uint4 batches per plane plus unchanged d bitstream; 840-byte raw planar tiles use scalar fallback because row/tile starts can be misaligned',
              'toolchain': args.toolchain,
              'fast_decoder': 'the aligned compressed-plane candidate changes QL/QH memory operations from sixteen 32-bit loads per 32-value slice to four aligned 128-bit loads in two decode batches; raw-planar 840-byte tile starts fall back to scalar loads; d coding and FP operations are unchanged',
              'synthetic_T_coverage': list(range(1, 9)) if args.action == 'bench' and not args.t and not args.activation else [],
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
    summary_path = RESULTS / f'q6-vectorload-{args.action}-summary.json'
    summary_path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('build', 'scan', 'bench'))
    parser.add_argument('--model', type=pathlib.Path, default=MODEL)
    parser.add_argument('--tensor')
    parser.add_argument('--activation', type=pathlib.Path)
    parser.add_argument('--t', type=int, choices=tuple(range(1, 9)))
    parser.add_argument('--toolchain', choices=('cu12', 'cu13'), default='cu13')
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    if args.action == 'build':
        build(args.toolchain)
    else:
        run(args)


if __name__ == '__main__':
    main()
