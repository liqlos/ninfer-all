import argparse
import hashlib
import json
from pathlib import Path
import struct

from run_q6_regbalance import execute, linux, tensor_directory


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
MODEL = Path('C:/AI/staging/qwen38-jonathan-ninfer/qwen38-jonathan-q4km-proposal.ninfer')
BINARY = RESULTS / 'q6-lane6-spread-probe-cu13-v1'


def source_hashes():
    names = ['q6_lane6_spread_probe.cu', 'q6_lane6_spread_fields.cuh',
             'q6_regbalance_fields.cuh', 'q6_fast_vector_fields.cuh',
             'source/src/ops/linear/gguf/q6_fields.cuh',
             'source/src/ops/linear/gguf/ggml_bridge_vec.cuh',
             'source/src/ops/linear/gguf/ggml_bridge_internal.cuh']
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}


def build():
    if BINARY.exists() or BINARY.with_suffix('.build.json').exists():
        raise FileExistsError('Preserving existing build')
    before = source_hashes()
    command = ['docker', 'run', '--rm', '-v', linux(ROOT) + ':/experiment',
               '-v', '/mnt/c/AI/staging/ninfer-all-pin:/production:ro',
               '--entrypoint', '/usr/local/cuda/bin/nvcc', 'ninfer-lossless-toolchain:cu13',
               '-O3', '-DNDEBUG', '-std=c++20', '-arch=sm_86', '-lineinfo',
               '-DNINFER_SM8X_COMPAT=1', '-Xptxas=-v', '-I/experiment/source/src',
               '-I/production/src', '-I/production/third_party/ggml-quants/src',
               '-I/production/third_party/ggml-quants/include',
               '/experiment/q6_lane6_spread_probe.cu', '-o', '/experiment/results/' + BINARY.name]
    execute(command, RESULTS / 'q6-lane6-spread-build-cu13-v1.txt')
    if source_hashes() != before:
        raise RuntimeError('Source changed during build')
    receipt = {'command': command, 'source_sha256': before,
               'binary_sha256': hashlib.sha256(BINARY.read_bytes()).hexdigest(),
               'gpu_executed': False}
    BINARY.with_suffix('.build.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(receipt, indent=2))


def run(args):
    if not args.tag or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in args.tag):
        raise ValueError('Use a unique safe --tag')
    metadata = json.loads(BINARY.with_suffix('.build.json').read_text(encoding='utf-8'))
    if hashlib.sha256(BINARY.read_bytes()).hexdigest() != metadata['binary_sha256']:
        raise RuntimeError('Binary differs from frozen receipt')
    if source_hashes() != metadata['source_sha256']:
        raise RuntimeError('Source differs from frozen receipt')
    artifact, base, items = tensor_directory(args.model)
    item = items[args.tensor]
    n, k = item['shape']
    if item['format'] != 'gguf_q6_k' or k != 5120 or n not in (10240, 248320) or item['bytes'] != n * 20 * 210:
        raise ValueError('Requires a complete Q6 GDN or output head K5120 tensor')
    command = [linux(BINARY), linux(args.model), base + item['offset'], n, k, args.tensor, args.action]
    cpu_gate = None
    if args.action == 'bench':
        for cpu_path in sorted(RESULTS.glob('q6-lane6-spread-scan-*-summary.json')):
            cpu_receipt = json.loads(cpu_path.read_text(encoding='utf-8'))
            if (cpu_receipt.get('tensor') == args.tensor and
                cpu_receipt.get('artifact_id') == artifact and
                cpu_receipt.get('absolute_payload_offset') == base + item['offset'] and
                cpu_receipt.get('probe_build', {}).get('binary_sha256') == metadata['binary_sha256'] and
                cpu_receipt.get('accepted_correctness_gate') is True):
                cpu_gate = {'path': str(cpu_path.resolve()),
                            'sha256': hashlib.sha256(cpu_path.read_bytes()).hexdigest()}
                break
        if cpu_gate is None:
            raise RuntimeError('Matching full CPU tensor scan required before GPU benchmark')
        if args.activation is None:
            raise ValueError('Real captured activation required')
        header = struct.unpack('<8I', args.activation.read_bytes()[:32])
        if header[3:6] != (5120, 4, 0):
            raise ValueError('Requires captured K5120 T4 ordinary input')
        command += [linux(args.activation), 4]
    output = RESULTS / f'q6-lane6-spread-{args.tensor.replace("/", "-")}-{args.action}-{args.tag}.jsonl'
    result = execute(command, output)
    rows = [json.loads(line) for line in result.stdout.decode().splitlines() if line.strip()]
    if len(rows) != 1:
        raise RuntimeError('Expected one complete probe record')
    gates = ['lane6_cpu_full_roundtrip_exact']
    if args.action == 'bench':
        gates += ['lane6_gpu_full_payload_reconstruction_exact', 'lane6_fp32_output_bit_exact',
                  'lane6_bf16_output_bit_exact', 'lane6_subview_graph_f32_bf16_exact',
                  'lane6_graph_fp32_bit_exact', 'lane6_graph_bf16_bit_exact', 'output_canaries_exact']
    if not all(rows[0].get(key) is True for key in gates):
        raise RuntimeError('Explicit candidate correctness gate missing')
    if args.action == 'scan':
        descriptor = (rows[0]['d_code_width_bits'], rows[0]['d_exp_min'],
                      rows[0]['d_exp_bits'], rows[0]['d_sign_variable'])
    else:
        header = rows[0]['format_header_u32']
        descriptor = (header[6], header[7] & 31, (header[7] >> 5) & 31,
                      (header[7] >> 10) & 1)
    if descriptor != (12, 0, 1, 1):
        raise RuntimeError('Requires validated fixed-width12 descriptor')
    receipt = {'artifact_id': artifact, 'absolute_payload_offset': base + item['offset'],
               'tensor': args.tensor, 'probe_build': metadata, 'records': rows,
               'scope': 'complete Q6 K5120 T4 root and odd-origin subviews',
               'hypothesis': 'size-neutral per-lane contiguous32x6bit codes; shorter combined operand extraction; unchanged math',
               'cpu_full_tensor_gate_receipt': cpu_gate,
               'accepted_correctness_gate': True}
    path = RESULTS / f'q6-lane6-spread-{args.action}-{args.tag}-summary.json'
    if path.exists():
        raise FileExistsError('Preserving existing summary')
    path.write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(receipt, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('build', 'scan', 'bench'))
    parser.add_argument('--tensor', default='weight/000000')
    parser.add_argument('--model', type=Path, default=MODEL)
    parser.add_argument('--activation', type=Path)
    parser.add_argument('--tag', default='')
    parser.add_argument('--t', type=int, choices=(4,), default=4)
    parser.add_argument('--toolchain', choices=('cu13',), default='cu13')
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    if args.action == 'build':
        build()
    else:
        run(args)


if __name__ == '__main__':
    main()
