import argparse
import hashlib
import json
from pathlib import Path
import struct

from run_q6_regbalance import execute, linux, tensor_directory


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
MODEL = Path(r'C:\AI\staging\qwen38-jonathan-ninfer\qwen38-jonathan-q4km-proposal.ninfer')
ACTIVATIONS = RESULTS / 'activations'
BINARY = RESULTS / 'q4-fused-v9r2-buckets-op-probe-cu13-v1'


def source_hashes():
    names = [
        'q4_fused_v9r2_buckets_probe.cu',
        'q4_fused_v9r2_fields.cuh',
        'q4_fused_v9_fields.cuh',
        'q4_fused_v9r2_op_probe.cu',
        'q4_halfword_probe.cu',
        'q4_fastfields_probe.cu',
        'source/src/ops/linear/gguf/ggml_bridge_vec.cuh',
        'source/src/ops/linear/gguf/ggml_bridge_internal.cuh',
    ]
    names += [p.name for p in ROOT.glob('q4_laneql*.cuh')]
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}


def build():
    if BINARY.exists() or BINARY.with_suffix('.build.json').exists():
        raise FileExistsError('Preserving existing build')
    before = source_hashes()
    command = [
        'docker', 'run', '--rm', '-v', linux(ROOT) + ':/experiment',
        '--entrypoint', '/usr/local/cuda/bin/nvcc', 'ninfer-lossless-toolchain:cu13',
        '-O3', '-DNDEBUG', '-std=c++20', '-arch=sm_86', '-lineinfo',
        '-DNINFER_SM8X_COMPAT=1', '-DQ4_HALFWORD_K=5120', '-Xptxas=-v',
        '-I/experiment/source/src',
        '-I/experiment/source/third_party/ggml-quants/src',
        '-I/experiment/source/third_party/ggml-quants/include',
        '/experiment/q4_fused_v9r2_buckets_probe.cu', '-o', '/experiment/results/' + BINARY.name,
    ]
    log = RESULTS / 'q4-fused-v9r2-buckets-build-cu13-v1.txt'
    result = execute(command, log)
    if result.returncode:
        raise RuntimeError(f'CUDA build failed; see {log}.stderr')
    if source_hashes() != before:
        raise RuntimeError('Probe sources changed during build')
    receipt = {
        'command': command,
        'source_sha256': before,
        'binary_sha256': hashlib.sha256(BINARY.read_bytes()).hexdigest(),
        'gpu_executed': False,
    }
    BINARY.with_suffix('.build.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(receipt, indent=2))


def choose_activation(args):
    if args.activation is not None:
        path = args.activation
    else:
        path = ACTIVATIONS / f'q12-k5120-t{args.t}-f1.bin'
        if not path.is_file():
            path = None
    if args.t in (1, 3, 4) and path is None:
        raise FileNotFoundError(f'Real captured T{args.t} activation is required')
    if path is not None:
        header = struct.unpack('<8I', path.read_bytes()[:32])
        if len(path.read_bytes()) < 32 or header[0:3] != (0x31534156, 1, 12) or header[3:6] != (5120, args.t, 1):
            raise ValueError(f'Activation must be a complete captured fused K5120 T{args.t} Q8_1 snapshot')
    return path


def run(args):
    if not args.tag or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in args.tag):
        raise ValueError('Use a unique safe --tag')
    metadata = json.loads(BINARY.with_suffix('.build.json').read_text(encoding='utf-8'))
    if hashlib.sha256(BINARY.read_bytes()).hexdigest() != metadata['binary_sha256']:
        raise RuntimeError('Binary differs from frozen receipt')
    if source_hashes() != metadata['source_sha256']:
        raise RuntimeError('Source differs from frozen receipt')
    artifact, base, items = tensor_directory(args.model)
    item = items.get(args.tensor)
    if item is None or item.get('format') != 'gguf_q4_k' or item.get('shape') != [34816, 5120] or item.get('bytes') != 100270080:
        raise ValueError('Requires complete fused MLP Q4 tensor weight/000001 with N34816 K5120')
    command = [linux(BINARY), linux(args.model), str(base + item['offset']), 34816, 5120,
               args.tensor, args.action, str(args.t)]
    activation = None
    if args.action == 'bench':
        activation = choose_activation(args)
        if activation is not None:
            command.append(linux(activation))
    elif args.activation is not None:
        raise ValueError('Activation is bench-only')
    suffix = f't{args.t}'
    output = RESULTS / f'q4-fused-v9r2-buckets-op-{args.tensor.replace("/", "-")}-{args.action}-{suffix}-{args.tag}.jsonl'
    result = execute(command, output)
    rows = [json.loads(line) for line in result.stdout.decode().splitlines() if line.strip()]
    if len(rows) != 1:
        raise RuntimeError('Expected one complete probe record')
    gates = ['cpu_full_payload_roundtrip_exact']
    if args.action == 'bench':
        gates += [
            'gpu_full_payload_reconstruction_exact',
            'laneql_fp32_bit_exact',
            'laneql_bf16_bit_exact',
            'laneql_graph_fp32_bit_exact',
            'laneql_graph_bf16_bit_exact',
            'fused_v9_fp32_bit_exact',
            'fused_v9_bf16_bit_exact',
            'fused_v9_graph_fp32_bit_exact',
            'fused_v9_graph_bf16_bit_exact',
            'output_canaries_exact',
        ]
        if activation is None and rows[0].get('activation_fixture') is not True:
            raise RuntimeError('Synthetic activation fixture provenance missing')
        if activation is not None and rows[0].get('activation_snapshot') is not True:
            raise RuntimeError('Captured activation provenance missing')
    if not all(rows[0].get(key) is True for key in gates):
        raise RuntimeError('Explicit r2 all-buckets candidate correctness gate missing')
    receipt = {
        'artifact_id': artifact,
        'absolute_payload_offset': base + item['offset'],
        'tensor': args.tensor,
        'T': args.t,
        'activation': str(activation) if activation else 'deterministic Q8_1 fixture',
        'probe_build': metadata,
        'records': rows,
        'scope': f'complete fused MLP T{args.t} root; no subview proof',
        'accepted_correctness_gate': True,
    }
    path = RESULTS / f'q4-fused-v9r2-buckets-op-{args.action}-t{args.t}-{args.tag}-summary.json'
    if path.exists():
        raise FileExistsError('Preserving existing summary')
    path.write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(receipt, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('build', 'scan', 'bench'))
    parser.add_argument('--tensor', default='weight/000001')
    parser.add_argument('--model', type=Path, default=MODEL)
    parser.add_argument('--activation', type=Path)
    parser.add_argument('--tag', default='')
    parser.add_argument('--t', type=int, choices=range(1, 9), default=4)
    parser.add_argument('--toolchain', choices=('cu13',), default='cu13')
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    if args.action == 'build':
        build()
    else:
        run(args)


if __name__ == '__main__':
    main()
