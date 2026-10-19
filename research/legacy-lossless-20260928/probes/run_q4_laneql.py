import argparse
import hashlib
import json
from pathlib import Path
import struct

from run_q6_regbalance import execute, linux, tensor_directory


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
MODEL = Path('C:/AI/staging/qwen38-jonathan-ninfer/qwen38-jonathan-q4km-proposal.ninfer')
BINARY = RESULTS / 'q4-laneql-probe-cu13-v1'


def source_hashes():
    names = ['q4_laneql_probe.cu', 'q4_halfword_probe.cu', 'q4_fastfields_probe.cu',
             'source/src/ops/linear/gguf/ggml_bridge_vec.cuh',
             'source/src/ops/linear/gguf/ggml_bridge_internal.cuh']
    names += [p.name for p in ROOT.glob('q4_laneql*.cuh')]
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}


def build():
    if BINARY.exists() or BINARY.with_suffix('.build.json').exists():
        raise FileExistsError('Preserving existing build')
    before = source_hashes()
    command = ['docker', 'run', '--rm', '-v', linux(ROOT) + ':/experiment',
               '--entrypoint', '/usr/local/cuda/bin/nvcc', 'ninfer-lossless-toolchain:cu13',
               '-O3', '-DNDEBUG', '-std=c++20', '-arch=sm_86', '-lineinfo',
               '-DNINFER_SM8X_COMPAT=1', '-DQ4_HALFWORD_K=5120', '-Xptxas=-v',
               '-I/experiment/source/src', '-I/experiment/source/third_party/ggml-quants/src',
               '-I/experiment/source/third_party/ggml-quants/include',
               '/experiment/q4_laneql_probe.cu', '-o', '/experiment/results/' + BINARY.name]
    execute(command, RESULTS / 'q4-laneql-build-cu13-v1.txt')
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
    if item['format'] != 'gguf_q4_k' or item['shape'] != [34816, 5120] or item['bytes'] != 100270080:
        raise ValueError('Requires a complete fused MLP Q4 tensor')
    command = [linux(BINARY), linux(args.model), base + item['offset'], 34816, 5120,
               args.tensor, args.action]
    if args.action == 'bench':
        if args.activation is None:
            raise ValueError('Real captured activation required')
        header = struct.unpack('<8I', args.activation.read_bytes()[:32])
        if header[3:6] != (5120, 4, 1):
            raise ValueError('Requires captured K5120 T4 fused input')
        command.append(linux(args.activation))
    output = RESULTS / f'q4-laneql-{args.tensor.replace("/", "-")}-{args.action}-{args.tag}.jsonl'
    result = execute(command, output)
    rows = [json.loads(line) for line in result.stdout.decode().splitlines() if line.strip()]
    if len(rows) != 1:
        raise RuntimeError('Expected one complete probe record')
    gates = ['cpu_full_payload_roundtrip_exact']
    if args.action == 'bench':
        gates += ['gpu_full_payload_reconstruction_exact', 'laneql_fp32_bit_exact',
                  'laneql_bf16_bit_exact', 'laneql_graph_fp32_bit_exact',
                  'laneql_graph_bf16_bit_exact', 'output_canaries_exact']
    if not all(rows[0].get(key) is True for key in gates):
        raise RuntimeError('Explicit candidate correctness gate missing')
    receipt = {'artifact_id': artifact, 'absolute_payload_offset': base + item['offset'],
               'tensor': args.tensor, 'probe_build': metadata, 'records': rows,
               'scope': 'complete fused MLP T4 root; no subview proof',
               'accepted_correctness_gate': True}
    path = RESULTS / f'q4-laneql-{args.action}-{args.tag}-summary.json'
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
