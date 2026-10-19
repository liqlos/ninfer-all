import argparse
import hashlib
import json
from pathlib import Path
import re
import struct

from run_q6_regbalance import execute, linux, tensor_directory


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
MODEL = Path(r'C:\AI\staging\qwen38-jonathan-ninfer\qwen38-jonathan-q4km-proposal.ninfer')
BINARY = RESULTS / 'q4-looprolled-probe-cu13'
SOURCE = ROOT / 'q4_looprolled_probe.cu'
EXPECTED_BINARY_SHA256 = 'bd7742f7ac2b3276496a62c7aa9ba445417ffe5ec9ca21190ccee4ea091f7e88'
PTXAS_RECORD = {
    'looprolled_laneql_t4': {'registers': 80, 'spill_stores': 0, 'spill_loads': 0},
    'fully_unrolled_laneql_t4': {'registers': 72, 'spill_stores': 0, 'spill_loads': 0},
}
SASS_PATH = RESULTS / 'q4-looprolled-sass.txt'


def source_hashes():
    names = [
        'q4_looprolled_probe.cu',
        'q4_laneql_probe_v2.cu',
        'q4_halfword_probe.cu',
        'q4_fastfields_probe.cu',
        'source/src/ops/linear/gguf/ggml_bridge_vec.cuh',
        'source/src/ops/linear/gguf/ggml_bridge_internal.cuh',
    ]
    names += [p.name for p in ROOT.glob('q4_laneql*.cuh')]
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in names}


def sass_counts():
    if not SASS_PATH.is_file():
        raise FileNotFoundError(f'Missing SASS dump: {SASS_PATH}')
    text = SASS_PATH.read_text(encoding='utf-8', errors='replace')
    symbols = {
        'looprolled_laneql_t4': '_Z19scale_kernel_rolledILi4ELb0ELb1ELi3ELi0ELi3ELi2ELb0ELb1EEv9ScaleArgs',
        'fully_unrolled_laneql_t4': '_Z12scale_kernelILi4ELb0ELb1ELi3ELi0ELi3ELi2ELb0ELb1EEv9ScaleArgs',
    }
    result = {}
    for label, symbol in symbols.items():
        match = re.search(r'Function : ' + re.escape(symbol) + r'\n(.*?)(?=\n\s*Function : |\Z)', text, re.S)
        if match is None:
            raise RuntimeError(f'Missing SASS function {symbol}')
        instructions = re.findall(r'/\*([0-9a-f]{4})\*/\s+([^\n]+)', match.group(1))
        if not instructions:
            raise RuntimeError(f'No SASS instructions found for {symbol}')
        result[label] = {
            'sass_instruction_count': len(instructions),
            'sass_code_bytes': int(instructions[-1][0], 16) + 16,
            'last_instruction_offset_hex': '0x' + instructions[-1][0],
        }
    return result


def build_command():
    return [
        'docker', 'run', '--rm', '-v', linux(ROOT) + ':/experiment',
        '--entrypoint', '/usr/local/cuda/bin/nvcc', 'ninfer-lossless-toolchain:cu13',
        '-O3', '-DNDEBUG', '-std=c++20', '-arch=sm_86', '-lineinfo',
        '-DNINFER_SM8X_COMPAT=1', '-Xptxas=-v',
        '-I/experiment/source/src',
        '-I/experiment/source/third_party/ggml-quants/src',
        '-I/experiment/source/third_party/ggml-quants/include',
        '/experiment/q4_looprolled_probe.cu', '-o', '/experiment/results/' + BINARY.name,
    ]


def build_receipt(build_mode):
    if not BINARY.is_file():
        raise FileNotFoundError(f'Missing frozen binary: {BINARY}')
    digest = hashlib.sha256(BINARY.read_bytes()).hexdigest()
    if digest != EXPECTED_BINARY_SHA256:
        raise RuntimeError('Existing binary hash differs from the previously built CUDA 13.1 binary')
    return {
        'binary': str(BINARY),
        'binary_sha256': digest,
        'source_sha256': source_hashes(),
        'toolchain': 'ninfer-lossless-toolchain:cu13, CUDA 13.1, sm_86',
        'build_command': build_command(),
        'build_mode': build_mode,
        'ptxas_resources': PTXAS_RECORD,
        'sass_comparison': sass_counts(),
        'gpu_executed': False,
    }


def ensure_receipt():
    path = BINARY.with_suffix('.build.json')
    if not BINARY.exists():
        command = build_command()
        build_log = RESULTS / 'q4-looprolled-build-cu13.txt'
        result = execute(command, build_log)
        if result.returncode:
            raise RuntimeError(f'CUDA build failed; see {build_log}.stderr')
        mode = 'built by runner'
    else:
        mode = 'existing binary registered; runner did not rebuild'
    current = build_receipt(mode)
    if path.exists():
        saved = json.loads(path.read_text(encoding='utf-8'))
        if saved.get('binary_sha256') != current['binary_sha256'] or saved.get('source_sha256') != current['source_sha256']:
            raise RuntimeError('Existing build receipt does not match the frozen binary and sources')
        return saved
    path.write_text(json.dumps(current, indent=2) + '\n', encoding='utf-8')
    return current


def action(args):
    if args.action == 'build':
        print(json.dumps(ensure_receipt(), indent=2))
        return
    if args.process != 0:
        raise ValueError('This probe is gated to process 0')
    if not args.tag or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in args.tag):
        raise ValueError('Use a unique safe --tag')
    metadata = ensure_receipt()
    if source_hashes() != metadata['source_sha256']:
        raise RuntimeError('Probe sources differ from the frozen build receipt')
    artifact, base, items = tensor_directory(args.model)
    item = items.get(args.tensor)
    if item is None or item.get('format') != 'gguf_q4_k' or item.get('shape') != [34816, 5120] or item.get('bytes') != 100270080:
        raise ValueError('Requires complete fused MLP Q4 tensor weight/000001 with N34816 K5120')
    command = [linux(BINARY), linux(args.model), str(base + item['offset']), 34816, 5120,
               args.tensor, args.action]
    if args.action == 'bench':
        if args.activation is None:
            raise ValueError('Real captured fused T4 activation required for bench')
        activation_bytes = args.activation.read_bytes()
        if len(activation_bytes) < 32:
            raise ValueError('Activation snapshot header is truncated')
        header = struct.unpack('<8I', activation_bytes[:32])
        if header[3:6] != (5120, 4, 1):
            raise ValueError('Requires process-0 fused MLP snapshot with K5120 T4')
        command.append(linux(args.activation))
    output = RESULTS / f'q4-looprolled-weight-000001-{args.action}-p{args.process}-{args.tag}.jsonl'
    result = execute(command, output)
    rows = [json.loads(line) for line in result.stdout.decode().splitlines() if line.strip()]
    if len(rows) != 1:
        raise RuntimeError('Expected one complete probe record')
    gates = ['cpu_full_payload_roundtrip_exact']
    if args.action == 'scan':
        gates.append('cpu_all_payload_bytes_exact')
    if args.action == 'bench':
        gates += [
            'gpu_full_payload_reconstruction_exact',
            'looprolled_fp32_bit_exact',
            'looprolled_bf16_bit_exact',
            'looprolled_graph_fp32_bit_exact',
            'looprolled_graph_bf16_bit_exact',
            'output_canaries_exact',
        ]
    if not all(rows[0].get(key) is True for key in gates):
        raise RuntimeError('Explicit loop-rolled candidate correctness gate missing')
    receipt = {
        'artifact_id': artifact,
        'absolute_payload_offset': base + item['offset'],
        'tensor': args.tensor,
        'process': args.process,
        'activation': str(args.activation) if args.activation else None,
        'probe_build': metadata,
        'records': rows,
        'scope': 'complete fused MLP T4 root; no subview proof',
        'accepted_correctness_gate': True,
    }
    summary_path = RESULTS / f'q4-looprolled-{args.action}-p{args.process}-{args.tag}-summary.json'
    if summary_path.exists():
        raise FileExistsError(f'Preserving existing summary: {summary_path}')
    summary_path.write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(receipt, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('build', 'scan', 'bench'))
    parser.add_argument('--tensor', default='weight/000001')
    parser.add_argument('--model', type=Path, default=MODEL)
    parser.add_argument('--activation', type=Path)
    parser.add_argument('--tag', default='')
    parser.add_argument('--process', type=int, default=0)
    parser.add_argument('--t4', action='store_true')
    parser.add_argument('--t', type=int, choices=(4,), default=4)
    parser.add_argument('--toolchain', choices=('cu13',), default='cu13')
    args = parser.parse_args()
    if args.action == 'bench' and not (args.t4 or args.t == 4):
        raise ValueError('Requires T4')
    RESULTS.mkdir(exist_ok=True)
    action(args)


if __name__ == '__main__':
    main()
