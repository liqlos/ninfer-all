import argparse
import hashlib
import json
import pathlib
import struct
import subprocess

from run_q4_fastfields import linux, tensor_directory, MODEL


ROOT = pathlib.Path(__file__).resolve().parent
RESULTS = ROOT / 'results'


def run_command(command, output):
    result = subprocess.run(['wsl.exe', '-d', 'Ubuntu-24.04', '-u', 'root', '--', *map(str, command)],
                            capture_output=True)
    output.write_bytes(result.stdout)
    output.with_suffix(output.suffix + '.stderr').write_bytes(result.stderr)
    return result


def cpu_fixture():
    phases = set()
    receipts = []
    for count in (1, 2, 3, 31, 32, 33, 127, 128, 129, 8192):
        payload_bytes = (count * 13 + 7) // 8
        padded_bytes = ((payload_bytes + 15) & ~15) + 4
        data = bytearray(padded_bytes)
        expected = []
        for block in range(count):
            value = (block * 7919 + count * 17) & 0x1FFF
            expected.append(value)
            bit = block * 13
            for i in range(13):
                mask = 1 << ((bit + i) & 7)
                index = (bit + i) >> 3
                if (value >> i) & 1:
                    data[index] |= mask
                else:
                    data[index] &= ~mask
            phases.add(bit & 31)
        for block, value in enumerate(expected):
            bit = block * 13
            byte = bit >> 3
            shift8 = bit & 7
            triple = int.from_bytes(data[byte:byte + 3], 'little')
            byte_result = (triple >> shift8) & 0x1FFF
            word = bit >> 5
            shift32 = bit & 31
            lo = int.from_bytes(data[word * 4:word * 4 + 4], 'little')
            hi = int.from_bytes(data[(word + 1) * 4:(word + 2) * 4], 'little')
            funnel = lo if shift32 == 0 else (lo >> shift32) | ((hi << (32 - shift32)) & 0xFFFFFFFF)
            always = funnel & 0x1FFF
            conditional = (lo >> shift32) & 0x1FFF if shift32 + 13 <= 32 else funnel & 0x1FFF
            if byte_result != value or always != value or conditional != value:
                raise RuntimeError(f'13-bit decoder mismatch at count={count}, block={block}, phase={shift32}')
        receipts.append({'blocks': count, 'logical_bytes': payload_bytes,
                         'word_stream_bytes_including_16B_alignment_and_4B_tail': padded_bytes,
                         'guarded_tail_bytes': 4})
    if phases != set(range(32)):
        raise RuntimeError(f'not all aligned-word phases covered: {sorted(phases)}')
    return {'cpu_byte_word_conditional_exact': True, 'all_32_word_phases': sorted(phases),
            'cpu_cases': receipts, 'word_extra_tail_bytes_per_stream': 4}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('build', 'selftest', 'scan', 'bench'))
    parser.add_argument('--k', type=int, choices=(5120, 17408), required=True)
    parser.add_argument('--model', type=pathlib.Path, default=MODEL)
    parser.add_argument('--tensor')
    parser.add_argument('--activation', type=pathlib.Path)
    parser.add_argument('--variant', choices=('v1', 'r2'), default='v1')
    parser.add_argument('--tag', default='first')
    args = parser.parse_args()
    if not args.tag or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in args.tag):
        raise ValueError('tag requires lowercase letters, digits or hyphens')
    RESULTS.mkdir(exist_ok=True)
    fixture = cpu_fixture()
    if args.variant == 'r2' and args.k != 5120:
        raise ValueError('r2 mixed d-byte/dmin-word path is limited to K5120')
    stem = f'q4-halfword-k{args.k}' if args.variant == 'v1' else f'q4-halfword-r2-k{args.k}'
    source = ROOT / ('q4_halfword_v1.cu' if args.variant == 'v1' else 'q4_halfword_probe.cu')
    binary = RESULTS / stem
    if args.action == 'build':
        command = ['docker', 'run', '--rm', '-v', f'{linux(ROOT)}:/experiment',
                   '--entrypoint', '/usr/local/cuda/bin/nvcc', 'ninfer-lossless-toolchain:cu13',
                   '-O3', '-DNDEBUG', '-std=c++20', '-arch=sm_86', '-lineinfo', '-DNINFER_SM8X_COMPAT=1',
                   '-Xptxas=-v', f'-DQ4_HALFWORD_K={args.k}',
                   '-I/experiment/source/src', '-I/experiment/source/third_party/ggml-quants/src',
                   '-I/experiment/source/third_party/ggml-quants/include',
                   f'/experiment/{source.name}', '-o', f'/experiment/results/{binary.name}']
        output = RESULTS / f'{stem}-build-{args.tag}.txt'
        if output.exists():
            raise RuntimeError(f'preserve existing receipt: {output}')
        result = run_command(command, output)
        metadata = {'command': command, 'exit_code': result.returncode,
                    'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                    'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest() if binary.exists() else None,
                    'cpu_fixture': fixture}
        output.with_suffix('.txt.metadata.json').write_text(json.dumps(metadata, indent=2))
        if result.returncode:
            raise SystemExit(f'build failed; see {output}.stderr')
        print(json.dumps(metadata, indent=2))
        return
    if not binary.is_file():
        raise SystemExit(f'build {stem} first')
    if args.action == 'selftest':
        output = RESULTS / f'{stem}-selftest-{args.tag}.jsonl'
        if output.exists():
            raise RuntimeError(f'preserve existing receipt: {output}')
        command = ['docker', 'run', '--rm', '-v', f'{linux(ROOT)}:/experiment',
                   '--entrypoint', f'/experiment/results/{binary.name}',
                   'ninfer-lossless-toolchain:cu13', '--cpu-selftest']
    else:
        if not args.tensor:
            raise ValueError('explicit --tensor required')
        if args.action == 'bench' and not args.activation:
            raise ValueError('captured --activation required')
        if args.activation:
            header = struct.unpack('<8I', args.activation.open('rb').read(32))
            expected = header[3] * header[4] + (header[3] // 32) * header[4] * 4
            if (header[0] != 0x31534156 or header[1] != 1 or header[2] != 12 or header[3] != args.k or
                    header[4] != 4 or header[5] not in (0, 1) or header[6] != expected or header[7] != 0):
                raise ValueError('activation snapshot framing mismatch')
        artifact_id, payload, items = tensor_directory(args.model)
        item = items[args.tensor]
        if item['format'] != 'gguf_q4_k' or item['shape'][1] != args.k or item['shape'][0] % 8:
            raise ValueError('tensor format or geometry mismatch')
        command = ['docker', 'run', '--rm']
        if args.action == 'bench':
            command += ['--gpus', 'all']
        command += ['-v', f'{linux(ROOT)}:/experiment', '-v', '/mnt/c/AI/staging:/models:ro',
                    '--entrypoint', f'/experiment/results/{binary.name}', 'ninfer-lossless-toolchain:cu13',
                    '/models/qwen38-jonathan-ninfer/qwen38-jonathan-q4km-proposal.ninfer',
                    str(payload + item['offset']), str(item['shape'][0]), str(args.k), args.tensor,
                    'bench' if args.action == 'bench' else 'scan']
        if args.activation:
            command += [f'/experiment/{args.activation.resolve().relative_to(ROOT).as_posix()}']
        output = RESULTS / f'{stem}-{args.tensor.replace("/", "-")}-{args.action}-{args.tag}.jsonl'
    result = run_command(command, output)
    metadata = {'command': command, 'exit_code': result.returncode,
                'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
                'artifact_id': artifact_id if args.action in ('scan', 'bench') else None,
                'cpu_fixture': fixture}
    output.with_suffix(output.suffix + '.metadata.json').write_text(json.dumps(metadata, indent=2))
    if result.returncode:
        raise SystemExit(f'{args.action} failed; see {output}.stderr')
    print(result.stdout.decode(errors='replace'))


if __name__ == '__main__':
    main()
