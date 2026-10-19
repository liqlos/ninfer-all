import argparse
import json
import pathlib
import re
import struct
import subprocess


ROOT = pathlib.Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
MODEL = pathlib.Path(r'C:\AI\staging\qwen38-jonathan-ninfer\qwen38-jonathan-q4km-proposal.ninfer')
VENDOR = pathlib.Path(r'C:\AI\staging\ninfer-all-pin\third_party\ggml-quants')
IMAGE = 'ninfer-lossless-q6-oracle:sm86'
TENSORS = ('weight/000001', 'weight/000060')


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
    if toolchain != 'cu13':
        raise ValueError('only the CUDA 13 container toolchain is supported')
    binary = RESULTS / 'q4-widepairmeta-probe-cu13'
    includes = ['/experiment/source/src', '/vendor/src', '/vendor/include', '/src/src',
                '/src/third_party/ggml-quants/src', '/src/third_party/ggml-quants/include', '/src/third_party']
    command = ['docker', 'run', '--rm', '-v', linux(ROOT) + ':/experiment',
               '-v', linux(VENDOR) + ':/vendor:ro', '--entrypoint', '/usr/local/cuda/bin/nvcc', IMAGE,
               '-O3', '-DNDEBUG', '-std=c++20', '-arch=sm_86', '-lineinfo', '-DNINFER_SM8X_COMPAT=1', '-Xptxas=-v',
               *['-I' + path for path in includes], '/experiment/q4_widepairmeta_probe.cu',
               '-o', '/experiment/results/q4-widepairmeta-probe-cu13']
    output_path = RESULTS / 'q4-widepairmeta-build-cu13.txt'
    result = execute(command, output_path)
    if result.returncode:
        raise RuntimeError(f'nvcc build failed; see {output_path}.stderr')
    log = output_path.with_suffix(output_path.suffix + '.stderr').read_text(errors='replace')
    entries = {}
    current = None
    for line in log.splitlines():
        if 'Compiling entry function' in line:
            match = re.search(r'scale_kernelILi(\d+)ELb([01])ELb([01])ELb([01])ELb([01])', line)
            if match:
                t, raw, fused, pair, wide = match.groups()
                current = {'T': int(t), 'raw_planar': bool(int(raw)), 'fused': bool(int(fused)),
                           'pair_metadata': bool(int(pair)), 'wide_words': bool(int(wide))}
            else:
                current = None
        elif current is not None and 'Used ' in line and ' registers' in line:
            current['registers_per_thread'] = int(line.split('Used ', 1)[1].split(' registers', 1)[0])
            current['spill_bytes'] = int(line.split('spill stores', 1)[0].rsplit(' ', 1)[1]) if 'spill stores' in line else 0
            entries[f"T{current['T']}-fused{int(current['fused'])}-raw{int(current['raw_planar'])}-pair{int(current['pair_metadata'])}-wide{int(current['wide_words'])}"] = current
            current = None
    report = {'build_succeeded': True, 'gpu_executed': False, 'binary': str(binary.resolve()),
              'toolchain_image': IMAGE, 'vendor_headers': str(VENDOR.resolve()),
              'ptxas_scale_kernel_resources': entries,
              'compiled_wide_candidates': any(row['wide_words'] and row['registers_per_thread'] for row in entries.values())}
    (RESULTS / 'q4-widepairmeta-build-cu13.json').write_text(json.dumps(report, indent=2) + '\n')
    if not report['compiled_wide_candidates']:
        raise RuntimeError('ptxas report did not contain a wide-pair scale kernel')
    print(json.dumps(report, indent=2))


def run(args):
    binary = RESULTS / ('q4-widepairmeta-probe-cu13' if args.toolchain == 'cu13' else 'q4-widepairmeta-probe')
    if not binary.is_file():
        raise SystemExit('run `python run_q4_widepairmeta.py build --toolchain cu13` first')
    artifact_id, payload_base, items = tensor_directory(args.model)
    results = []
    for identity in ((args.tensor,) if args.tensor else TENSORS):
        item = items[identity]
        n, k = item['shape']
        if item['format'] != 'gguf_q4_k' or k != 5120 or n % 8 or item['bytes'] != n * (k // 256) * 144:
            raise ValueError(f'{identity} is not a complete K5120 GGUF Q4_K tensor')
        offset = payload_base + item['offset']
        suffix = 'bench' if args.action == 'bench' else 'scan'
        output_path = RESULTS / f'q4-widepairmeta-{identity.replace("/", "-")}-{suffix}.jsonl'
        if args.action == 'bench':
            if args.activation is None or not args.activation.resolve().is_relative_to(ROOT.resolve()):
                raise ValueError('bench requires a saved activation snapshot inside the task results tree')
            command = ['docker', 'run', '--rm', '--gpus', 'all', '-v', linux(ROOT) + ':/experiment',
                       '-v', linux(args.model) + ':/model.ninfer:ro', '--entrypoint', '/bin/sh', IMAGE,
                       '/experiment/results/q4-widepairmeta-launch.sh', '/model.ninfer', offset, n, k,
                       identity, suffix, '/experiment/' + args.activation.resolve().relative_to(ROOT.resolve()).as_posix()]
        else:
            if args.activation:
                raise ValueError('CPU scan does not accept an activation snapshot')
            command = [linux(binary), linux(args.model), offset, n, k, identity, suffix]
        result = execute(command, output_path)
        lines = [json.loads(line) for line in result.stdout.decode().splitlines() if line.strip()]
        for row in lines:
            row.update({'tensor': identity, 'source': str(args.model.resolve()),
                        'artifact_id': artifact_id, 'absolute_payload_offset': offset})
        if args.action == 'scan':
            if len(lines) != 1 or not lines[0].get('cpu_all_payload_bytes_exact') or not lines[0].get('cpu_wide_pairmeta_full_payload_bytes_exact'):
                raise RuntimeError(f'CPU raw or wide-pair round trip failed for {identity}')
            result_path = RESULTS / f'q4-widepairmeta-{identity.replace("/", "-")}.json'
            result_path.write_text(json.dumps(lines[0], indent=2) + '\n')
        else:
            if len(lines) != (1 if args.activation else 5) or not all(row.get('fp32_output_bit_exact') and row.get('bf16_output_bit_exact')
                                          for row in lines):
                raise RuntimeError(f'GPU arithmetic comparison failed for {identity}')
        results.extend(lines)
    report = {'format': 'q4k-widepairmeta-half-exp-window-v1',
              'encoding': 'per tensor d/dmin exponent base and minimum fixed width; mantissa10 plus optional sign per half; raw half fallback when either field reaches 16 bits',
              'planes': 'Q4 data unchanged; same eight contiguous 12-bit scale/minimum pairs in 12B/block; decoder reads aligned uint32 pairs via __funnelshift_r; wide plane adds 4 tail-padding bytes; separate fixed-stride d and dmin bitstreams',
              'cpu_roundtrip': 'full source Q4_K payload reconstructed exactly using raw, old pairmeta, and wide-word pairmeta decoders before any GPU bench',
              'cpu_full_payload_roundtrip': args.action == 'scan',
              'gpu_run': args.action == 'bench',
              'activation': str(args.activation) if args.activation else None,
              'tensors': results}
    summary_path = RESULTS / f'q4-widepairmeta-{args.action}-summary.json'
    summary_path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('build', 'scan', 'bench'))
    parser.add_argument('--model', type=pathlib.Path, default=MODEL)
    parser.add_argument('--tensor', choices=TENSORS)
    parser.add_argument('--activation', type=pathlib.Path)
    parser.add_argument('--toolchain', choices=('cu12', 'cu13'), default='cu13')
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)
    if args.action == 'build':
        build(args.toolchain)
    else:
        run(args)


if __name__ == '__main__':
    main()

