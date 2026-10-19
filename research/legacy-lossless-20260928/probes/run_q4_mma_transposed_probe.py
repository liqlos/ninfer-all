import argparse
import hashlib
import json
import pathlib
import struct
import subprocess
import threading
import time
import urllib.request


ROOT = pathlib.Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
MODEL = pathlib.Path(r'C:\AI\staging\qwen38-jonathan-ninfer\qwen38-jonathan-q4km-proposal.ninfer')


def linux(path):
    path = pathlib.Path(path).resolve()
    return '/mnt/' + path.drive[0].lower() + '/' + '/'.join(path.parts[1:])


def build():
    binary = RESULTS / 'q4-mma-transposed-probe-cu13'
    includes = ['source/src', 'source/third_party/ggml-quants/src',
                'source/third_party/ggml-quants/include']
    command = ['docker', 'run', '--rm', '-v', f'{linux(ROOT)}:/experiment',
               '--entrypoint', '/usr/local/cuda/bin/nvcc', 'ninfer-lossless-toolchain:cu13',
               '-O3', '-DNDEBUG', '-std=c++20', '-arch=sm_86', '-lineinfo', '-DNINFER_SM8X_COMPAT=1', '-Xptxas=-v',
               *['-I/experiment/' + include for include in includes],
               '/experiment/q4_mma_transposed_probe.cu', '-o', f'/experiment/results/{binary.name}']
    result = subprocess.run(['wsl.exe', '-d', 'Ubuntu-24.04', '-u', 'root', '--', *command], capture_output=True)
    (RESULTS / 'q4-mma-transposed-probe-cu13-build.txt').write_bytes(result.stdout)
    (RESULTS / 'q4-mma-transposed-probe-cu13-build.txt.stderr').write_bytes(result.stderr)
    if result.returncode:
        raise SystemExit(f'nvcc exited {result.returncode}; see build receipt')
    cpu_command = ['docker', 'run', '--rm', '-v', f'{linux(ROOT)}:/experiment',
                   '--entrypoint', f'/experiment/results/{binary.name}', 'ninfer-lossless-toolchain:cu13', '--cpu-check']
    cpu_result = subprocess.run(['wsl.exe', '-d', 'Ubuntu-24.04', '-u', 'root', '--', *cpu_command], capture_output=True)
    (RESULTS / 'q4-mma-transposed-probe-cpu-check.jsonl').write_bytes(cpu_result.stdout)
    (RESULTS / 'q4-mma-transposed-probe-cpu-check.jsonl.stderr').write_bytes(cpu_result.stderr)
    if cpu_result.returncode:
        raise SystemExit(f'CPU fragment check exited {cpu_result.returncode}; see CPU check receipt')
    cpu_rows = [json.loads(line) for line in cpu_result.stdout.splitlines() if line.strip()]
    if not cpu_rows or not cpu_rows[0].get('cpu_fragment_mapping_exact') or not cpu_rows[0].get('cpu_shared_bank_mapping_exact') or not cpu_rows[0].get('transposed_cpu_ql_roundtrip_exact') or cpu_rows[0].get('old_store_conflict_degree') != 16 or cpu_rows[0].get('padded_store_conflict_degree') != 1:
        raise SystemExit('CPU bank mapping gate failed; see CPU check receipt')
    print(f'built {binary}; no GPU code was run')


def tensor_offset(model, identity='weight/000001'):
    with model.open('rb') as stream:
        magic, length, artifact_id = struct.unpack('<8sQ16s', stream.read(32))
        if magic != b'NINFER\0\3':
            raise ValueError('NInfer v3 required')
        directory = json.loads(stream.read(length))
    payload = (32 + length + 4095) // 4096 * 4096
    item = next(item for item in directory['objects'] if item['id'] == identity)
    if item['format'] != 'gguf_q4_k' or item['shape'] != [34816, 5120]:
        raise ValueError('weight/000001 is not the expected fused Q4_K matrix')
    return payload + item['offset']


def bench(args):
    with urllib.request.urlopen('http://127.0.0.1:8096/v1/status') as response:
        status = json.load(response)
    if status.get('leases') or status.get('agents') or status.get('instances') or status.get('inference', {}).get('active_requests') or status.get('external_growth_pressure', {}).get('active'):
        raise RuntimeError('GPU ownership is unavailable')
    stem = 'q4-mma-transposed-cu13-weight-000001-' + args.tag
    receipt = RESULTS / (stem + '.jsonl')
    if receipt.exists():
        raise RuntimeError('Preserve existing benchmark receipt')
    binary = RESULTS / 'q4-mma-transposed-probe-cu13'
    command = ['wsl.exe', '-d', 'Ubuntu-24.04', '-u', 'root', '--', 'docker', 'run', '--rm', '--gpus', 'all',
               '-v', f'{linux(ROOT)}:/experiment', '-v', '/mnt/c/AI/staging:/models:ro',
               '--entrypoint', '/experiment/results/' + binary.name, 'ninfer-lossless-toolchain:cu13',
               '/models/qwen38-jonathan-ninfer/' + args.model.name, str(tensor_offset(args.model)), '34816',
               '/experiment/' + str(args.activation.resolve().relative_to(ROOT)).replace('\\', '/')]
    stop = threading.Event()
    samples = []

    def monitor():
        while not stop.is_set():
            before = time.time_ns()
            result = subprocess.run(['wsl.exe', '-d', 'Ubuntu-24.04', '-u', 'root', '--', 'nvidia-smi',
                                     '--query-gpu=timestamp,clocks.sm,clocks.mem,temperature.gpu,power.draw,memory.used,utilization.gpu',
                                     '--format=csv,noheader,nounits'], capture_output=True, text=True)
            samples.append({'epoch_ns': before, 'gpu_csv': result.stdout.strip(), 'exit_code': result.returncode})
            stop.wait(0.25)

    metadata = {'command': command, 'source_sha256': hashlib.sha256((ROOT / 'q4_mma_transposed_probe.cu').read_bytes()).hexdigest(),
                'parent_source_sha256': hashlib.sha256((ROOT / 'q4_mma_padded_probe.cu').read_bytes()).hexdigest(),
                'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest(),
                'activation_sha256': hashlib.sha256(args.activation.read_bytes()).hexdigest(),
                'start_epoch_ns': time.time_ns(), 'power_or_clock_controls_changed': False}
    observer = threading.Thread(target=monitor)
    observer.start()
    try:
        result = subprocess.run(command, capture_output=True)
    finally:
        stop.set()
        observer.join()
    receipt.write_bytes(result.stdout)
    receipt.with_suffix('.jsonl.stderr').write_bytes(result.stderr)
    metadata.update(end_epoch_ns=time.time_ns(), exit_code=result.returncode, samples=samples)
    receipt.with_suffix('.telemetry.json').write_text(json.dumps(metadata, indent=2))
    print(result.stdout.decode(errors='replace'))
    if result.returncode:
        raise SystemExit(f'GPU probe exited {result.returncode}; see {receipt}')
    rows = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    required = ('transposed_cpu_full_payload_roundtrip_exact', 'transposed_gpu_full_payload_reconstruction_exact',
                'transposed_fp32_bit_exact', 'transposed_bf16_bit_exact', 'transposed_graph_fp32_bit_exact',
                'transposed_graph_bf16_bit_exact', 'output_canaries_exact')
    if not rows or not all(row.get('fp32_bit_exact') and row.get('bf16_bit_exact') and row.get('graph_fp32_bit_exact') and row.get('gpu_all_payload_bytes_exact') and row.get('cpu_shared_bank_mapping_exact') and row.get('shared_bytes_control') == 16384 and row.get('shared_bytes_padded') == 16896 and row.get('shared_bytes_transposed') == 16896 and all(row.get(flag) for flag in required) for row in rows):
        raise RuntimeError('Missing exactness gate')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('build', 'show-run', 'bench'))
    parser.add_argument('--tag', default='first')
    parser.add_argument('--model', type=pathlib.Path, default=MODEL)
    parser.add_argument('--activation', type=pathlib.Path,
                        default=RESULTS / 'activations' / 'q12-k5120-t4-f1.bin')
    args = parser.parse_args()
    if not args.tag or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in args.tag):
        raise ValueError('tag requires lowercase letters, digits or hyphens')
    RESULTS.mkdir(exist_ok=True)
    if args.action == 'build':
        build()
    elif args.action == 'bench':
        bench(args)
    else:
        binary = RESULTS / 'q4-mma-transposed-probe-cu13'
        print(' '.join(map(str, [linux(binary), linux(args.model), tensor_offset(args.model), 34816, linux(args.activation)])))


if __name__ == '__main__':
    main()

