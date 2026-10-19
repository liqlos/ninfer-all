import argparse
import json
import pathlib
import subprocess
import struct
import threading
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument('codec', choices=['huffman', 'word_rans', 'scales', 'q4_register', 'q4_fields', 'q4_pairwarp', 'q4_pairmeta', 'q4_fastfields', 'q4_dotilp', 'q6_layout', 'q6_byte_specialized', 'q6_tilelocal', 'q6_vectorload', 'q6_onewarp_probe', 'q6_head_width12', 'q4_output_fields', 'q6_shared_activation', 'q6_shared_activation_v2', 'q4_halfword', 'q6_fast_vector', 'q6_high_lut', 'q6_regbalance', 'q6_partial_unroll', 'q4q6_paired', 'q6_laneplanes', 'q6_lane6', 'q6_lane6_spread', 'q4_cpasync', 'q4_laneql', 'q4_laneql_v2', 'q4_columnwarp', 'q4_fused_v9_op', 'q4_fused_v9r2_op', 'q4_fused_v9r2_buckets', 'q4_fused_v9r3_buckets', 'q4_warp2', 'q4_looprolled', 'q4_looprolled_down', 'q4_laneql_down', 'q4_metadata_shuffle'])
    p.add_argument('--k', type=int)
    p.add_argument('--t', type=int, choices=range(1, 9))
    p.add_argument('--variant', choices=('v1', 'r2'), default='v1')
    p.add_argument('--tensor', default='weight/000060')
    p.add_argument('--activation', type=pathlib.Path)
    p.add_argument('--toolchain', choices=('cu12', 'cu13'), default='cu12')
    p.add_argument('--tag', default='')
    args = p.parse_args()
    if any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in args.tag):
        raise ValueError('tag requires lowercase letters, digits or hyphens')
    label = '-' + args.activation.stem if args.activation else ''
    if args.tag:
        label += '-' + args.tag
    path = ROOT / 'results' / f'measure-{args.codec}-{args.tensor.replace("/", "-")}{label}.json'
    if path.exists():
        raise FileExistsError('Preserving existing GPU measurement receipt')
    with urllib.request.urlopen('http://127.0.0.1:8096/v1/status') as r:
        status = json.load(r)
    if status.get('leases') or status.get('agents') or status.get('instances') or status.get('inference', {}).get('active_requests') or status.get('external_growth_pressure', {}).get('active'):
        raise RuntimeError('GPU ownership is unavailable')
    command = ['python', str(ROOT / ('run_' + args.codec + '.py')), 'run' if args.codec == 'q6_onewarp_probe' else 'bench', '--tensor', args.tensor]
    if args.codec in ('q6_high_lut', 'q6_regbalance', 'q6_partial_unroll', 'q4q6_paired', 'q6_laneplanes', 'q6_lane6', 'q6_lane6_spread', 'q4_cpasync', 'q4_laneql', 'q4_laneql_v2', 'q4_columnwarp', 'q4_fused_v9_op', 'q4_fused_v9r2_op', 'q4_fused_v9r2_buckets', 'q4_fused_v9r3_buckets', 'q4_warp2', 'q4_looprolled', 'q4_looprolled_down', 'q4_laneql_down'):
        command.extend(['--tag', args.tag])
    if args.codec in ('q4_output_fields', 'q4_halfword', 'q4_metadata_shuffle'):
        if not args.k:
            raise ValueError('output fields requires explicit K')
        command.extend(['--k', str(args.k), '--tag', args.tag or 'first'])
        if args.codec == 'q4_halfword':
            command.extend(['--variant', args.variant])
    if args.codec in ('q4_register', 'q4_fields', 'q4_pairwarp', 'q4_pairmeta', 'q4_fastfields', 'q4_dotilp', 'q6_layout', 'q6_byte_specialized', 'q6_tilelocal', 'q6_vectorload'):
        command.extend(['--toolchain', args.toolchain])
    if args.codec in ('q6_head_width12', 'q6_shared_activation', 'q6_shared_activation_v2', 'q6_fast_vector', 'q6_high_lut', 'q6_regbalance', 'q6_partial_unroll', 'q4q6_paired', 'q6_laneplanes', 'q6_lane6', 'q6_lane6_spread', 'q4_cpasync', 'q4_laneql', 'q4_laneql_v2', 'q4_columnwarp', 'q4_fused_v9_op', 'q4_fused_v9r2_op', 'q4_fused_v9r2_buckets', 'q4_fused_v9r3_buckets', 'q4_warp2', 'q4_looprolled', 'q4_looprolled_down', 'q4_laneql_down'):
        command.extend(['--toolchain', args.toolchain])
    if args.codec in ('q4_fused_v9r2_buckets', 'q4_fused_v9r3_buckets'):
        if args.t is None:
            raise ValueError('all-buckets probe requires explicit --t')
        command.extend(['--t', str(args.t)])
    elif args.t is not None:
        raise ValueError('--t is only accepted for all-buckets probe')
    if args.activation:
        command.extend(['--activation', str(args.activation)])
        if args.codec in ('q6_layout', 'q6_byte_specialized', 'q6_tilelocal', 'q6_vectorload', 'q6_head_width12', 'q6_shared_activation', 'q6_shared_activation_v2', 'q6_fast_vector', 'q6_high_lut', 'q6_regbalance', 'q6_partial_unroll', 'q4q6_paired', 'q6_laneplanes', 'q6_lane6', 'q6_lane6_spread', 'q4_cpasync', 'q4_laneql', 'q4_laneql_v2', 'q4_columnwarp', 'q4_fused_v9_op', 'q4_fused_v9r2_op', 'q4_warp2', 'q4_looprolled', 'q4_looprolled_down', 'q4_laneql_down'):
            header = struct.unpack('<8I', args.activation.read_bytes()[:32])
            command.extend(['--t', str(header[4])])
    stop = threading.Event()
    samples = []

    def monitor():
        while not stop.is_set():
            before = time.time_ns()
            r = subprocess.run(['wsl.exe', '-d', 'Ubuntu-24.04', '-u', 'root', '--', 'nvidia-smi',
                                '--query-gpu=timestamp,clocks.sm,clocks.mem,temperature.gpu,power.draw,memory.used,utilization.gpu',
                                '--format=csv,noheader,nounits'], capture_output=True, text=True)
            samples.append({'epoch_ns': before, 'gpu_csv': r.stdout.strip(), 'exit_code': r.returncode})
            stop.wait(0.25)

    observer = threading.Thread(target=monitor)
    observer.start()
    begin = time.time_ns()
    try:
        result = subprocess.run(command, check=False)
    finally:
        stop.set()
        observer.join()
    receipt = {'command': command, 'start_epoch_ns': begin, 'end_epoch_ns': time.time_ns(), 'exit_code': result.returncode,
               'sampling_fields': ['timestamp', 'sm_mhz', 'memory_mhz', 'temperature_c', 'power_w', 'used_memory_mib', 'gpu_util_percent'],
               'samples': samples, 'power_or_clock_controls_changed': False}
    path.write_text(json.dumps(receipt, indent=2))
    if result.returncode == 0 and args.tag and args.codec in ('q4_register', 'q4_fields', 'q4_pairwarp', 'q4_pairmeta', 'q4_fastfields', 'q4_dotilp', 'q6_layout', 'q6_byte_specialized', 'q6_tilelocal', 'q6_vectorload'):
        suffix = '-t' + str(header[4]) if args.codec in ('q6_layout', 'q6_byte_specialized', 'q6_tilelocal', 'q6_vectorload') and args.activation else ''
        stem = (('q6g' if args.toolchain == 'cu13' else 'q6d') if args.codec == 'q6_layout' else 'q4-fastfields-v3' if args.codec == 'q4_fastfields' else 'q6-vectorload-cu13' if args.codec == 'q6_vectorload' and args.toolchain == 'cu13' else 'q6-vectorload-cu12' if args.codec == 'q6_vectorload' else args.codec.replace('_', '-')) + '-' + args.tensor.replace('/', '-')
        source = ROOT / 'results' / (stem + '-bench' + suffix + '.jsonl')
        target = ROOT / 'results' / (stem + label + '.jsonl')
        if target.exists():
            raise RuntimeError('Preserve existing tagged benchmark receipt')
        target.write_bytes(source.read_bytes())
    if result.returncode == 0 and args.tag and args.codec == 'q6_onewarp_probe':
        source = ROOT / 'results' / f'q6-onewarp-{args.tensor.replace("/", "-")}-t4.jsonl'
        target = source.with_name(source.stem + '-' + args.tag + '.jsonl')
        if target.exists():
            raise RuntimeError('Preserve existing tagged benchmark receipt')
        target.write_bytes(source.read_bytes())
    if result.returncode == 0 and args.tag and args.codec in ('q6_head_width12', 'q6_shared_activation', 'q6_shared_activation_v2', 'q6_fast_vector'):
        stem = args.codec.replace('_', '-') + '-' + args.toolchain + '-' + args.tensor.replace('/', '-')
        source = ROOT / 'results' / (stem + '-bench-t' + str(header[4]) + '.jsonl')
        target = ROOT / 'results' / (stem + label + '.jsonl')
        if target.exists():
            raise RuntimeError('Preserve existing tagged benchmark receipt')
        target.write_bytes(source.read_bytes())
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
