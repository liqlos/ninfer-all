import argparse
import concurrent.futures
import json
import pathlib
import subprocess
import threading
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
NAME = 'ninfer-lossless-profile'
PORT = 18033


def docker(*args, check=True):
    result = subprocess.run(['wsl.exe', '-d', 'Ubuntu-24.04', '-u', 'root', '--', 'docker', *args], capture_output=True, text=True)
    if check and result.returncode:
        raise RuntimeError(f'docker {args[0]} exited {result.returncode}: {result.stderr.strip()}')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', default='cupti13')
    parser.add_argument('--image', default='ninfer-codex-20260927:sm86')
    parser.add_argument('--serve-binary')
    parser.add_argument('--mode', choices=('off', 'q6-fields', 'q4q6-fields'), default='off')
    parser.add_argument('--concurrency', type=int, choices=(1, 2, 4), default=1)
    parser.add_argument('--workload', choices=('mixed', 'short'), default='mixed')
    parser.add_argument('--warmup-batch', action='store_true')
    parser.add_argument('--max-tokens', type=int, default=64)
    parser.add_argument('--seed-base', type=int)
    parser.add_argument('--no-trace', action='store_true')
    parser.add_argument('--matrix-snapshots', action='store_true')
    parser.add_argument('--matrix-snapshot-instrumented', action='store_true')
    parser.add_argument('--matrix-snapshot-library', default='libninfer-matrix-snapshot.so')
    options = parser.parse_args()
    if options.matrix_snapshot_instrumented:
        options.matrix_snapshots = True
    if options.no_trace and options.matrix_snapshots:
        raise ValueError('matrix snapshots require tracing')
    if not options.tag or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in options.tag):
        raise ValueError('tag requires lowercase letters, digits or hyphens')
    prefix = 'profile-' + options.tag
    with urllib.request.urlopen('http://127.0.0.1:8096/v1/status') as r:
        state = json.load(r)
    if state.get('leases') or state.get('agents') or state.get('inference', {}).get('active_requests'):
        raise RuntimeError('GPU belongs to another request')
    if state.get('external_growth_pressure', {}).get('active') or state.get('instances'):
        raise RuntimeError('Isolated profiling requires an unloaded broker and no external GPU pressure')
    config = json.loads(pathlib.Path(r'C:\AI\gpu-broker\config.json').read_text())
    command = config['container_runtime']['models']['qwen3.8-27b']['command'].copy()
    command[command.index('--request-log-jsonl') + 1] = '/experiment/results/' + prefix + '-server-events.jsonl'
    trace = RESULTS / ('runtime-kernels-' + options.tag + '.jsonl')
    if trace.exists() and not options.no_trace:
        raise RuntimeError('Trace already exists; preserve previous receipt')
    snapshot_folder = RESULTS / ('matrix-activations-' + options.tag)
    if options.matrix_snapshots:
        if snapshot_folder.exists():
            raise RuntimeError('Snapshot folder already exists; preserve previous receipt')
        snapshot_folder.mkdir()
    preload = '/experiment/results/' + options.matrix_snapshot_library if options.matrix_snapshots else '/experiment/results/libninfer-cupti13.so'
    preload_env = '' if options.no_trace or options.matrix_snapshot_instrumented else '/experiment/results/tooling/cu13/nvidia/cu13/lib/libcupti.so.13:' + preload
    args = ['run', '-d', '--name', NAME, '--gpus', 'all', '--ipc', 'host', '-p', f'127.0.0.1:{PORT}:8080',
            '-v', '/root/ninfer-stage/model:/model:ro',
            '-v', '/mnt/c/AI/tuning/ninfer-lossless-20260928:/experiment',
            '-e', 'LD_PRELOAD=' + preload_env,
            '-e', 'NINFER_CUPTI_EXECUTABLE=ninfer-serve',
            '-e', 'NINFER_CUPTI_TRACE=/experiment/results/' + trace.name,
            '-e', 'NINFER_LOSSLESS_MODE=' + options.mode,
            '-e', 'NINFER_MATRIX_SNAPSHOTS=' + ('/experiment/results/' + snapshot_folder.name if options.matrix_snapshots else ''),
            '--entrypoint', '/usr/local/bin/ninfer-serve',
            options.image, *command[1:]]
    if options.serve_binary:
        args[args.index('--entrypoint'):args.index('--entrypoint')] = [
            '--mount', 'type=bind,source=' + options.serve_binary +
            ',target=/usr/local/bin/ninfer-serve,readonly']
    if options.matrix_snapshot_instrumented:
        args[args.index('--entrypoint'):args.index('--entrypoint')] = [
            '-e', 'NINFER_MATRIX_ACTIVATION_SNAPSHOTS=/experiment/results/' + snapshot_folder.name]
    result = {'command': args, 'requests': [], 'profiling_only': not options.no_trace, 'concurrency': options.concurrency, 'workload': options.workload, 'config': config['container_runtime']['models']['qwen3.8-27b']}
    result['matrix_input_capture'] = options.matrix_snapshots
    result['matrix_capture_method'] = 'prequantize GPU copy' if options.matrix_snapshot_instrumented else 'CUPTI postlaunch wrapper' if options.matrix_snapshots else None
    print(docker(*args).stdout.strip(), flush=True)
    try:
        for _ in range(180):
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/health', timeout=2) as r:
                    if r.status == 200:
                        break
            except Exception:
                pass
            time.sleep(1)
        else:
            raise RuntimeError('Isolated profiler server did not become ready')
        print('profile server ready', flush=True)
        if options.matrix_snapshots:
            if options.matrix_snapshot_instrumented and not (snapshot_folder / 'ready').exists():
                raise RuntimeError('Instrumented matrix snapshot did not initialize before readiness')
            (snapshot_folder / 'arm').write_text('readiness confirmed; capture user-request inputs\n')
            if options.matrix_snapshot_instrumented:
                for _ in range(100):
                    if (snapshot_folder / 'armed').exists():
                        break
                    time.sleep(0.1)
                else:
                    raise RuntimeError('Instrumented matrix snapshot did not acknowledge arming')
        code = '\n'.join((ROOT / 'source/src/ops/linear/gguf' / p).read_text() for p in ['gguf_linear.cpp', 'ggml_bridge_vec.cuh', 'ggml_bridge.cu'])
        prompts = [('short', 'Write a numbered list of 100 everyday actions.'),
                   ('code', 'Implement a Python LRU cache with a linked list and dictionary. Include tests.'),
                   ('long', 'Explain the performance and numerical behavior of this source:\n' + code)]
        if options.workload == 'short':
            chosen = [('actions', 'Write a numbered list of distinct everyday actions until the output limit.'),
                      ('code', 'Implement a Python LRU cache with a linked list and dictionary, then explain it.'),
                      ('history', 'Explain the history of the telescope in chronological order with examples.'),
                      ('math', 'Explain how to solve quadratic equations and give several worked examples.')][:options.concurrency]
        else:
            chosen = (prompts * 2)[:options.concurrency]
        barrier = threading.Barrier(len(chosen))

        def profile_one(item, max_tokens):
            label, prompt = item
            barrier.wait()
            before = docker('exec', '-e', 'LD_PRELOAD=', '-e', 'NINFER_CUPTI_TRACE=', NAME, 'date', '+%s%N').stdout.strip()
            body = {'model': 'qwen3.8-27b', 'messages': [{'role': 'system', 'content': 'Profile ' + label}, {'role': 'user', 'content': prompt}],
                    'temperature': 0, 'reasoning_effort': 'none', 'max_tokens': max_tokens, 'stream': False}
            if options.seed_base is not None:
                body['seed'] = options.seed_base + [item[0] for item in chosen].index(label)
            req = urllib.request.Request(f'http://127.0.0.1:{PORT}/v1/chat/completions', json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=600) as r:
                response = json.load(r)
            after = docker('exec', '-e', 'LD_PRELOAD=', '-e', 'NINFER_CUPTI_TRACE=', NAME, 'date', '+%s%N').stdout.strip()
            return {'label': label, 'settings': body, 'response': response, 'host_start_epoch_ns': before, 'host_end_epoch_ns': after}

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(chosen)) as pool:
            if options.warmup_batch:
                result['warmup'] = list(pool.map(lambda item: profile_one(item, 16), chosen))
            result['measurement_start_epoch_ns'] = time.time_ns()
            result['requests'] = list(pool.map(lambda item: profile_one(item, options.max_tokens), chosen))
            result['measurement_end_epoch_ns'] = time.time_ns()
        for entry in result['requests']:
            (RESULTS / (prefix + '-requests.json')).write_text(json.dumps(result, indent=2))
            print(entry['label'], entry['response'].get('usage'), flush=True)
    finally:
        (RESULTS / (prefix + '-container.log')).write_text(docker('logs', NAME, check=False).stdout + docker('logs', NAME, check=False).stderr)
        docker('stop', '-t', '30', NAME, check=False)
        docker('rm', NAME, check=False)
        (RESULTS / (prefix + '-requests.json')).write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
