import argparse
import hashlib
import json
import pathlib
import statistics
import subprocess
import threading
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
CONFIG = pathlib.Path(r'C:\AI\gpu-broker\config.json')
NAME = 'ninfer-lossless-full-ab'
PORT = 18035


def docker(*args, check=True):
    return subprocess.run(['wsl.exe', '-d', 'Ubuntu-24.04', '-u', 'root', '--', 'docker', *args],
                          capture_output=True, text=True, check=check)


def gpu_guard():
    with urllib.request.urlopen('http://127.0.0.1:8096/v1/status', timeout=10) as response:
        state = json.load(response)
    if state.get('leases') or state.get('agents') or state.get('instances') or state.get('inference', {}).get('active_requests') or state.get('external_growth_pressure', {}).get('active'):
        raise RuntimeError('GPU ownership is unavailable')
    return state


def json_request(body):
    request = urllib.request.Request(f'http://127.0.0.1:{PORT}/v1/chat/completions',
                                    json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    started = time.perf_counter()
    first_content = None
    content = ''
    chunks = []
    usage = None
    finish_reasons = []
    done = False
    with urllib.request.urlopen(request, timeout=1800) as response:
        for line in response:
            if not line.startswith(b'data: '):
                continue
            value = line[6:].strip()
            if value == b'[DONE]':
                done = True
                break
            chunk = json.loads(value)
            chunks.append(chunk)
            if chunk.get('usage'):
                usage = chunk['usage']
            for choice in chunk.get('choices', []):
                if choice.get('finish_reason'):
                    finish_reasons.append(choice['finish_reason'])
                text = choice.get('delta', {}).get('content') or ''
                if text and first_content is None:
                    first_content = time.perf_counter() - started
                content += text
    if not done or usage is None or first_content is None:
        raise RuntimeError('Incomplete SSE response or missing usage')
    return {'api_seconds': time.perf_counter() - started, 'api_ttft_seconds': first_content,
            'content': content, 'content_sha256': hashlib.sha256(content.encode()).hexdigest(),
            'usage': usage, 'chunks': chunks, 'finish_reasons': finish_reasons}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--mode', required=True)
    parser.add_argument('--pairs', type=int, default=3)
    parser.add_argument('--tokens', type=int, choices=(512, 1024), default=512)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--validate-device', action='store_true')
    parser.add_argument('--control-off', action='store_true')
    options = parser.parse_args()
    if options.smoke:
        options.pairs = 1
        options.tokens = 64
    if not options.tag or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in options.tag) or (options.pairs < 3 and not options.smoke):
        raise ValueError('Use a unique safe tag and at least three pairs')
    if options.mode == 'off' or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in options.mode):
        raise ValueError('Use a non-off safe candidate mode')
    result_path = RESULTS / f'full-ab-{options.tag}.json'
    if result_path.exists():
        raise RuntimeError('Existing results must be preserved; choose another tag')
    if docker('inspect', NAME, check=False).returncode == 0:
        raise RuntimeError('An experiment container already owns this name')
    config_bytes = CONFIG.read_bytes()
    config = json.loads(config_bytes)
    base_command = config['container_runtime']['models']['qwen3.8-27b']['command'].copy()
    for flag, expected in [('--spec', 'mtp'), ('--draft-tokens', '3'), ('--kv-dtype', 'int8'),
                           ('--max-context', '122880'), ('--kv-capacity', '122880')]:
        if flag not in base_command or base_command[base_command.index(flag) + 1] != expected:
            raise RuntimeError('Authoritative configuration differs from the required baseline')
    production = pathlib.Path(r'C:\AI\staging\ninfer-all-pin\src\ops\linear\gguf')
    code = '\n'.join((production / p).read_text(encoding='utf-8') for p in ('gguf_linear.cpp', 'ggml_bridge_vec.cuh', 'ggml_bridge.cu'))
    prompts = [('short', 'Write a numbered list of 1000 distinct everyday actions. Continue until the output limit.'),
               ('code', 'Implement a complete Python LRU cache with a linked list and dictionary, exhaustive tests, and a detailed walkthrough. Continue until the output limit.'),
               ('long', 'Explain the performance and numerical behavior of this source in detail. Include concrete worked examples. Continue until the output limit.\n' + code)]
    bodies = {label: {'model': 'qwen3.8-27b', 'messages': [{'role': 'system', 'content': 'You are a helpful assistant.'}, {'role': 'user', 'content': prompt}],
                     'temperature': 0, 'seed': 1729, 'reasoning_effort': 'none', 'max_tokens': options.tokens,
                     'stream': True, 'stream_options': {'include_usage': True}} for label, prompt in prompts}
    result = {'config_sha256': hashlib.sha256(config_bytes).hexdigest(), 'image': options.image,
              'image_inspect': json.loads(docker('image', 'inspect', options.image).stdout),
              'baseline_command': base_command, 'candidate_mode': options.mode, 'pairs': options.pairs,
              'both_arms_compression_off': options.control_off,
              'bodies': bodies, 'requests': [], 'telemetry': [], 'power_or_clock_controls_changed': False,
              'cache_protocol': 'Fresh server per arm, identical warmup and scenario order; same command and bodies; prefix hits validated per pair',
              'profiling_enabled': False}
    result['correctness_only'] = options.smoke or options.validate_device or options.control_off

    def save():
        result_path.write_text(json.dumps(result, indent=2), encoding='utf-8')

    save()
    for pair in range(options.pairs):
        for mode in (('off', options.mode) if pair % 2 == 0 else (options.mode, 'off')):
            if CONFIG.read_bytes() != config_bytes:
                raise RuntimeError('Production config changed during comparison')
            result['gpu_before'] = gpu_guard()
            label = f'full-ab-{options.tag}-p{pair}-{mode}'
            event_path = RESULTS / f'{label}-events.jsonl'
            if event_path.exists():
                raise RuntimeError('Existing event receipt would be overwritten')
            command = base_command.copy()
            command[command.index('--request-log-jsonl') + 1] = '/experiment/results/' + event_path.name
            args = ['run', '-d', '--name', NAME, '--gpus', 'all', '--ipc', 'host', '-p', f'127.0.0.1:{PORT}:8080',
                    '-v', '/root/ninfer-stage/model:/model:ro', '-v', '/mnt/c/AI/tuning/ninfer-lossless-20260928:/experiment',
                    '-e', 'NINFER_LOSSLESS_MODE=' + ('off' if options.control_off else mode), '-e', 'NINFER_LOSSLESS_VALIDATE=' + ('1' if options.validate_device else '0'),
                    '--entrypoint', '/usr/local/bin/ninfer-serve', options.image, *command[1:]]
            stop = threading.Event()

            def observe():
                while not stop.is_set():
                    reading = subprocess.run(['wsl.exe', '-d', 'Ubuntu-24.04', '-u', 'root', '--', 'nvidia-smi',
                                              '--query-gpu=timestamp,clocks.sm,clocks.mem,temperature.gpu,power.draw,memory.used,utilization.gpu',
                                              '--format=csv,noheader,nounits'], capture_output=True, text=True)
                    result['telemetry'].append({'arm': label, 'epoch_ns': time.time_ns(), 'csv': reading.stdout.strip(), 'exit_code': reading.returncode})
                    stop.wait(0.5)

            observer = threading.Thread(target=observe)
            try:
                docker(*args)
                observer.start()
                for _ in range(240):
                    running = docker('inspect', '--format', '{{.State.Running}}', NAME, check=False)
                    if running.returncode or running.stdout.strip() != 'true':
                        raise RuntimeError('Experiment server exited before readiness; see container log')
                    try:
                        with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/health', timeout=2) as response:
                            if response.status == 200:
                                break
                    except Exception:
                        pass
                    time.sleep(1)
                else:
                    raise RuntimeError('Server readiness timed out')
                warmup = dict(bodies['short'])
                warmup['messages'] = [{'role': 'user', 'content': 'List several types of plants.'}]
                warmup['max_tokens'] = 64
                json_request(warmup)
                for scenario, _ in prompts:
                    response = json_request(bodies[scenario])
                    if response['usage']['completion_tokens'] != options.tokens or response['finish_reasons'] != ['length']:
                        raise RuntimeError('Requested output-token gate failed')
                    response.update({'pair': pair, 'mode': mode, 'scenario': scenario, 'arm': label,
                                     'body_sha256': hashlib.sha256(json.dumps(bodies[scenario], sort_keys=True).encode()).hexdigest()})
                    result['requests'].append(response)
                    save()
                    print(pair, mode, scenario, response['usage'], response['api_seconds'], flush=True)
                repeat = json_request(bodies['long'])
                repeat.update({'pair': pair, 'mode': mode, 'scenario': 'long-repeat', 'arm': label,
                               'body_sha256': hashlib.sha256(json.dumps(bodies['long'], sort_keys=True).encode()).hexdigest()})
                result['requests'].append(repeat)
                time.sleep(1)
                events = [json.loads(line) for line in event_path.read_text(encoding='utf-8').splitlines() if line.strip()]
                starts = [event for event in events if event.get('event') == 'server_start']
                if len(starts) != 1:
                    raise RuntimeError('Missing server configuration receipt')
                engine = starts[0]['engine']
                if engine['speculative_backend'] != 'mtp' or engine['speculative_draft_window'] != 3 or not engine['cuda_graph'] or not engine['prefix_reuse']:
                    raise RuntimeError('Required runtime configuration gate failed')
                result.setdefault('server_starts', {})[label] = starts[0]
                memory = starts[0]['memory']
                result.setdefault('resident_memory', {})[label] = {
                    'weights_bytes': memory['weights']['capacity_bytes'],
                    'workspace_bytes': memory['workspace']['capacity_bytes'],
                    'sequence_bytes': memory['sequence']['capacity_bytes'],
                    'cuda_graph_allowance_bytes': memory['cuda_graph_allowance_bytes'],
                    'available_after_startup_bytes': memory['available_after_startup_bytes']}
                done = [event for event in events if event.get('event') == 'request_done']
                if len(done) != len(prompts) + 2:
                    raise RuntimeError('Server request receipt count differs')
                for response, event in zip(result['requests'][-len(prompts)-1:], done[1:]):
                    spec = event.get('speculative')
                    if not spec or spec['backend'] != 'mtp' or spec['draft_window'] != 3 or spec['rounds'] <= 0:
                        raise RuntimeError('Missing real MTP3 verification receipt')
                    response['server_event'] = event
                    tokens = event['result']['completion_tokens']
                    response['decode_ms_per_token'] = 1000 * event['timings_seconds']['decode'] / max(tokens - 1, 1)
                    response['decode_tokens_per_second'] = 1000 / response['decode_ms_per_token']
                    response['mtp_acceptance'] = spec['accepted_tokens'] / max(spec['drafted_tokens'], 1)
                    response['accepted_tokens_per_verification'] = spec['accepted_tokens'] / spec['rounds']
                if repeat['server_event']['result']['prefix_cache_hit_tokens'] <= 0:
                    raise RuntimeError('Repeated long request did not exercise prefix reuse')
                save()
            finally:
                stop.set()
                if observer.is_alive():
                    observer.join()
                logs = docker('logs', NAME, check=False)
                (RESULTS / f'{label}-container.log').write_text(logs.stdout + logs.stderr, encoding='utf-8')
                docker('stop', '-t', '30', NAME, check=False)
                docker('rm', NAME, check=False)
                if docker('inspect', NAME, check=False).returncode == 0:
                    raise RuntimeError('Experiment container cleanup failed')
                save()
    comparisons = []
    for pair in range(options.pairs):
        for scenario in [label for label, _ in prompts] + ['long-repeat']:
            a, b = [next(r for r in result['requests'] if r['pair'] == pair and r['scenario'] == scenario and r['mode'] == mode)
                    for mode in ('off', options.mode)]
            exact = a['content_sha256'] == b['content_sha256'] and a['usage'] == b['usage']
            cache_equal = a['server_event']['result']['prefix_cache_hit_tokens'] == b['server_event']['result']['prefix_cache_hit_tokens']
            speculative_equal = a['server_event'].get('speculative') == b['server_event'].get('speculative')
            comparisons.append({'pair': pair, 'scenario': scenario, 'greedy_output_exact': exact,
                                'cache_equal': cache_equal, 'speculative_equal': speculative_equal,
                                'api_latency_reduction': 1 - b['api_seconds'] / a['api_seconds'],
                                'decode_latency_reduction': 1 - b['decode_ms_per_token'] / a['decode_ms_per_token'],
                                'api_ttft_reduction': 1 - b['api_ttft_seconds'] / a['api_ttft_seconds'],
                                'prefill_reduction': 1 - b['server_event']['timings_seconds']['prefill'] / a['server_event']['timings_seconds']['prefill']})
    result['comparisons'] = comparisons
    result['summary'] = {scenario: {metric: statistics.median(r[metric] for r in comparisons if r['scenario'] == scenario)
                                   for metric in ('api_latency_reduction', 'decode_latency_reduction', 'api_ttft_reduction', 'prefill_reduction')}
                         for scenario, _ in prompts}
    result['correctness_gate_passed'] = all(r['greedy_output_exact'] and r['cache_equal'] and r['speculative_equal'] for r in comparisons)
    result['absolute_metrics'] = {
        scenario: {
            mode: {
                metric: {'median': statistics.median(values), 'min': min(values), 'max': max(values)}
                for metric in ('api_seconds', 'api_ttft_seconds', 'decode_ms_per_token',
                               'decode_tokens_per_second', 'mtp_acceptance', 'accepted_tokens_per_verification')
                for values in [[r[metric] for r in result['requests'] if r['scenario'] == scenario and r['mode'] == mode]]}
            for mode in ('off', options.mode)}
        for scenario, _ in prompts}
    result['production_config_unchanged'] = CONFIG.read_bytes() == config_bytes
    result['gpu_after'] = gpu_guard()
    save()
    print(json.dumps(result['summary'], indent=2))
    if not result['correctness_gate_passed']:
        raise RuntimeError('Greedy/cache/MTP equality gate failed; latency is not an accepted result')


if __name__ == '__main__':
    main()
