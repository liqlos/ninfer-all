import argparse
import hashlib
import json
import pathlib
import statistics
import subprocess
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent
BROKER = 'http://127.0.0.1:8096'


def api(path, body=None, method=None, timeout=1200):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(BROKER + path, data=data, method=method, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def telemetry():
    r = subprocess.run(['wsl.exe', '-d', 'Ubuntu-24.04', '--', 'nvidia-smi', '--query-gpu=timestamp,clocks.sm,clocks.mem,temperature.gpu,power.draw,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    return r.stdout.strip()


def request(lease, prompt, nonce, limit):
    body = {'model': 'qwen3.8-27b', 'messages': [{'role': 'system', 'content': f'Experiment run {nonce}. Follow the user request.'}, {'role': 'user', 'content': prompt}], 'max_tokens': limit, 'temperature': 0, 'reasoning_effort': 'none', 'stream': True, 'stream_options': {'include_usage': True}}
    result = {'settings': body, 'telemetry_before': telemetry()}
    req = urllib.request.Request(BROKER + f'/v1/leases/{lease}/chat/completions', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    pieces = []
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as response:
        for raw in response:
            line = raw.decode().strip()
            if line == 'data: [DONE]':
                result['done'] = True
            elif line.startswith('data:'):
                event = json.loads(line[5:])
                if event.get('error'):
                    raise RuntimeError(event['error'])
                choices = event.get('choices', [])
                if choices:
                    content = choices[0].get('delta', {}).get('content')
                    if content:
                        result.setdefault('ttft_seconds', time.perf_counter() - start)
                        pieces.append(content)
                    if choices[0].get('finish_reason'):
                        result['finish_reason'] = choices[0]['finish_reason']
                if event.get('usage'):
                    result['usage'] = event['usage']
                if event.get('timings'):
                    result['timings'] = event['timings']
    result['api_seconds'] = time.perf_counter() - start
    result['output_sha256'] = hashlib.sha256(''.join(pieces).encode()).hexdigest()
    result['output_prefix'] = ''.join(pieces)[:160]
    intervals = max(result['usage']['completion_tokens'] - 1, 0)
    result['decode_seconds_client'] = result['api_seconds'] - result['ttft_seconds']
    result['decode_tok_s_client'] = intervals / result['decode_seconds_client']
    result['telemetry_after'] = telemetry()
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repeats', type=int, default=3)
    args = p.parse_args()
    before = api('/v1/status')
    if before.get('leases') or before.get('agents') or before.get('external_growth_pressure', {}).get('active'):
        raise RuntimeError('GPU is not free')
    lease = api('/v1/leases', {'owner': 'lossless baseline measurement', 'kind': 'agent', 'model_name': 'qwen3.8-27b', 'context_length': 32768, 'ttl_seconds': 1800})['lease_id']
    source = ROOT / 'source/src/ops/linear/gguf'
    code = '\n'.join((source / name).read_text() for name in ['gguf_linear.cpp', 'ggml_bridge_vec.cuh', 'ggml_bridge.cu'])
    prompts = {
        'short': 'Write a numbered list of 100 practical everyday actions, one short sentence per item. Continue the list without summarizing.',
        'code': 'Implement a Python LRU cache with a doubly linked list and dictionary. Include the complete implementation, unit tests, and explanations of eviction and edge cases. Be thorough.',
        'long_code': 'Analyze the C++/CUDA source below and explain the small-T GGUF route, quantization, fused SwiGLU, numerical behavior, workspace and performance. Be thorough and give concrete recommendations.\n' + code,
    }
    result = {'scope': 'untouched_production_baseline_only', 'mtp': '3', 'config_sha256': hashlib.sha256(pathlib.Path(r'C:\AI\gpu-broker\config.json').read_bytes()).hexdigest(), 'before': before, 'runs': []}
    try:
        result['warmup'] = request(lease, 'Write the numbers one to twenty.', 'warmup', 64)
        result['loaded_status'] = api('/v1/status')
        for r in range(args.repeats):
            for label, prompt in prompts.items():
                run = request(lease, prompt, f'{label}-{r}', 512)
                run['label'] = label
                run['repeat'] = r
                result['runs'].append(run)
                (ROOT / 'results/baseline.json').write_text(json.dumps(result, indent=2))
                print(json.dumps({k: run[k] for k in ['label', 'repeat', 'usage', 'ttft_seconds', 'api_seconds', 'decode_tok_s_client'] }), flush=True)
    finally:
        api(f'/v1/leases/{lease}', method='DELETE')
        result['after'] = api('/v1/status')
        result['summary'] = {label: {key: {'median': statistics.median(run[key] for run in result['runs'] if run['label'] == label), 'min': min(run[key] for run in result['runs'] if run['label'] == label), 'max': max(run[key] for run in result['runs'] if run['label'] == label)} for key in ['ttft_seconds', 'api_seconds', 'decode_tok_s_client']} for label in prompts if any(run['label'] == label for run in result['runs'])}
        (ROOT / 'results/baseline.json').write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
