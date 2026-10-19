import json
import hashlib
import math
import pathlib
import struct
import subprocess
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
NAME = 'ninfer-lossless-snapshot'
PORT = 18034


def docker(*args, check=True):
    return subprocess.run(['wsl.exe', '-d', 'Ubuntu-24.04', '-u', 'root', '--', 'docker', *args], capture_output=True, text=True, check=check)


def main():
    with urllib.request.urlopen('http://127.0.0.1:8096/v1/status') as response:
        state = json.load(response)
    if state.get('leases') or state.get('agents') or state.get('instances') or state.get('external_growth_pressure', {}).get('active'):
        raise RuntimeError('Snapshot request requires an unloaded idle broker and free GPU')
    folder = RESULTS / 'activations'
    folder.mkdir(exist_ok=True)
    if any(folder.iterdir()):
        raise RuntimeError('Preserve previous activation snapshots before repeating')
    config = json.loads(pathlib.Path(r'C:\AI\gpu-broker\config.json').read_text())
    command = config['container_runtime']['models']['qwen3.8-27b']['command'].copy()
    command[command.index('--request-log-jsonl') + 1] = '/experiment/results/snapshot-server-events.jsonl'
    args = ['run', '-d', '--name', NAME, '--gpus', 'all', '--ipc', 'host', '-p', f'127.0.0.1:{PORT}:8080',
            '-v', '/root/ninfer-stage/model:/model:ro', '-v', '/mnt/c/AI/tuning/ninfer-lossless-20260928:/experiment',
            '-e', 'NINFER_VECTOR_SNAPSHOTS=/experiment/results/activations', '--entrypoint', '/usr/local/bin/ninfer-serve',
            'ninfer-lossless-snapshot:sm86', *command[1:]]
    receipt = {'command': args, 'requests': [], 'diagnostic_only': True, 'config': config['container_runtime']['models']['qwen3.8-27b']}
    print(docker(*args).stdout.strip(), flush=True)
    try:
        for _ in range(180):
            try:
                with urllib.request.urlopen(f'http://127.0.0.1:{PORT}/health', timeout=2) as response:
                    if response.status == 200:
                        break
            except Exception:
                pass
            time.sleep(1)
        else:
            raise RuntimeError('Snapshot server failed to become healthy')
        (folder / 'arm').touch()
        for _ in range(100):
            if (folder / 'armed').exists():
                break
            time.sleep(0.02)
        else:
            raise RuntimeError('Snapshot device collector did not acknowledge arming')
        receipt['armed_after_server_ready'] = True
        for prompt in ['Write a numbered list of 100 everyday actions.', 'Implement a Python LRU cache with a linked list and dictionary. Include tests.']:
            body = {'model': 'qwen3.8-27b', 'messages': [{'role': 'user', 'content': prompt}], 'temperature': 0,
                    'reasoning_effort': 'none', 'max_tokens': 64, 'stream': False}
            request = urllib.request.Request(f'http://127.0.0.1:{PORT}/v1/chat/completions', json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(request, timeout=600) as response:
                output = json.load(response)
            receipt['requests'].append({'settings': body, 'response': output})
            print(output.get('usage'), flush=True)
    finally:
        docker('stop', '-t', '30', NAME, check=False)
        logs = docker('logs', NAME, check=False)
        (RESULTS / 'snapshot-container.log').write_text(logs.stdout + logs.stderr)
        docker('rm', NAME, check=False)
        receipt['snapshots'] = []
        for path in sorted(folder.glob('*.bin')):
            data = path.read_bytes()
            magic, version, qtype, k, columns, fused, size, reserved = struct.unpack('<8I', data[:32])
            if magic != 0x31534156 or version != 1 or size != k * columns * 9 // 8 or len(data) != size + 32 or reserved:
                raise RuntimeError('Snapshot header/size mismatch: ' + str(path))
            if not any(data[32:32 + k * columns]):
                raise RuntimeError('Snapshot contains only zero activations: ' + str(path))
            half_values = [value[0] for value in struct.iter_unpack('<e', data[32 + k * columns:])]
            receipt['snapshots'].append({'path': str(path), 'qtype': qtype, 'K': k, 'T': columns, 'fused': bool(fused), 'bytes': size, 'nonzero': True,
                                        'sha256': hashlib.sha256(data).hexdigest(), 'nonfinite_half_values': sum(not math.isfinite(value) for value in half_values),
                                        'selection': 'first nonzero input per qtype/K/T/fused after API-ready arming; exact layer identity is not recorded'})
        (RESULTS / 'snapshot-requests.json').write_text(json.dumps(receipt, indent=2))
    if not receipt['snapshots']:
        raise RuntimeError('No real nonzero activations were captured')
    print(json.dumps(receipt['snapshots'], indent=2))


if __name__ == '__main__':
    main()
