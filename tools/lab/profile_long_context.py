import copy
import ctypes
import datetime
import hashlib
import json
import pathlib
import shutil
import socket
import subprocess
import time
import urllib.request
import uuid


ROOT = pathlib.Path(__file__).resolve().parents[3]
OUT = ROOT / 'results' / 'profile32k'
IMAGE = 'sha256:0400a382e1b5ce533cd169379afe3c3b99f73c908b8168f9e554bb03616f0c5b'
EXE_SHA = 'ff5effe98b81194b936171aff99cdcef2d5a70ba101f5b9382dfba98d6bde9ad'
MODEL_SHA = '57006def6105eb420606d3cfcf9e929c716076192dbab872c787aa6227f09b03'
INPUT_SHA = '17b4707b03b7121ab4aef6644b190ab8b4b55530b9979595bd50b96f63b2996e'
BROKER = 'http://127.0.0.1:8096'
OWNER = 'ninfer-profile32k'
WSL_ROOT = '/mnt/c/Users/aleksei/Projects/NInfer'
TAG = 'json-r2'


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def run(args, timeout=60, check=True):
    p = subprocess.run(args, capture_output=True, text=True,
                       creationflags=subprocess.CREATE_NO_WINDOW, timeout=timeout)
    if check and p.returncode:
        raise RuntimeError(str(args[:5]) + ': ' + p.stderr[-2500:] + p.stdout[-2500:])
    return p


def docker(*args, timeout=60, check=True):
    return run(['wsl.exe', '-d', 'Ubuntu-24.04', '--exec', '/usr/bin/docker', *args], timeout, check)


def api(path, body=None, method='GET', timeout=30):
    request = urllib.request.Request(BROKER + path,
        json.dumps(body).encode() if body is not None else None,
        {'Content-Type': 'application/json'}, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def admission(reserved=False):
    s = api('/v1/status')
    a = s['admission']
    assert not s['leases'] and not s['agents'] and not s['queue']
    assert not s['inference']['active_requests'] and not s['inference']['active_sequences']
    assert not a['runtime_maintenance']['active'] and not a['active_batches']
    assert not s.get('external_growth_pressure', {}).get('active')
    reservation = a['photo_gpu_reservation']
    assert reservation['active'] == reserved
    if reserved:
        assert reservation['owner'] == OWNER
    else:
        assert not a['draining']
    w = s['windows_gpu']
    assert w['available'] and w['external_reserve_fraction'] == 0.3
    desktop = {'dwm', 'explorer', 'ChatGPT', 'Codex', 'chrome', 'msedge', 'firefox', 'csrss'}
    assert all(p.get('process') in desktop for p in w['process_memory']), w['process_memory']
    capacities = s['capacity_estimate']
    assert len(capacities) == 1
    c = capacities[0]
    assert c['sample_age_seconds'] <= 15 and c['measurement'] == 'measured'
    assert c['external_reserve_satisfied'] and c['available_llm_budget_bytes'] >= 24_000_000_000
    ps = "$m=Get-CimInstance Win32_PerfRawData_PerfOS_Memory; @{free_commit=([uint64]$m.CommitLimit-[uint64]$m.CommittedBytes);comfy=(Get-Service homelab-comfyui).Status.ToString()}|ConvertTo-Json -Compress"
    host = json.loads(run(['powershell.exe', '-NoProfile', '-NonInteractive', '-WindowStyle', 'Hidden', '-Command', ps]).stdout)
    assert host['free_commit'] >= 40 * 2**30 and shutil.disk_usage('C:/').free >= 20 * 2**30
    if host['comfy'] == 'Running':
        with urllib.request.urlopen('http://127.0.0.1:8188/queue', timeout=5) as response:
            q = json.load(response)
        assert not q['queue_running'] and not q['queue_pending']
    ids = docker('ps', '-q').stdout.split()
    for ident in ids:
        item = json.loads(docker('inspect', ident).stdout)[0]
        assert not item['HostConfig'].get('DeviceRequests'), item['Name']
    return {'at': now(), 'budget': c, 'host': host, 'process_memory': w['process_memory']}


def request(body):
    encoded = json.dumps(body, ensure_ascii=False, separators=(',', ':')).encode()
    req = urllib.request.Request('http://127.0.0.1:18042/v1/chat/completions', encoded,
                                 {'Content-Type': 'application/json'})
    start = time.monotonic()
    output, usage, finish, done = '', None, None, False
    with urllib.request.urlopen(req, timeout=300) as response:
        for line in response:
            if not line.startswith(b'data: '):
                continue
            if line[6:].strip() == b'[DONE]':
                done = True
                continue
            event = json.loads(line[6:])
            if event.get('usage'):
                usage = event['usage']
            for choice in event.get('choices', []):
                output += choice.get('delta', {}).get('content') or ''
                finish = choice.get('finish_reason') or finish
    assert done and usage is not None
    return {'request_sha256': hashlib.sha256(encoded).hexdigest(), 'usage': usage,
            'output_sha256': hashlib.sha256(output.encode()).hexdigest(),
            'finish_reason': finish, 'wall_seconds_instrumented': time.monotonic() - start}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    receipt = OUT / (TAG + '-receipt.json')
    assert not receipt.exists() and not list(OUT.glob(TAG + '-nsys*')), 'Existing run must not be repeated'
    k = ctypes.WinDLL('kernel32', use_last_error=True)
    k.CreateMutexW.restype = ctypes.c_void_p
    k.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    k.ReleaseMutex.argtypes = [ctypes.c_void_p]
    k.CloseHandle.argtypes = [ctypes.c_void_p]
    handles, reservation, ident = [], None, None
    record = {'schema': 'ninfer-profile32k-v1', 'started_at': now(), 'diagnostic_only': True,
              'performance_claim': False, 'phases_seconds': {}}
    config_sha = sha(ROOT / 'broker/config.json')
    try:
        for name in ['NInferV17RuntimeScreen', 'NInferV14RuntimeMeasurement',
                     'NInferV15ExportedOperatorSeries', 'NInferV14CpuVariantPreparation',
                     'NInferJonathanSpecShort20261018']:
            h = k.CreateMutexW(None, False, 'Local\\' + name)
            if k.WaitForSingleObject(h, 0) not in (0, 128):
                k.CloseHandle(h)
                raise RuntimeError('Existing laboratory worker owns ' + name)
            handles.append(h)
        record['admission_before'] = admission()
        with socket.socket() as sock:
            assert sock.connect_ex(('127.0.0.1', 18042)) != 0
        source = ROOT / 'results/jonathan-long-screen-r1-inputs.json'
        assert sha(source) == INPUT_SHA
        hash_start = time.monotonic()
        assert sha(ROOT / 'models/dflash/jonathan.ninfer') == MODEL_SHA
        assert sha(ROOT / 'tools/nsys/nsys') == 'ed489f1cdacf354aa6d91201e3f367863a4195153d5aa75a773719235b49315f'
        record['phases_seconds']['model_identity'] = time.monotonic() - hash_start
        manifest = json.loads(source.read_text(encoding='utf-8-sig'))
        measured = next(r for r in manifest['categories'] if r['category'] == 'json')
        warm = copy.deepcopy(next(r for r in manifest['categories'] if r['category'] == 'ru')['body'])
        warm['max_tokens'] = 32
        warm['messages'][-1]['content'] = 'Warmup ' + uuid.uuid4().hex + '\n' + warm['messages'][-1]['content']
        golden = next(json.loads(line) for line in (ROOT / 'results/jonathan-long-screen-r1-mtp3.jsonl').read_text(encoding='utf-8-sig').splitlines() if json.loads(line)['category'] == 'json')
        args = json.loads((ROOT / 'results/jonathan-long-screen-r1-mtp3-identity.json').read_text(encoding='utf-8-sig'))['command']
        args[0] = '/usr/local/bin/ninfer-serve'
        args[-1] = '/results/' + TAG + '-events.jsonl'
        record.update(image=IMAGE, executable_sha256=EXE_SHA, model_sha256=MODEL_SHA,
                      input_manifest_sha256=INPUT_SHA, category='json', command=args,
                      sampling=manifest['sampling_body'], expected_usage=golden['usage'])
        reservation = uuid.uuid4().hex
        created = api('/v1/photo-gpu-reservations', {'owner': OWNER, 'reservation_id': reservation,
                      'ttl_seconds': 1800}, 'POST', timeout=90)
        assert created['reservation_id'] == reservation and created['owner'] == OWNER and created['state'] == 'active'
        record['reservation_id'] = reservation
        record['admission_owned'] = admission(True)
        name = 'ninfer-lab-profile32k-' + reservation[:8]
        command = ['create', '--name', name, '--label', 'io.codex.ninfer-experiment=profile32k',
                   '--label', 'io.codex.ninfer-owner=' + reservation, '--gpus', 'all', '--ipc=host',
                   '-p', '127.0.0.1:18042:8080',
                   '--mount', 'type=bind,source=' + WSL_ROOT + '/models/dflash/jonathan.ninfer,target=/model/jonathan-dflash2.ninfer,readonly',
                   '--mount', 'type=bind,source=' + WSL_ROOT + '/tools/nsys,target=/nsys/target-linux-x64,readonly',
                   '--mount', 'type=bind,source=' + WSL_ROOT + '/results/profile32k,target=/results',
                   '-e', 'CUDA_CACHE_PATH=/tmp/ninfer-jonathan-spec', '--entrypoint', '/bin/sh', IMAGE,
                   '-c', 'ln -s /nsys/target-linux-x64/nsys /tmp/ninfer-nsys; exec /tmp/ninfer-nsys "$@"',
                   'nsys', 'profile', '--trace=cuda,nvtx', '--sample=none', '--cpuctxsw=none',
                   '--cuda-graph-trace=node', '--force-overwrite=false', '-o', '/results/' + TAG + '-nsys', *args]
        t = time.monotonic()
        ident = docker(*command).stdout.strip()
        assert len(ident) == 64
        record['container_id'] = ident
        docker('start', ident)
        for attempt in range(180):
            try:
                with urllib.request.urlopen('http://127.0.0.1:18042/health', timeout=2):
                    break
            except Exception:
                item = json.loads(docker('inspect', ident).stdout)[0]
                assert item['State']['Running'], docker('logs', ident).stdout[-2000:]
                time.sleep(1)
        else:
            raise RuntimeError('Lab server readiness timed out')
        record['phases_seconds']['load_and_startup_instrumented'] = time.monotonic() - t
        proof = docker('exec', ident, 'sha256sum', '/usr/local/bin/ninfer-serve').stdout
        assert proof.split()[0] == EXE_SHA
        print('Profile server ready; warming one distinct 32K prompt.', flush=True)
        record['warmup'] = request(warm)
        print('Warmup complete; profiling frozen 32K JSON request.', flush=True)
        current = api('/v1/status')
        assert not current['leases'] and not current['agents'] and not current['queue']
        assert current['admission']['photo_gpu_reservation']['owner'] == OWNER
        desktop = {'dwm', 'explorer', 'ChatGPT', 'Codex', 'chrome', 'msedge', 'firefox', 'csrss', 'vmmemWSL', 'vmmem'}
        assert all(p.get('process') in desktop for p in current['windows_gpu']['process_memory'])
        result = request(measured['body'])
        record['measured'] = result
        assert result['usage'] == golden['usage']
        assert result['output_sha256'] == golden['output_sha256']
        assert result['finish_reason'] == golden['finish_reason']
        record['equality_gate'] = 'PASS'
    except BaseException as exc:
        record['failure'] = type(exc).__name__ + ': ' + str(exc)
    finally:
        if ident:
            try:
                item = json.loads(docker('inspect', ident).stdout)[0]
                assert item['Id'] == ident and item['Image'] == IMAGE
                assert item['Config']['Labels']['io.codex.ninfer-owner'] == reservation
                shell = 'for p in /proc/[0-9]*; do [ "$(readlink "$p/exe" 2>/dev/null)" = /usr/local/bin/ninfer-serve ] || continue; [ "$(sha256sum "$p/exe" | cut -d" " -f1)" = ' + EXE_SHA + ' ] || exit 3; kill -TERM "${p#/proc/}"; done; true'
                if item['State']['Running']:
                    docker('exec', ident, 'sh', '-c', shell)
                    try:
                        docker('wait', ident, timeout=75)
                    except Exception:
                        docker('stop', '--time', '15', ident, timeout=30)
                log = docker('logs', ident, check=False)
                (OUT / (TAG + '-server.log')).write_text(log.stdout + log.stderr, encoding='utf-8')
                docker('rm', ident)
                record['owned_container_removed'] = True
            except BaseException as exc:
                record['cleanup_failure'] = str(exc)
        if reservation and not record.get('cleanup_failure'):
            try:
                released = api('/v1/photo-gpu-reservations/' + reservation, {'owner': OWNER}, 'DELETE')
                assert released['reservation_id'] == reservation and released['state'] == 'released'
                record['reservation_released'] = True
            except BaseException as exc:
                record['cleanup_failure'] = str(exc)
        for h in reversed(handles):
            k.ReleaseMutex(h)
            k.CloseHandle(h)
        record['production_config_unchanged'] = sha(ROOT / 'broker/config.json') == config_sha
        record['finished_at'] = now()
        record['status'] = 'CAPTURED' if not record.get('failure') and not record.get('cleanup_failure') else 'INCOMPLETE'
        with receipt.open('x', encoding='utf-8') as f:
            json.dump(record, f, indent=2)
        print(json.dumps({k: record.get(k) for k in ['status', 'equality_gate', 'failure', 'cleanup_failure', 'owned_container_removed', 'reservation_released']}, indent=2), flush=True)
    if record['status'] != 'CAPTURED':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
