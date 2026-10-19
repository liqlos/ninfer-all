import ctypes
import json
import pathlib
import sys
import time
import uuid


ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'source/tools/lab'))
import profile_long_context as lab

OUT_SUBDIR = 'ffn-pair'
PROBE_NAME = 'ffn-pair-probe'


def main():
    tag = sys.argv[1] if len(sys.argv) == 2 else 'screen'
    assert tag.isalnum() or tag.replace('-', '').isalnum()
    out = ROOT / 'results' / OUT_SUBDIR
    receipt = out / (tag + '-receipt.json')
    assert not receipt.exists()
    image = 'sha256:501c92a90c3d85adc616455a6783bb5ae8b82008f469cf8dc3d6da538f75475b'
    reservation, ident, handles = None, None, []
    record = {'tag': tag, 'exploratory': tag.startswith('screen'), 'model_performance_claim': False,
              'started_at': lab.now()}
    config = lab.sha(ROOT / 'broker/config.json')
    k = ctypes.WinDLL('kernel32')
    k.CreateMutexW.restype = ctypes.c_void_p
    k.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    k.ReleaseMutex.argtypes = [ctypes.c_void_p]
    k.CloseHandle.argtypes = [ctypes.c_void_p]
    try:
        for name in ['NInferV17RuntimeScreen', 'NInferV14RuntimeMeasurement',
                     'NInferV15ExportedOperatorSeries', 'NInferV14CpuVariantPreparation',
                     'NInferJonathanSpecShort20261018']:
            h = k.CreateMutexW(None, False, 'Local\\' + name)
            if k.WaitForSingleObject(h, 0) not in (0, 128):
                k.CloseHandle(h)
                raise RuntimeError('Laboratory mutex busy: ' + name)
            handles.append(h)
        record['admission'] = lab.admission()
        fixture = ROOT / 'fixtures/operators/q4-full-k5120.bin'
        assert lab.sha(fixture) == 'be10db58b96b5614830c01afe1e28f9d55c4581a6c481827fa7b9d885a6e2790'
        binary = ROOT / 'artifacts/experimental' / PROBE_NAME
        record['binary_sha256'] = lab.sha(binary)
        reservation = uuid.uuid4().hex
        created = lab.api('/v1/photo-gpu-reservations', {'owner': lab.OWNER,
                          'reservation_id': reservation, 'ttl_seconds': 900}, 'POST', 90)
        assert created['reservation_id'] == reservation and created['state'] == 'active'
        record['reservation_id'] = reservation
        record['owned_admission'] = lab.admission(True)
        ident = lab.docker('create', '--name', 'ninfer-lab-ffn-pair-' + reservation[:8],
            '--label', 'io.codex.ninfer-owner=' + reservation, '--gpus', 'all',
            '--mount', 'type=bind,source=' + lab.WSL_ROOT + ',target=/lab,readonly',
            '--entrypoint', '/lab/artifacts/experimental/' + PROBE_NAME, image,
            '/lab/fixtures/operators/q4-full-k5120.bin', tag + '-' + reservation).stdout.strip()
        assert len(ident) == 64
        record['container_id'] = ident
        start = time.monotonic()
        p = lab.docker('start', '-a', ident, timeout=180, check=False)
        (out / (tag + '.jsonl')).write_text(p.stdout, encoding='utf-8')
        record['stderr'] = p.stderr
        state = json.loads(lab.docker('inspect', ident).stdout)[0]['State']
        assert not state['Running']
        record['exit_code'] = state['ExitCode']
        record['wall_seconds'] = time.monotonic() - start
        assert p.returncode == 0 and state['ExitCode'] == 0, p.stderr + p.stdout[-2000:]
        rows = [json.loads(line) for line in p.stdout.splitlines() if line.startswith('{')]
        assert rows[0]['kind'] == 'correctness' and rows[0]['status'] == 'PASS'
        assert rows[-1]['kind'] == 'complete' and rows[-1]['status'] == 'PASS'
        assert len([r for r in rows if r['kind'] == 'timing']) == 3
        record['status'] = 'PASS'
        print(p.stdout, flush=True)
    except BaseException as exc:
        record['status'] = 'INCOMPLETE'
        record['failure'] = type(exc).__name__ + ': ' + str(exc)
    finally:
        try:
            if ident:
                item = json.loads(lab.docker('inspect', ident).stdout)[0]
                assert item['Id'] == ident and item['Image'] == image
                assert item['Config']['Labels']['io.codex.ninfer-owner'] == reservation
                if item['State']['Running']:
                    lab.docker('stop', '--time', '5', ident)
                lab.docker('rm', ident)
                record['owned_container_removed'] = True
            if reservation:
                released = lab.api('/v1/photo-gpu-reservations/' + reservation,
                                   {'owner': lab.OWNER}, 'DELETE')
                assert released['reservation_id'] == reservation and released['state'] == 'released'
                record['reservation_released'] = True
        except BaseException as exc:
            record['cleanup_failure'] = str(exc)
        for h in reversed(handles):
            k.ReleaseMutex(h)
            k.CloseHandle(h)
        record['production_config_unchanged'] = lab.sha(ROOT / 'broker/config.json') == config
        record['finished_at'] = lab.now()
        with receipt.open('x', encoding='utf-8') as f:
            json.dump(record, f, indent=2)
        print(json.dumps({key: record.get(key) for key in
              ['status', 'failure', 'cleanup_failure', 'owned_container_removed', 'reservation_released']}))
    if record['status'] != 'PASS' or record.get('cleanup_failure'):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
