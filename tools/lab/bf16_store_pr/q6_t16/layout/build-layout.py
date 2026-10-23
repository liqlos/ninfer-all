import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time

out = pathlib.Path(__file__).resolve().parent
root = out.parents[1]
repo = root / 'source'
prefix = ['wsl.exe', '-d', 'Ubuntu-24.04', '--cd', '/', '--exec', 'docker']

def run(args):
    return subprocess.check_output(args, text=True, creationflags=subprocess.CREATE_NO_WINDOW).strip()

head = run(['git', '-C', str(repo), 'rev-parse', 'HEAD'])
assert head == '26003d67c1cfb97123b4c66311c85a9f5ae2e16f'
run(prefix + ['start', 'ninfer-builder'])
changed = set(run(['git', '-C', str(repo), 'diff', '--name-only']).splitlines())
for name in run(['git', '-C', str(repo), 'ls-files']).splitlines():
    p = repo / name
    if name not in changed and p.is_file():
        os.utime(p, (time.time() - 86400, time.time() - 86400))
(out / 'bin').mkdir(exist_ok=True)
record = {'status': 'RUNNING', 'source': head, 'files': {n: hashlib.sha256((repo / n).read_bytes()).hexdigest() for n in changed}}
try:
    record['host_compile'] = run(prefix + ['exec', 'ninfer-builder', 'g++', '-std=c++17', '-Wall', '-Wextra', '-Werror',
        '-fsanitize=undefined', '-fno-sanitize-recover=all', '-I/lab/source/src',
        '/lab/source/tests/ops/linear/test_gguf_mmq_limits.cpp', '-o', '/tmp/ninfer-mmq-limits'])
    record['host_test'] = run(prefix + ['exec', 'ninfer-builder', '/tmp/ninfer-mmq-limits'])
    print(record['host_test'], flush=True)
    subprocess.run([sys.executable, '-B', str(root / 'scripts/build-q6-store-regression.py'), 'layout'], check=True,
                   creationflags=subprocess.CREATE_NO_WINDOW)
    record['cmake_cpu_target'] = run(prefix + ['exec', 'ninfer-builder', 'cmake', '--build', '/build', '--target',
        'ninfer_linear_gguf_mmq_limits_test', 'ninfer_pr_check', '-j', '4'])
    assert all(hashlib.sha256((repo / n).read_bytes()).hexdigest() == sha for n, sha in record['files'].items())
    record['status'] = 'PASS_HOST_UBSAN_AND_OP_BUILD'
finally:
    (out / 'layout-host-build.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
print(record['status'], flush=True)
