import json
import os
import pathlib
import subprocess
import sys
import time

root = pathlib.Path(__file__).resolve().parents[2]
out = pathlib.Path(__file__).resolve().parent
repo = root / 'source'
prefix = ['wsl.exe', '-d', 'Ubuntu-24.04', '--cd', '/', '--exec', 'docker']

def run(args):
    return subprocess.check_output(args, text=True, creationflags=subprocess.CREATE_NO_WINDOW).strip()

assert run(['git', '-C', str(repo), 'rev-parse', 'HEAD']) == '1e0412299f15f237f2ce5b7b04d8b81de3e1da22'
run(prefix + ['start', 'ninfer-builder'])
changed = set(run(['git', '-C', str(repo), 'diff', '--name-only']).splitlines())
tracked = run(['git', '-C', str(repo), 'ls-files']).splitlines()
stamp = time.time() - 86400
for relative in tracked:
    path = repo / relative
    if relative not in changed and path.is_file():
        os.utime(path, (stamp, stamp))
(out / 'bin').mkdir(exist_ok=True)
cmd = ['g++', '-std=c++17', '-Wall', '-Wextra', '-Werror', '-fsanitize=undefined', '-fno-sanitize-recover=all',
       '-I/lab/source/src', '/lab/source/tests/ops/linear/test_gguf_mmq_limits.cpp', '-o', '/tmp/ninfer-mmq-limits']
record = {'source_base': '1e0412299f15f237f2ce5b7b04d8b81de3e1da22', 'status': 'RUNNING', 'host_compile': cmd}
try:
    record['host_compile_output'] = run(prefix + ['exec', 'ninfer-builder', *cmd])
    record['host_test'] = run(prefix + ['exec', 'ninfer-builder', '/tmp/ninfer-mmq-limits'])
    print(record['host_test'], flush=True)
    subprocess.run([sys.executable, '-B', str(root / 'scripts/build-q6-store-regression.py'), 'bounds'],
                   creationflags=subprocess.CREATE_NO_WINDOW, check=True)
    record['status'] = 'PASS_HOST_UBSAN_AND_OP_BUILD'
finally:
    (out / 'index-bounds-build.json').write_text(json.dumps(record, indent=2) + '\n')
print(record['status'], flush=True)
