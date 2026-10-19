import argparse
import json
import os
import pathlib
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parent
SOURCE = ROOT / 'cupti_trace.cpp'
DEFAULT_CUPTI_ROOT = ROOT / 'results' / 'tooling' / 'cu13' / 'nvidia' / 'cu13'


def build(args):
    root = args.cupti_root.resolve()
    include = root / 'include'
    library = root / 'lib'
    output = args.output.resolve()
    if not (include / 'cupti.h').is_file():
        raise SystemExit(f'CUPTI headers missing: {include}')
    if not any(library.glob('libcupti.so*')):
        raise SystemExit(f'CUPTI library missing: {library}')
    compiler = args.cxx or os.environ.get('CXX', 'g++')
    command = [compiler, '-std=c++17', '-O2', '-fPIC', '-shared', '-pthread', '-fmax-errors=5',
               f'-I{include}', '-I/root/miniconda3/envs/trellis2/include', str(SOURCE), f'-L{library}', f'-Wl,-rpath,{library}',
               '-l:libcupti.so.13', '-o', str(output)]
    result = subprocess.run(command, check=False)
    if result.returncode:
        raise SystemExit(result.returncode)
    version_header = include / 'cupti_version.h'
    api = None
    for line in version_header.read_text(errors='replace').splitlines():
        if line.startswith('#define CUPTI_API_VERSION '):
            api = line.rsplit(' ', 1)[-1]
            break
    print(json.dumps({'library': str(output), 'cupti_root': str(root), 'header_api': api,
                      'cuda_runtime_compatibility': 'not established; confirm the loaded CUPTI version and trace_error on the isolated CUDA 13.1 runtime'}))


def run(args):
    if not args.command:
        raise SystemExit('trace requires a command after --')
    library = args.library.resolve()
    if not library.is_file():
        raise SystemExit(f'profiler library missing: {library}')
    trace_path = args.output.resolve()
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment['LD_PRELOAD'] = str(library) + (':' + environment['LD_PRELOAD'] if environment.get('LD_PRELOAD') else '')
    environment['NINFER_CUPTI_TRACE'] = str(trace_path)
    result = subprocess.run(args.command, env=environment, check=False)
    print(json.dumps({'exit_code': result.returncode, 'trace': str(trace_path), 'command': args.command}))
    raise SystemExit(result.returncode)


def summarize(args):
    counts = {}
    durations = {}
    rows = 0
    errors = []
    with args.trace.open(errors='replace') as stream:
        for line in stream:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                errors.append(line.rstrip())
                continue
            if record.get('type') != 'kernel':
                if record.get('type') in ('trace_error', 'dropped'):
                    errors.append(record)
                continue
            name = record.get('name', '<unknown>')
            counts[name] = counts.get(name, 0) + 1
            durations[name] = durations.get(name, 0) + record.get('duration_ns', 0)
            rows += 1
    top = sorted(counts, key=lambda name: durations[name], reverse=True)
    print(json.dumps({'kernel_records': rows, 'unique_kernels': len(counts),
                      'kernels': [{'name': name, 'launches': counts[name], 'total_ns': durations[name],
                                   'mean_ns': durations[name] / counts[name]} for name in top],
                      'trace_issues': errors}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='action', required=True)
    compile_parser = commands.add_parser('build')
    compile_parser.add_argument('--cupti-root', type=pathlib.Path, default=DEFAULT_CUPTI_ROOT)
    compile_parser.add_argument('--cxx')
    compile_parser.add_argument('--output', type=pathlib.Path, default=ROOT / 'results' / 'libninfer-cupti.so')
    compile_parser.set_defaults(handler=build)
    run_parser = commands.add_parser('trace')
    run_parser.add_argument('--library', type=pathlib.Path, default=ROOT / 'results' / 'libninfer-cupti.so')
    run_parser.add_argument('--output', type=pathlib.Path, default=ROOT / 'results' / 'runtime-kernels.jsonl')
    run_parser.add_argument('command', nargs=argparse.REMAINDER)
    run_parser.set_defaults(handler=run)
    summary_parser = commands.add_parser('summary')
    summary_parser.add_argument('trace', type=pathlib.Path)
    summary_parser.set_defaults(handler=summarize)
    args = parser.parse_args()
    if getattr(args, 'command', None) and args.command[0] == '--':
        args.command = args.command[1:]
    args.handler(args)


if __name__ == '__main__':
    main()
