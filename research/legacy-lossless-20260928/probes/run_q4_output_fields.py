import argparse
import hashlib
import json
import pathlib
import subprocess

from run_q4_fastfields import linux, tensor_directory, MODEL


ROOT = pathlib.Path(__file__).resolve().parent
RESULTS = ROOT / 'results'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('build', 'scan', 'bench'))
    parser.add_argument('--k', type=int, choices=(6144, 10240, 17408), required=True)
    parser.add_argument('--tensor')
    parser.add_argument('--activation', type=pathlib.Path)
    parser.add_argument('--tag', default='first')
    args = parser.parse_args()
    if not args.tag or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-' for c in args.tag):
        raise ValueError('tag requires lowercase letters, digits or hyphens')
    stem = f'q4-output-fields-k{args.k}'
    source = ROOT / (stem + '.cu')
    binary = RESULTS / stem
    if args.action == 'build':
        base = (ROOT / 'q4_fastfields_probe.cu').read_text()
        head = base[:base.index('int main(int argc, char** argv)')]
        main = base[base.index('int main(int argc, char** argv)'):]
        at = main.index('    if (argc == 8) {\n        const Activation activation')
        main = main[:at] + '''    if (argc != 8) { throw std::runtime_error("captured activation required"); }
    const Activation activation = load_activation(argv[7]);
    if (activation.t != 4 || activation.fused != 0) { throw std::runtime_error("ordinary T4 required"); }
    measure<4>(raw, packed, argv[5], &activation);
    return 0;
}
'''
        source.write_text((head + main).replace('5120', str(args.k)))
        command = ['wsl.exe', '-d', 'Ubuntu-24.04', '-u', 'root', '--', 'docker', 'run', '--rm',
                   '-v', linux(ROOT) + ':/experiment', '--entrypoint', '/usr/local/cuda/bin/nvcc',
                   'ninfer-lossless-toolchain:cu13', '-O3', '-DNDEBUG', '-std=c++20', '-arch=sm_86',
                   '-lineinfo', '-DNINFER_SM8X_COMPAT=1', '-Xptxas=-v',
                   '-I/experiment/source/src', '-I/experiment/source/third_party/ggml-quants/src',
                   '-I/experiment/source/third_party/ggml-quants/include', '/experiment/' + source.name,
                   '-o', '/experiment/results/' + binary.name]
    else:
        if not args.tensor:
            raise ValueError('explicit tensor required')
        _, payload, items = tensor_directory(MODEL)
        item = items[args.tensor]
        if item['format'] != 'gguf_q4_k' or item['shape'][1] != args.k:
            raise ValueError('tensor geometry mismatch')
        if args.action == 'bench' and not args.activation:
            raise ValueError('captured activation required')
        command = ['wsl.exe', '-d', 'Ubuntu-24.04', '-u', 'root', '--', 'docker', 'run', '--rm']
        if args.action == 'bench':
            command += ['--gpus', 'all']
        command += ['-v', linux(ROOT) + ':/experiment', '-v', '/mnt/c/AI/staging:/models:ro',
                    '--entrypoint', '/experiment/results/' + binary.name, 'ninfer-lossless-toolchain:cu13',
                    '/models/qwen38-jonathan-ninfer/' + MODEL.name, str(payload + item['offset']),
                    str(item['shape'][0]), str(args.k), args.tensor, args.action]
        if args.activation:
            command += ['/experiment/' + str(args.activation.resolve().relative_to(ROOT)).replace('\\', '/')]
    path = RESULTS / (stem + '-' + args.action + '-' + args.tag + '.jsonl')
    if path.exists():
        raise RuntimeError('Preserve existing receipt')
    result = subprocess.run(command, capture_output=True)
    path.write_bytes(result.stdout)
    path.with_suffix('.jsonl.stderr').write_bytes(result.stderr)
    metadata = {'command': command, 'exit_code': result.returncode,
                'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'base_source_sha256': hashlib.sha256((ROOT / 'q4_fastfields_probe.cu').read_bytes()).hexdigest(),
                'binary_sha256': hashlib.sha256(binary.read_bytes()).hexdigest() if binary.exists() else None}
    path.with_suffix('.metadata.json').write_text(json.dumps(metadata, indent=2))
    print(result.stdout.decode(errors='replace'))
    if result.returncode:
        raise SystemExit(f'probe exited {result.returncode}; see {path}')


if __name__ == '__main__':
    main()
