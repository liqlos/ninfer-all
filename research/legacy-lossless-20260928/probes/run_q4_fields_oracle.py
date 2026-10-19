import json
import pathlib
import struct
import subprocess


ROOT = pathlib.Path(__file__).resolve().parent
SOURCE = ROOT / 'source'
RESULTS = ROOT / 'results'
MODEL = pathlib.Path(r'C:\AI\staging\qwen38-jonathan-ninfer\qwen38-jonathan-q4km-proposal.ninfer')
IDENTITY = 'weight/000060'


def linux(path):
    path = pathlib.Path(path).resolve()
    return '/mnt/' + path.drive[0].lower() + '/' + '/'.join(path.parts[1:])


def execute(command, output_path):
    result = subprocess.run(['wsl.exe', '-d', 'Ubuntu-24.04', '-u', 'root', '--', *map(str, command)],
                            capture_output=True)
    output_path.write_bytes(result.stdout)
    output_path.with_suffix(output_path.suffix + '.stderr').write_bytes(result.stderr)
    if result.returncode:
        raise RuntimeError(f'command failed with exit {result.returncode}; see {output_path}.stderr')
    return result.stdout


def artifact_tensor(model):
    with model.open('rb') as stream:
        magic, length, artifact_id = struct.unpack('<8sQ16s', stream.read(32))
        if magic != b'NINFER\0\3':
            raise ValueError('NInfer v3 artifact required')
        directory = json.loads(stream.read(length))
    payload = (32 + length + 4095) // 4096 * 4096
    items = {item['id']: item for item in directory['objects']}
    return artifact_id.hex(), payload, items[IDENTITY]


def main():
    RESULTS.mkdir(exist_ok=True)
    artifact_id, payload_base, item = artifact_tensor(MODEL)
    rows, columns = item['shape']
    if item['format'] != 'gguf_q4_k' or rows != 16384 or columns != 5120:
        raise ValueError(f'{IDENTITY} no longer matches the Q4 fields candidate filter')
    binary = RESULTS / 'q4-fields-oracle'
    include_src = linux(SOURCE / 'src')
    include_third_party = linux(SOURCE / 'third_party')
    command = ['/usr/bin/g++', '-O3', '-DNDEBUG', '-std=c++20', '-I' + include_src,
               '-I' + include_third_party,
               linux(ROOT / 'q4_fields_oracle.cpp'),
               linux(SOURCE / 'src/core/weight_view.cpp'),
               linux(SOURCE / 'src/core/tensor.cpp'),
               linux(SOURCE / 'src/core/dtype.cpp'),
               linux(SOURCE / 'src/artifact/q4_fields.cpp'), '-o', linux(binary)]
    execute(command, RESULTS / 'q4-fields-oracle-build.txt')
    offset = payload_base + item['offset']
    output = execute([linux(binary), linux(MODEL), offset, rows, columns],
                     RESULTS / 'q4-fields-oracle-run.txt')
    result = json.loads(output.decode().strip())
    result.update({'tensor': IDENTITY, 'artifact_id': artifact_id, 'source': str(MODEL.resolve()),
                   'absolute_payload_offset': offset, 'format': item['format'],
                   'cpu_only': True, 'gpu_executed': False})
    if (not result.get('full_original_bytes_exact') or
            not result.get('header_invalid_version_shape_size_rejected') or
            not result.get('ineligible_sign_exponent_shape_size_fallback') or
            result['encoded_bytes'] >= result['raw_bytes']):
        raise RuntimeError('Q4 artifact codec CPU oracle failed')
    result_path = RESULTS / 'q4-fields-artifact-oracle.json'
    result_path.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
