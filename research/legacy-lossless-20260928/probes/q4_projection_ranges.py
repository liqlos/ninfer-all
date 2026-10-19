import collections
import hashlib
import json
import mmap
import pathlib
import struct
import time

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parent
MODEL = pathlib.Path('/mnt/c/AI/staging/qwen38-jonathan-ninfer/qwen38-jonathan-q4km-proposal.ninfer')


def main():
    begin = time.monotonic()
    with MODEL.open('rb') as stream:
        magic, length, artifact_id = struct.unpack('<8sQ16s', stream.read(32))
        if magic != b'NINFER\0\3':
            raise ValueError('NInfer v3 required')
        raw_directory = stream.read(length)
        directory = json.loads(raw_directory)
        payload = (32 + length + 4095) // 4096 * 4096
        with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
            rows = []
            for item in directory['objects']:
                if item.get('format') != 'gguf_q4_k' or item['shape'][1] not in (6144, 10240, 17408):
                    continue
                blocks = item['bytes'] // 144
                halves = np.ndarray((blocks, 2), dtype='<u2', buffer=mapped,
                                    offset=payload + item['offset'], strides=(144, 2)).copy()
                ranges = []
                for field in range(2):
                    exponents = (halves[:, field] >> 10) & 31
                    signs = halves[:, field] >> 15
                    lo, hi = int(exponents.min()), int(exponents.max())
                    exp_bits = (hi - lo).bit_length()
                    sign_variable = bool(signs.min() != signs.max())
                    ranges.append({'exponent_min': lo, 'exponent_max': hi, 'exponent_bits': exp_bits,
                                   'sign_variable': sign_variable, 'sign_value': int(signs.min()),
                                   'coded_width': 10 + exp_bits + int(sign_variable)})
                rows.append({'id': item['id'], 'shape': item['shape'], 'source_bytes': item['bytes'],
                             'absolute_offset': payload + item['offset'], 'blocks': blocks,
                             'header_sha256': hashlib.sha256(halves.tobytes()).hexdigest(), 'fields': ranges,
                             'ideal_saved_bytes_before_alignment': blocks * 4 - sum((blocks * r['coded_width'] + 7) // 8 for r in ranges)})
                del halves
    result = {'artifact_id': artifact_id.hex(), 'directory_sha256': hashlib.sha256(raw_directory).hexdigest(),
              'scope': 'all Q4 roots with K6144/K10240/K17408; full half headers only, no quant payload roundtrip',
              'scan_seconds': time.monotonic() - begin, 'tensor_count': len(rows), 'tensors': rows}
    (ROOT / 'results' / 'q4-output-projection-half-ranges.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({'tensor_count': len(rows), 'shape_counts': dict(collections.Counter(str(r['shape']) for r in rows)),
                      'field_descriptors': dict(collections.Counter(str(r['fields']) for r in rows)),
                      'scan_seconds': result['scan_seconds']}, indent=2))


if __name__ == '__main__':
    main()
