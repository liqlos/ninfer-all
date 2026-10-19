import argparse
import collections
import csv
import hashlib
import json
import mmap
import pathlib
import struct
import zlib
import numpy as np


def entropy(a, alphabet):
    counts = np.bincount(a.ravel(), minlength=alphabet)
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum()), counts


def examine(raw, fmt, blocksizes):
    stride = 144 if fmt == 'gguf_q4_k' else 210
    a = raw.reshape(-1, stride)
    low = a[:, 16:144] if stride == 144 else a[:, :128]
    codes = np.stack((low & 15, low >> 4), axis=-1).reshape(-1)
    h, hist = entropy(codes, 16)
    top = np.argsort(hist)[-7:][::-1]
    escaped = ~np.isin(codes, top)
    result = {'low_entropy_bits': h, 'low_hist': hist.tolist(), 'top7': top.tolist(), 'escape_fraction': float(escaped.mean())}
    result['candidates'] = []
    for nb in blocksizes:
        usable = len(a) // nb * nb
        e = escaped[:usable * 256].reshape(-1, nb * 256).sum(axis=1)
        coded = nb * 16 + nb * 96 + (e + 1) // 2 if stride == 144 else nb * 82 + nb * 96 + (e + 1) // 2
        coded = ((coded + 15) // 16) * 16
        stored = np.minimum(coded, nb * stride) + 8
        result['candidates'].append({'codec': 'top7_escape', 'gguf_blocks_per_tile': nb, 'ratio': float(stored.sum() / (usable * stride)), 'raw_fraction': float((coded >= nb * stride).mean()), 'table_bytes_per_tensor': 16, 'index_bytes_per_tile': 8, 'payload_alignment': 16})
        sampled = a[:min(usable, 4096)].reshape(-1, nb * stride)
        size = sum(min(len(row), len(zlib.compress(row.tobytes(), 1))) + 8 for row in sampled)
        result['candidates'].append({'codec': 'zlib1_raw_fallback', 'gguf_blocks_per_tile': nb, 'ratio': size / sampled.size, 'note': 'size reference only, no fused decoder'})
    if stride == 210:
        high = a[:, 128:192]
        result['high_byte_entropy_bits'], hh = entropy(high, 256)
        result['high_byte_zero_fraction'] = float((high == 0).mean())
        ql = a[:, :128].reshape(-1, 2, 64)
        qh = high.reshape(-1, 2, 32)
        residual = np.empty_like(qh)
        for j in range(4):
            lo = ql[:, :, :32] if j in (0, 2) else ql[:, :, 32:]
            nibble = (lo >> (4 if j >= 2 else 0)) & 15
            actual = ((qh >> (j * 2)) & 3) ^ 2
            expected = np.where(nibble >= 8, 3, 0).astype(np.uint8)
            if j == 0:
                residual.fill(0)
            residual |= (actual ^ expected) << (j * 2)
        result['residual_high_entropy_bits'], rh = entropy(residual, 256)
        result['residual_high_zero_fraction'] = float((residual == 0).mean())
        for nb in blocksizes:
            usable = len(a) // nb * nb
            nz = (residual.reshape(-1, 64)[:usable] != 0).reshape(-1, nb * 64).sum(axis=1)
            stored = np.minimum(((nb * 146 + nb * 8 + nz + 15) // 16) * 16, nb * 210) + 8
            result['candidates'].append({'codec': 'q6_sign_residual_bitmap_byte', 'gguf_blocks_per_tile': nb, 'ratio': float(stored.sum() / (usable * stride))})
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--full', nargs='*', default=[])
    args = p.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(args.model, 'rb') as f:
        magic, length, aid = struct.unpack('<8sQ16s', f.read(32))
        assert magic == b'NINFER\0\3'
        directory = json.loads(f.read(length))
        start = (32 + length + 4095) // 4096 * 4096
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        roles = collections.defaultdict(list)
        for name, binding in directory['bindings'].items():
            if 'object' in binding:
                roles[binding['object']].append(name)
            for part in binding.get('parts', []):
                roles[part['object']].append(name)
        rows = []
        for t in directory['objects']:
            if t['kind'] != 'tensor':
                continue
            row = dict(t, roles=roles[t['id']])
            if t['format'] in ('gguf_q4_k', 'gguf_q6_k'):
                stride = 144 if t['format'] == 'gguf_q4_k' else 210
                total = t['bytes'] // stride
                full = t['id'] in args.full
                if full:
                    raw = np.frombuffer(mm, dtype=np.uint8, offset=start + t['offset'], count=t['bytes'])
                else:
                    sections = []
                    count = min(512, total // 4)
                    for frac in (0, 0.333, 0.666, 0.99):
                        b = min(int(total * frac), total - count)
                        sections.append(np.frombuffer(mm, dtype=np.uint8, offset=start + t['offset'] + b * stride, count=count * stride).copy())
                    raw = np.concatenate(sections)
                row['analysis'] = examine(raw, t['format'], (1, 4, 16, 64))
                row['analysis']['mode'] = 'full' if full else 'stratified_sample'
                row['analysis']['analyzed_bytes'] = raw.size
                if full:
                    row['analysis']['sha256'] = hashlib.sha256(raw).hexdigest()
                del raw
            rows.append(row)
        result = {'model': str(pathlib.Path(args.model).resolve()), 'file_bytes': len(mm), 'payload_offset': start, 'artifact_id': aid.hex(), 'metadata': directory['metadata'], 'provenance': directory['provenance'], 'format_counts': dict(collections.Counter(t['format'] for t in rows)), 'tensors': rows}
        (out / 'analysis.json').write_text(json.dumps(result, indent=2))
        fields = ['id', 'roles', 'format', 'bytes', 'mode', 'low_entropy_bits', 'escape_fraction', 'codec', 'gguf_blocks_per_tile', 'ratio']
        with (out / 'analysis.csv').open('w', newline='') as g:
            w = csv.DictWriter(g, fieldnames=fields)
            w.writeheader()
            for row in rows:
                a = row.get('analysis')
                if a:
                    for candidate in a['candidates']:
                        w.writerow({key: ({**row, **a, **candidate}).get(key) for key in fields})
        print(json.dumps({'formats': result['format_counts'], 'sampled': sum('analysis' in t for t in rows), 'best': sorted([(min(c['ratio'] for c in t['analysis']['candidates'] if c['codec'] != 'zlib1_raw_fallback'), t['id'], t['roles'], t['analysis']['escape_fraction']) for t in rows if 'analysis' in t])[:12]}, indent=2))


if __name__ == '__main__':
    main()
