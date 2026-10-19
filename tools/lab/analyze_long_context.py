import collections
import hashlib
import json
import pathlib
import sqlite3


ROOT = pathlib.Path(__file__).resolve().parents[3]
OUT = ROOT / 'results/profile32k'


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def union_ns(intervals):
    total, right = 0, -1
    for start, end in sorted(intervals):
        total += max(0, end - max(start, right))
        right = max(right, end)
    return total


def family(name):
    if 'mul_mat_q_stream_k_fixup<' in name:
        return 'mmq_fixup'
    if 'mul_mat_q<' in name:
        return 'mmq_q6' if '(ggml_type)14,' in name else 'mmq_q4' if '(ggml_type)12,' in name else 'mmq_other'
    if any(s in name for s in ('causal_attention_', 'sliding_window_attention_')):
        return 'attention'
    if 'causal_conv1d_' in name:
        return 'gdn_convolution_support'
    if any(s in name for s in ('gated_delta_net', 'gdn_', 'prepare_wy_wu', 'state_passing')):
        return 'gdn'
    if 'quantize_' in name or 'dequantize_' in name:
        return 'quantize_dequantize'
    if 'store_plane_kernel' in name:
        return 'output_conversion'
    if 'gguf' in name and ('vec' in name.lower() or 'paired_q4' in name):
        return 'gguf_vector'
    return 'other'


def main():
    db_path = OUT / 'json-r2-nsys.sqlite'
    receipt_path = OUT / 'json-r2-receipt.json'
    capture = json.loads(receipt_path.read_text())
    assert capture['equality_gate'] == 'PASS' and capture['status'] == 'CAPTURED'
    db = sqlite3.connect(f'file:{db_path.as_posix()}?mode=ro', uri=True)
    contexts = db.execute('select distinct globalPid,contextId from CUPTI_ACTIVITY_KIND_KERNEL').fetchall()
    assert len(contexts) == 1, 'Capture contains multiple processes or CUDA contexts'
    strings = dict(db.execute('select id,value from StringIds'))
    ranges = collections.defaultdict(list)
    for start, end, name in db.execute('select start,end,textId from NVTX_EVENTS where eventType=59 and end is not null order by start'):
        ranges[strings.get(name)].append((start, end))
    groups = []
    for window in ranges['prefill']:
        if not groups or window[0] - groups[-1][-1][1] > 1_000_000:
            groups.append([])
        groups[-1].append(window)
    assert len(groups) == 3, 'Expected startup, distinct warmup, frozen measured request'
    prefill = groups[-1]
    decode = [w for w in ranges['decode'] if w[0] >= prefill[-1][1]]
    events = [json.loads(line) for line in (OUT / 'json-r2-events.jsonl').read_text().splitlines()]
    done = [r for r in events if r['event'] == 'request_done']
    assert len(done) == 2 and done[-1]['request']['requested_output_tokens'] == 512
    assert done[-1]['result']['prompt_tokens'] == 32756 and done[-1]['result']['prefix_cache_hit_tokens'] == 0
    assert len(prefill) == done[-1]['engine_timing']['units']['prefill']
    assert len(decode) == done[-1]['engine_timing']['decode']['rounds']
    kernels = list(db.execute('select start,end,demangledName,gridX,gridY,gridZ,blockX,blockY,blockZ,registersPerThread,staticSharedMemory,dynamicSharedMemory from CUPTI_ACTIVITY_KIND_KERNEL order by start'))
    api = {c: (a, b, tid) for a, b, c, tid in db.execute('select start,end,correlationId,globalTid from CUPTI_ACTIVITY_KIND_RUNTIME')}
    stages = [(a, b, strings[t], tid) for a, b, t, tid in db.execute('select start,end,textId,globalTid from NVTX_EVENTS where eventType=59 and end is not null') if strings.get(t) in {'prefill.post_mixer', 'prefill.attention', 'prefill.gdn', 'prefill.mtp_chunk'}]
    launch_stages = collections.Counter()
    for start, end, text_id, correlation in db.execute('select start,end,demangledName,correlationId from CUPTI_ACTIVITY_KIND_KERNEL'):
        name = strings[text_id]
        if not (prefill[0][0] <= start and end <= prefill[-1][1] and 'mul_mat_q<(ggml_type)12,' in name and '(bool)0, (bool)0>' in name):
            continue
        a, b, tid = api.get(correlation, (-1, -1, -1))
        candidates = [(y - x, label) for x, y, label, stage_tid in stages if stage_tid == tid and x <= a and b <= y]
        launch_stages[min(candidates)[1] if candidates else 'unmapped'] += end - start
    phases = {}
    for phase, windows in [('prefill', prefill), ('decode', decode)]:
        assert all(a[1] <= b[0] for a, b in zip(windows, windows[1:]))
        counts, ns = collections.Counter(), collections.Counter()
        by_name, layouts = collections.Counter(), collections.Counter()
        intervals, partial, partial_ns = [], 0, 0
        for row in kernels:
            start, end, text_id, *layout = row
            overlap = [(a, b) for a, b in windows if start < b and end > a]
            if not overlap:
                continue
            if not any(a <= start and end <= b for a, b in overlap):
                partial += 1
                partial_ns += end - start
                continue
            name = strings[text_id]
            bucket = family(name)
            counts[bucket] += 1
            ns[bucket] += end - start
            by_name[name] += end - start
            layouts[(name, *layout)] += 1
            intervals.append((start, end))
        total = sum(ns.values())
        wall = sum(b - a for a, b in windows)
        covered = union_ns(intervals)
        phases[phase] = {
            'ranges': len(windows), 'start_ns': windows[0][0], 'end_ns': windows[-1][1],
            'nvtx_wall_seconds': wall / 1e9, 'summed_kernel_seconds': total / 1e9,
            'kernel_interval_union_seconds': covered / 1e9,
            'nvtx_wall_without_contained_kernels_seconds': (wall - covered) / 1e9,
            'partially_overlapping_kernel_count': partial, 'partial_kernel_seconds': partial_ns / 1e9,
            'families': {k: {'count': counts[k], 'summed_seconds': v / 1e9, 'share_of_kernel_sum_pct': 100 * v / total} for k, v in ns.most_common()},
            'kernels': [{'name': k, 'summed_seconds': v / 1e9, 'share_of_kernel_sum_pct': 100 * v / total} for k, v in by_name.most_common()],
            'launch_shapes': [{'name': k[0], 'grid': list(k[1:4]), 'block': list(k[4:7]), 'registers_per_thread': k[7], 'static_shared_bytes': k[8], 'dynamic_shared_bytes': k[9], 'count': v} for k, v in layouts.most_common()],
        }
    result = {
        'schema': 'ninfer-json32k-profile-v1', 'diagnostic_only': True, 'performance_claim': False,
        'method': 'Final contiguous outer prefill group and subsequent decode ranges. Startup and distinct warmup excluded. Counts cross-checked against request log. Fully contained GPU kernels only; boundary overlaps reported separately. Kernel sums and interval unions are not critical-path fractions or occupancy. Uncovered NVTX wall includes API, copies, synchronization and trace overhead; it is not necessarily removable host overhead.',
        'capture_receipt_sha256': digest(receipt_path), 'sqlite_sha256': digest(db_path),
        'request': {'usage': capture['measured']['usage'], 'output_sha256': capture['measured']['output_sha256'], 'timings_instrumented_seconds': done[-1]['timings_seconds'], 'engine_timing': done[-1]['engine_timing']},
        'q4_fp32_launch_stage_seconds': {k: v / 1e9 for k, v in launch_stages.most_common()},
        'launch_stage_method': 'Kernel correlationId to CUDA API, then smallest containing named NVTX stage on the same thread. Stage labels do not identify individual gate/up weight roles.',
        'phases': phases,
    }
    path = ROOT / 'provenance/json32k-profile-analysis.json'
    with path.open('w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: {x: y for x, y in v.items() if x not in ('kernels', 'launch_shapes')} for k, v in phases.items()}, indent=2))


if __name__ == '__main__':
    main()
