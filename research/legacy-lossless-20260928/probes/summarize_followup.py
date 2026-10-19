import collections
import json
import pathlib
import re
import statistics

ROOT = pathlib.Path(__file__).resolve().parent
RESULTS = ROOT / 'results'


def kernel_group(name):
    fields = re.search(r'q([46])_fields6kernelILi(\d+)', name)
    if fields:
        width = re.search(r'q4_fields6kernelILi\d+ELi(\d+)', name)
        return f'fields_Q{fields[1]}_T{fields[2]}' + (f'_K{width[1]}' if width else '')
    if 'q4_fields_cu' in name and 'unpack_kernel' in name:
        return 'Q4_full_unpack'
    m = re.search(r'vec6kernelIL9ggml_type(\d+)ELi(\d+)ELb([01])', name)
    if m:
        return f'vec_Q{4 if m[1] == "12" else 6 if m[1] == "14" else m[1]}_T{m[2]}_' + ('swiglu' if m[3] == '1' else 'projection')
    if 'mul_mat_q' in name:
        return 'MMQ_projection'
    if 'quantize_vector' in name or 'quantize_matrix' in name:
        return 'activation_quantization'
    if 'gated_delta_net' in name:
        return 'GDN_state'
    if 'attention' in name or 'gqa' in name or 'flash_attn' in name:
        return 'attention'
    return 'other'


def profile(tag='cupti13'):
    trace = [json.loads(line) for line in (RESULTS / f'runtime-kernels-{tag}.jsonl').open()]
    issues = [r for r in trace if r['type'] in ('trace_error', 'dropped')]
    geometry = collections.defaultdict(set)
    for r in trace:
        if r['type'] == 'vec_geometry':
            geometry[r['name']].add((r['rows'], r['K'], r['row_bytes']))
    correlations = {r['correlation_id']: r for r in trace if r['type'] == 'vec_geometry'}
    directly_matched = collections.defaultdict(list)
    for r in trace:
        if r['type'] == 'kernel' and r['correlation_id'] in correlations:
            g = correlations[r['correlation_id']]
            directly_matched[(g['name'], g['rows'], g['K'], r['graph_id'])].append(r['duration_ns'])
    request_receipt = json.loads((RESULTS / f'profile-{tag}-requests.json').read_text())
    requests = request_receipt['requests']
    events = [json.loads(line) for line in (RESULTS / f'profile-{tag}-server-events.jsonl').open()]
    done = [r for r in events if r['event'] == 'request_done']
    if len(done) != len(requests) or issues:
        raise ValueError('Profile requests or trace integrity failed')
    rows = []
    for request, event in zip(requests, done):
        start = int(request['host_start_epoch_ns'])
        end = int(request['host_end_epoch_ns'])
        split = int(event['timestamp_unix_ms'] * 1000000 - event['timings_seconds']['decode'] * 1000000000)
        phases = {}
        for phase, lo, hi in [('prefill', start, split), ('decode', split, end)]:
            kernels = [r for r in trace if r['type'] == 'kernel' and lo <= r['start_ns'] < hi]
            groups = collections.defaultdict(list)
            for r in kernels:
                groups[kernel_group(r['name'])].append(r)
            total = sum(r['duration_ns'] for r in kernels)
            phases[phase] = {'kernel_count': len(kernels), 'sum_kernel_ns': total, 'groups': {}}
            for group, items in groups.items():
                ns = sum(r['duration_ns'] for r in items)
                phases[phase]['groups'][group] = {'launches': len(items), 'total_ns': ns, 'fraction_of_kernel_time': ns / total if total else None,
                    'median_kernel_ns': statistics.median(r['duration_ns'] for r in items),
                    'registers_range': [min(r['registers_per_thread'] for r in items), max(r['registers_per_thread'] for r in items)],
                    'shared_range': [min(r['static_shared_bytes'] + r['dynamic_shared_bytes'] for r in items), max(r['static_shared_bytes'] + r['dynamic_shared_bytes'] for r in items)]}
        rows.append({'scenario': request['label'], 'usage': request['response']['usage'], 'server_timing_profiled': event['timings_seconds'], 'speculative_profiled': event.get('speculative'), 'phases': phases})
    return {'method': 'CUPTI13.1 Activity in isolated container; phase boundaries inferred from server request_done timing, not NVTX',
            'tag': tag, 'command': request_receipt['command'],
            'traffic_measured': False, 'nsight_error': 'ERR_NVGPUCTRPERM including Windows-admin WSL launch', 'trace_issues': issues,
            'direct_geometry_timings': [{'kernel': name, 'rows': n, 'K': k, 'graph_id': graph, 'launches': len(ns),
                                         'median_kernel_ns': statistics.median(ns)} for (name, n, k, graph), ns in sorted(directly_matched.items())],
            'direct_geometry_limit': 'correlation-ID join covers direct launches only; graph replay kernels lack geometry mapping, so per-role totals cannot be extrapolated to all decode kernels',
            'vec_geometry': [{'kernel': k, 'geometries': [{'rows': n, 'K': width, 'row_bytes': rb, 'logical_stored_bytes_per_pass': n * rb * (2 if 'ELb1' in k else 1)} for n, width, rb in sorted(v)]} for k, v in geometry.items()], 'requests': rows}


def micro():
    rows = []
    files = [RESULTS / 'old-repro-20260929.jsonl', *sorted((RESULTS / 'huffman').glob('*-bench.jsonl')), *sorted(RESULTS.glob('word-rans-weight-*-bench.jsonl')), *sorted(RESULTS.glob('scales-weight-*-bench.jsonl')),
             *sorted(RESULTS.glob('q4-register-weight-*-real-*.jsonl')),
             *sorted(RESULTS.glob('q6-layout-weight-*-bench-t*.jsonl'))]
    for path in files:
        if not path.exists() or not path.stat().st_size:
            continue
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if 'samples_ms' not in r:
                continue
            samples = collections.defaultdict(list)
            for s in r['samples_ms']:
                samples[s['mode']].append(s['ms'])
            sizes = {k: r[k] for k in ('baseline_bytes', 'compressed_bytes_with_index_and_lut', 'resident_bytes_with_index_lut', 'resident_bytes', 'scale_format_bytes', 'raw_planar_bytes_including_headers', 'raw_planar_bytes_including_header', 'compressed_bytes_including_header', 'transposed_bytes_including_header', 'shared_bytes_per_cta', 'resident_ctas_per_sm') if k in r}
            timings = {k: {'median': statistics.median(v), 'min': min(v), 'max': max(v)} for k, v in samples.items()}
            base = timings.get('baseline', timings.get('raw', timings.get('dynamic_baseline')))
            compressed = r.get('compressed_bytes_with_index_and_lut', r.get('resident_bytes_with_index_lut', r.get('resident_bytes', r.get('scale_format_bytes'))))
            ratio = compressed / r['baseline_bytes'] if compressed is not None else None
            budget = base['median'] * (1 - ratio) if base and ratio is not None else None
            rows.append({'file': str(path.relative_to(ROOT)), 'matrix': r.get('matrix'), 'T': r['T'], 'sizes': sizes, 'timing_ms': timings, 'resources': r.get('resources'),
                         'activation_snapshot': r.get('activation_snapshot'), 'fused': r.get('fused'),
                         'fp32_output_bit_exact': r.get('fp32_output_bit_exact'), 'bf16_output_bit_exact': r.get('bf16_output_bit_exact'),
                         'optimistic_read_only_saving_ms': budget, 'budget_method': 'raw operator time times resident byte fraction; optimistic proxy, not measured DRAM saving'})
    return rows


def telemetry():
    rows = []
    for path in sorted(RESULTS.glob('measure-*.json')):
        receipt = json.loads(path.read_text())
        values = [s['gpu_csv'].split(',') for s in receipt['samples'] if s['exit_code'] == 0]
        active = [v for v in values if float(v[6]) >= 90]
        rows.append({'file': str(path.relative_to(ROOT)), 'exit_code': receipt['exit_code'],
                     'samples': len(values), 'active_samples': len(active),
                     'active_ranges': {key: [min(float(v[i]) for v in active), max(float(v[i]) for v in active)]
                         for i, key in enumerate(('timestamp', 'sm_mhz', 'memory_mhz', 'temperature_c', 'power_w', 'used_memory_mib', 'gpu_util_percent')) if i and active},
                     'power_or_clock_controls_changed': receipt['power_or_clock_controls_changed']})
    return rows


def main():
    result = {'profile': profile(), 'microbenchmarks': micro(), 'telemetry': telemetry()}
    for name, path in [('real_activation_requests', RESULTS / 'snapshot-requests.json'),
                       ('activation_gating_check', RESULTS / 'activation-gating-check.json'),
                       ('register_fields', RESULTS / 'register-fields.json')]:
        if path.exists():
            result[name] = json.loads(path.read_text())
    (RESULTS / 'followup-summary.json').write_text(json.dumps(result, indent=2))
    for row in result['microbenchmarks']:
        if row['T'] == 1:
            print(row['file'], row['timing_ms'], row['optimistic_read_only_saving_ms'])
    for row in result['profile']['requests']:
        print(row['scenario'], {p: {k: round(v['fraction_of_kernel_time'] * 100, 2) for k, v in x['groups'].items()} for p, x in row['phases'].items()})


if __name__ == '__main__':
    main()
