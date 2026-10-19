import csv
import json
import pathlib
import statistics


ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT / 'results'


def stats(values):
    return {'median': statistics.median(values), 'min': min(values), 'max': max(values)}


def main():
    baseline = json.loads((OUT / 'baseline.json').read_text())
    done = [json.loads(s) for s in (OUT / 'baseline-server-events.jsonl').read_text().splitlines() if json.loads(s).get('event') == 'request_done']
    done.sort(key=lambda r: r['request']['request_id'])
    assert len(done) == len(baseline['runs']) + 1
    grouped = {}
    for run, server in zip(baseline['runs'], done[1:]):
        assert run['usage']['prompt_tokens'] == server['result']['prompt_tokens']
        assert run['usage']['completion_tokens'] == server['result']['completion_tokens']
        assert server['result']['prefix_cache_hit_tokens'] == 0
        grouped.setdefault(run['label'], []).append((run, server))
    serving = {}
    for label, rows in grouped.items():
        serving[label] = {'prompt_tokens': [r[1]['result']['prompt_tokens'] for r in rows], 'completion_tokens': 512, 'client': baseline['summary'][label]}
        metrics = {
            'server_decode_tok_s_512_over_decode': [512/s['timings_seconds']['decode'] for r,s in rows],
            'server_decode_ms_per_token': [s['timings_seconds']['decode']*1000/512 for r,s in rows],
            'server_prefill_seconds': [s['timings_seconds']['prefill'] for r,s in rows],
            'server_ttft_seconds': [s['timings_seconds']['ttft'] for r,s in rows],
            'mtp_acceptance_fraction': [s['speculative']['accepted_tokens']/s['speculative']['drafted_tokens'] for r,s in rows],
            'mtp_accepted_tokens_per_round': [s['speculative']['accepted_tokens']/s['speculative']['rounds'] for r,s in rows],
            'mtp_completion_tokens_per_round': [512/s['speculative']['rounds'] for r,s in rows]
        }
        serving[label]['server'] = {k: stats(v) for k,v in metrics.items()}
    projections = []
    for identity in ('weight-000060', 'weight-000001'):
        packed = json.loads((OUT / (identity+'-pack.json')).read_text())
        for s in (OUT / (identity+'-bench.jsonl')).read_text().splitlines():
            row = json.loads(s)
            raw = stats([r['ms'] for r in row['samples_ms'] if r['mode']=='baseline'])
            ans = stats([r['ms'] for r in row['samples_ms'] if r['mode']=='ans'])
            row.update(baseline_ms=raw, ans_ms=ans, latency_ratio=ans['median']/raw['median'], saved_resident_fraction=1-row['compressed_bytes_with_index_and_lut']/row['baseline_bytes'], saved_file_fraction=1-packed['artifact_file_bytes']/packed['raw_bytes'], artifact_file_bytes=packed['artifact_file_bytes'], maximum_saved_read_time_us=1000*raw['median']*(1-row['compressed_bytes_with_index_and_lut']/row['baseline_bytes']))
            projections.append(row)
    result = {'production_baseline': serving, 'projection_microbench': projections, 'compressed_whole_engine_A_B': 'not implemented or measured: prototype failed operator latency gate', 'unsupported': ['prefill MMQ', 'fused SwiGLU', 'NInfer artifact loader for experimental layout', 'compressed greedy/logits/MTP verification end to end'], 'footprint_note': 'matrix resident includes offsets and LUT; prototype rotates three matrix copies per arm for validation/benchmark only, not two full models; validation global expanded buffers are freed before timing'}
    (OUT / 'summary.json').write_text(json.dumps(result,indent=2))
    fields = ['matrix','T','N','K','baseline_bytes','artifact_file_bytes','compressed_bytes_with_index_and_lut','saved_resident_fraction','baseline_ms','ans_ms','latency_ratio']
    with (OUT/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader()
        for row in projections:
            d={k:row[k] for k in fields}
            d['baseline_ms']=row['baseline_ms']['median']
            d['ans_ms']=row['ans_ms']['median']
            w.writerow(d)
    print(json.dumps({'production':serving,'projection_T1':[r for r in projections if r['T']==1]},indent=2))


if __name__ == '__main__':
    main()
