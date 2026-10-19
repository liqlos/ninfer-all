import json
import pathlib
import statistics

root = pathlib.Path(__file__).resolve().parent
results = root / 'results'
records = [json.loads(line) for line in (results / 'integration-q4-gdn-byteword-weight-000060-t4.jsonl').read_text().splitlines()]
timings = []
for rows in (16384, 10240, 6144):
    medians = {variant: statistics.median([record['milliseconds_per_launch'] for record in records
               if record.get('kind') == 'timing' and record['rows'] == rows and record['variant'] == variant])
               for variant in ('raw_vector', 'v6_public_word_default_grid', 'byte_reconstructed_cap6', 'word_cap6')}
    timings.append({'rows': rows, 'median_ms': medians})
matrix = json.loads((results / 'q4-mmq-private-v2-v2-first-t9-12-1024-summary.json').read_text())
mmq = []
for tensor in matrix['matrices']:
    for case in tensor['cases']:
        if 'T' not in case:
            continue
        medians = {key: statistics.median(case[key]) for key in
                   ('raw_graph_ms', 'unpack_raw_graph_ms', 'generic_direct_graph_ms', 'fast_direct_graph_ms')}
        mmq.append({'tensor': tensor['tensor'], 'T': case['T'], 'view': case['view'], 'median_ms': medians})
summary = {'gdn_vector': timings, 'gdn_exact': records[-1]['all_checks_exact'],
           'byte_control': 'reconstructed current kernel, not sealed v5',
           'mmq_v2': mmq, 'mmq_inputs': 'synthetic BF16', 'full_answer_gain_established': False}
(results / 'q4-latest-operator-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
