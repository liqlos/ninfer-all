import collections
import json
import pathlib

p = pathlib.Path('/root/ninfer-stage/results/production-requests.jsonl')
rows = collections.deque(maxlen=400)
with p.open() as f:
    for line in f:
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
instance = rows[-1]['server_instance_id']
selected = [r for r in rows if r.get('server_instance_id') == instance]
out = pathlib.Path('/mnt/c/AI/tuning/ninfer-lossless-20260928/results/baseline-server-events.jsonl')
out.write_text(''.join(json.dumps(r) + '\n' for r in selected))
print(json.dumps({'server_instance': instance, 'events': dict(collections.Counter(r.get('event') for r in selected)), 'saved': str(out)}))
