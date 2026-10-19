import argparse
import gzip
import hashlib
import json
import pathlib
import urllib.request

parser = argparse.ArgumentParser(description='Check four archived outputs and token counts against a running endpoint; no timing measurement.')
parser.add_argument('--url', required=True)
parser.add_argument('--output')
args = parser.parse_args()
destination = pathlib.Path(args.output) if args.output else None
if destination and destination.exists():
    parser.error('refusing to overwrite an existing report')
root = pathlib.Path(__file__).resolve().parent
rows = [json.loads(line) for line in gzip.decompress((root / 'raw/hw20-20261017-p01-b0.jsonl.gz').read_bytes()).splitlines()]
checks = []
for row in rows:
    if row['repeat'] != 0:
        continue
    request = urllib.request.Request(args.url, json.dumps(row['request']).encode(), {'Content-Type': 'application/json'})
    output = ''
    usage = None
    finish = None
    with urllib.request.urlopen(request, timeout=180) as response:
        for line in response:
            if not line.startswith(b'data: ') or line[6:].strip() == b'[DONE]':
                continue
            event = json.loads(line[6:])
            if event.get('usage'):
                usage = event['usage']
            for choice in event.get('choices', []):
                output += choice.get('delta', {}).get('content') or ''
                if choice.get('finish_reason'):
                    finish = choice['finish_reason']
    passed = output == row['output'] and finish == row['finish_reason']
    passed = passed and usage is not None and all(usage.get(key) == row['usage'][key] for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'))
    checks.append({'category': row['category'], 'passed': passed, 'output_sha256': hashlib.sha256(output.encode()).hexdigest(), 'finish_reason': finish})
report = {'status': 'PASS' if len(checks) == 4 and all(check['passed'] for check in checks) else 'FAIL', 'scope': 'Four deterministic archived outputs and token counts; no speed claim.', 'checks': checks}
if destination:
    with destination.open('x', encoding='utf-8') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
print(json.dumps(report, indent=2))
raise SystemExit(0 if report['status'] == 'PASS' else 1)
