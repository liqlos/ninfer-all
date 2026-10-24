import json,math,statistics,sys
from pathlib import Path
out=Path(__file__).resolve().parents[1]/'results/gate-bf16-pr';tag=sys.argv[1];x=json.loads((out/(tag+'.json')).read_text());assert x['status']=='PASS'
y={'input':tag,'scope':'public GGUF_Q4_K linear_swiglu','endpoints':{}}
for t in (16,128,512,1024):
 values=[]
 for block in sorted({v['block'] for v in x['runs']}):
  scores={a:[] for a in ('control','candidate')}
  for run in x['runs']:
   if run['block']==block:scores[run['arm']].append(float(next(p['median_us'] for p in run['points'] if int(p['t'])==t)))
  values.append(statistics.mean(map(math.log,scores['control']))-statistics.mean(map(math.log,scores['candidate'])))
 mean=statistics.mean(values);r={'blocks':len(values),'speedup_percent':100*math.expm1(mean),'block_speedup_percent':[100*math.expm1(v) for v in values]}
 if len(values)==8:
  width=2.364624251*statistics.stdev(values)/math.sqrt(8);r['speedup_95ci_percent']=[100*math.expm1(mean-width),100*math.expm1(mean+width)]
 y['endpoints'][str(t)]=r
primary=y['endpoints']['1024'];y['eligible_for_confirmation']=primary['speedup_percent']>0.3 and all(v>0 for v in primary['block_speedup_percent']) and all(v['speedup_percent']>-.5 for v in y['endpoints'].values())
(out/(tag+'-statistics.json')).write_text(json.dumps(y,indent=2),encoding='utf-8');print(json.dumps(y,indent=2))
