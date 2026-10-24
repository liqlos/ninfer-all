import json,math,statistics,sys
from pathlib import Path
OUT=Path(__file__).resolve().parents[1]/"results/cooperative-loads-pr"
def summarize(values):
    n=len(values);mean=statistics.mean(values)
    result={"blocks":n,"speedup_percent":100*math.expm1(mean),"block_speedup_percent":[100*math.expm1(v) for v in values]}
    if n==8:
        width=2.364624251*statistics.stdev(values)/math.sqrt(n)
        result["speedup_95ci_percent"]=[100*math.expm1(mean-width),100*math.expm1(mean+width)]
        result["ci_method"]="two-sided t interval of independent block log ratios, df7; systematic error excluded"
    return result
x=json.loads((OUT/(sys.argv[1]+".json")).read_text())
if x["status"]!="PASS":raise RuntimeError("series incomplete")
result={"input":sys.argv[1],"endpoints":{}}
if "prompt_gen" in x.get("protocol",{}):
    for metric in ("decode_seconds","total_seconds","prefill_seconds"):
        values=[]
        for b in sorted({r["block"] for r in x["runs"]}):
            rows=[r for r in x["runs"] if r["block"]==b]
            scores={arm:[r["report"]["tests"][0]["reps"][0]["timings"][metric] for r in rows if r["arm"]==arm] for arm in ("control","candidate")}
            values.append(statistics.mean(map(math.log,scores["control"]))-statistics.mean(map(math.log,scores["candidate"])))
        result["endpoints"][metric]=summarize(values)
    primary=result["endpoints"]["decode_seconds"]
    result["eligible_for_confirmation"]=primary["speedup_percent"]>0.3 and all(v>0 for v in primary["block_speedup_percent"])
else:
    for q,n,k,t in x["protocol"]["points"]:
        values=[]
        for b in sorted({r["block"] for r in x["runs"]}):
            rows=[r for r in x["runs"] if (r["qtype"],r["n"],r["k"],r["t"],r["block"])==(q,n,k,t,b)]
            scores={arm:[r["median_us"] for r in rows if r["arm"]==arm] for arm in ("control","candidate")}
            values.append(statistics.mean(map(math.log,scores["control"]))-statistics.mean(map(math.log,scores["candidate"])))
        result["endpoints"][f"{q}-n{n}-k{k}-t{t}"]=summarize(values)
(OUT/(sys.argv[1]+"-statistics.json")).write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
