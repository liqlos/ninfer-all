import gzip, json, math, pathlib, statistics

root=pathlib.Path(__file__).resolve().parent
records={}
for path in sorted((root/"raw").glob("*.jsonl.gz")):
    for line in gzip.decompress(path.read_bytes()).splitlines():
        row=json.loads(line)
        records.setdefault(row["tag"],[]).append(row)
schedule=json.loads((root/"provenance"/"schedule.json").read_text(encoding="utf-8"))
if len(records)!=40 or sum(map(len,records.values()))!=480:
    raise SystemExit("expected 40 tags and 480 request records")
pairs=0
blocks=[]
for block in schedule["blocks"]:
    scores=[]
    for pair in block["pairs"]:
        baseline=records[pair["baseline_tag"]]
        candidate=records[pair["candidate_tag"]]
        categories=[]
        for category in ("ru","en","code","json"):
            b=[r for r in baseline if r["category"]==category]
            c=[r for r in candidate if r["category"]==category]
            if len(b)!=3 or len(c)!=3:
                raise SystemExit("each tag/category must have three repeats")
            for repeat in range(3):
                br=next(r for r in b if r["repeat"]==repeat)
                cr=next(r for r in c if r["repeat"]==repeat)
                for field in ("request","prompt_sha256","usage","finish_reason","output","cache_class"):
                    if br[field]!=cr[field]:
                        raise SystemExit("paired equality failed: "+field)
            bm=statistics.median(r["elapsed_seconds"] for r in b)
            cm=statistics.median(r["elapsed_seconds"] for r in c)
            categories.append((bm-cm)/bm*100.0)
        scores.append(statistics.mean(categories))
        pairs+=1
    blocks.append(statistics.mean(scores))
mean=statistics.mean(blocks)
se=statistics.stdev(blocks)/math.sqrt(len(blocks))
tcrit=2.7764451051977943
low=mean-tcrit*se
high=mean+tcrit*se
expected=(5.4070059690939685,5.003848973882475,5.810162964305462)
if any(abs(a-b)>1e-11 for a,b in zip((mean,low,high),expected)):
    raise SystemExit("recomputed result differs from published analysis")
print(json.dumps({"requests":480,"pairs":pairs,"block_means_percent":blocks,"mean_percent":mean,"ci95_percent":[low,high]},indent=2))
