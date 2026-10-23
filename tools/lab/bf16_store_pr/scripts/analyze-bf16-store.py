import json
import math
import statistics
from pathlib import Path


OUT = Path(__file__).resolve().parents[1] / "results/bf16-store-pr"
record = json.loads((OUT / "qualification.json").read_text())
if record["status"] != "PASS_GPU_TESTS_AND_FIXED_OP_SCREEN":
    raise RuntimeError("incomplete fixed screen")
result = {
    "source": record["protocol"]["builds"],
    "scope": record["protocol"]["scope"],
    "estimand": "geometric mean of control/candidate process median latency ratios",
    "gain_percent": "100*(control/candidate-1); positive is faster",
    "uncertainty": "two-sided 95% Student t interval on three balanced-block log ratios (df=2)",
    "limitations": "exploratory Op screen, only three independent blocks; intervals are unadjusted for eight endpoints and exclude timer quantization/systematic error; no whole-model claim",
    "points": [],
}
for qtype, n, k in record["protocol"]["points"]:
    for t in record["protocol"]["tokens"]:
        rows = [x for x in record["benchmarks"] if x["qtype"] == qtype and x["t"] == t]
        if len(rows) != 12:
            raise RuntimeError("expected complete ABBA/BAAB/ABBA point")
        blocks = []
        pairs = []
        for block in range(3):
            ordered = sorted((x for x in rows if x["block"] == block), key=lambda x: x["position"])
            means = {arm: statistics.geometric_mean(x["median_us"] for x in ordered if x["arm"] == arm)
                     for arm in ("control", "candidate")}
            blocks.append(math.log(means["control"] / means["candidate"]))
            for start in (0, 2):
                pair = {x["arm"]: x["median_us"] for x in ordered[start:start + 2]}
                pairs.append(pair["control"] / pair["candidate"])
        center = statistics.mean(blocks)
        radius = 4.3026527299 * statistics.stdev(blocks) / math.sqrt(3)
        result["points"].append({"qtype": qtype, "n": n, "k": k, "t": t,
            "control_us": statistics.geometric_mean(x["median_us"] for x in rows if x["arm"] == "control"),
            "candidate_us": statistics.geometric_mean(x["median_us"] for x in rows if x["arm"] == "candidate"),
            "speed_gain_percent": 100 * math.expm1(center),
            "latency_reduction_percent": 100 * (1 - math.exp(-center)),
            "block_t95_gain_percent": [100 * math.expm1(center - radius), 100 * math.expm1(center + radius)],
            "block_log_ratios": blocks, "adjacent_pair_ratios": pairs})
(OUT / "statistics.json").write_text(json.dumps(result, indent=2) + "\n")
for p in result["points"]:
    print(f'{p["qtype"]} T{p["t"]}: {p["control_us"]:.3f} -> {p["candidate_us"]:.3f} us; speed {p["speed_gain_percent"]:+.3f}%')
