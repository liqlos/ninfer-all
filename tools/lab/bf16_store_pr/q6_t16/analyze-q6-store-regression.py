import hashlib
import json
import math
import statistics
from pathlib import Path


OUT = Path(__file__).resolve().parents[1] / "results/bf16-store-q6-t16"


def load(name):
    record = json.loads((OUT / (name + ".json")).read_text(encoding="utf-8"))
    if record["status"] != "PASS":
        raise RuntimeError("incomplete series " + name)
    if record["cleanup_errors"] or not record["production_config_unchanged"]:
        raise RuntimeError("runtime cleanup/config gate failed")
    return record


def summarize(record, reference, candidate):
    groups = {}
    for row in record["runs"]:
        groups.setdefault((row["qtype"], row["t"]), []).append(row)
    result = []
    for (qtype, t), rows in groups.items():
        blocks = []
        pairs = []
        for block in sorted({x["block"] for x in rows}):
            ordered = sorted((x for x in rows if x["block"] == block), key=lambda x: x["position"])
            if len(ordered) != 4:
                raise RuntimeError("incomplete balanced block")
            values = {arm: statistics.geometric_mean(x["median_us"] for x in ordered if x["arm"] == arm)
                      for arm in (reference, candidate)}
            blocks.append(math.log(values[reference] / values[candidate]))
            for offset in (0, 2):
                pair = {x["arm"]: x["median_us"] for x in ordered[offset:offset + 2]}
                pairs.append(pair[reference] / pair[candidate])
        mean = statistics.mean(blocks)
        row = {"qtype": qtype, "n": rows[0]["n"], "k": rows[0]["k"], "t": t,
            "reference": reference, "candidate": candidate, "blocks": len(blocks), "pairs": len(pairs),
            "reference_us": statistics.geometric_mean(x["median_us"] for x in rows if x["arm"] == reference),
            "candidate_us": statistics.geometric_mean(x["median_us"] for x in rows if x["arm"] == candidate),
            "speed_gain_percent": 100 * math.expm1(mean), "latency_reduction_percent": 100 * -math.expm1(-mean),
            "block_speed_gains_percent": [100 * math.expm1(v) for v in blocks], "adjacent_pair_ratios": pairs}
        if len(blocks) == 8:
            radius = 2.3646242516 * statistics.stdev(blocks) / math.sqrt(8)
            row["speed_gain_t95_percent"] = [100 * math.expm1(mean - radius), 100 * math.expm1(mean + radius)]
            positive = sum(x > 0 for x in blocks)
            negative = sum(x < 0 for x in blocks)
            n = positive + negative
            row["exact_block_sign_p_two_sided"] = min(1, 2 * sum(math.comb(n, i) for i in range(min(positive, negative) + 1)) / 2**n)
        result.append(row)
    return result


def main():
    build = json.loads((OUT / "checked-build.json").read_text(encoding="utf-8"))
    result = {"scope": "public Linear Op, synthetic packed weights, RTX3090, cold-cache eager",
        "source_base": build["source"], "working_diff_sha256": hashlib.sha256(build["diff"].encode()).hexdigest(),
        "primary": summarize(load("confirm-control-checked"), "control", "checked")[0],
        "secondary_screens": summarize(load("secondary-control-checked"), "control", "checked"),
        "attribution_screen": summarize(load("attribution-before-checked"), "before", "checked")[0],
        "pilot_not_in_confirmation": summarize(load("pilot-control-checked"), "control", "checked")[0],
        "statistics": "geometric means of process event medians; primary95%t interval uses8 balanced-block log ratios (df7); sign test uses block signs; no timer/systematic-error interval or whole-model claim; secondary/attribution are exploratory"}
    (OUT / "statistics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for row in [result["primary"], *result["secondary_screens"], result["attribution_screen"]]:
        print(f'{row["reference"]}: {row["qtype"]} T{row["t"]} {row["reference_us"]:.3f} -> {row["candidate_us"]:.3f} us, {row["speed_gain_percent"]:+.3f}%')
    print(result["primary"]["speed_gain_t95_percent"], result["primary"]["exact_block_sign_p_two_sided"])


if __name__ == "__main__":
    main()
