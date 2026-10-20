import argparse
import hashlib
import json
import pathlib


def sha(data):
    return hashlib.sha256(data).hexdigest()


def replace_once(data, old, new):
    assert data.count(old) == 1, old[:80]
    return data.replace(old, new)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=pathlib.Path, required=True)
    parser.add_argument("--probe-root", type=pathlib.Path, required=True)
    parser.add_argument("--output-root", type=pathlib.Path, required=True)
    args = parser.parse_args()
    source, probe, out = (p.resolve() for p in (args.source_root, args.probe_root, args.output_root))
    assert not out.is_relative_to(source) and not out.is_relative_to(probe)
    raw = (probe.parent / "lab/mmq_swiglu_probe.cuh").read_bytes()
    assert sha(raw) == "97045b5b86c5849896d4bbc311228d2686f31f84f96935ae13a41279d2d725f1"
    text = raw.decode("utf-8")
    text = replace_once(text, '#include "../ffn_pair_probe/probe_api.h"',
                        '#include "' + (probe.parent / "ffn_pair_probe/probe_api.h").as_posix() + '"')
    marker = "        std::vector<std::array<double, 2>> block_times(3, {0.0, 0.0});"
    warmup = '''        const bool inverse_order = launch_id.find("r2-b") != std::string::npos;
        for (int warm = 0; warm < 4; ++warm) {
            const bool candidate = ((warm & 1) != 0) != inverse_order;
            const double ms = time_arm(candidate, gate, up, activation,
                static_cast<float*>(gate_plane.data), static_cast<float*>(up_plane.data),
                gate_fixup.data, up_fixup.data, candidate ? candidate_out : baseline_out, nullptr, 32);
            std::cout << "{\\"kind\\":\\"warmup\\",\\"sample\\":" << warm
                      << ",\\"candidate\\":" << (candidate ? "true" : "false")
                      << ",\\"repetitions\\":32,\\"ms\\":" << ms << "}\\n";
        }
'''
    text = replace_once(text, marker, warmup + marker)
    text = replace_once(text, "const bool candidate_first = (order_state & 1U) != 0;",
                        "const bool candidate_first = ((block & 1) != 0) != inverse_order;")
    text = replace_once(text, "            for (const int arm : sequence) {",
                        "            int sample = 0;\n            for (const int arm : sequence) {")
    text = replace_once(text, "is_candidate ? candidate_out : baseline_out, nullptr, 4);",
                        "is_candidate ? candidate_out : baseline_out, nullptr, 32);")
    marker = "                block_times[block][arm] += ms / 2.0;"
    sample = '''                std::cout << std::setprecision(10)
                          << "{\\"kind\\":\\"sample\\",\\"block\\":" << block
                          << ",\\"sample\\":" << sample++ << ",\\"arm\\":" << arm
                          << ",\\"repetitions\\":32,\\"ms\\":" << ms << "}\\n";
'''
    text = replace_once(text, marker, sample + marker)
    text = replace_once(text, 'repetitions_per_sample\\\":4', 'repetitions_per_sample\\\":32')
    text = replace_once(text, 'speedup_percent', 'time_reduction_percent')
    original_main = (probe / "main.cu").read_bytes()
    generated_main = replace_once(original_main.decode("utf-8"),
                                  '#include "../lab/mmq_swiglu_probe.cuh"',
                                  '#include "' + (out / "timing.cuh").as_posix() + '"')
    generated_main = replace_once(generated_main, '#include "../ffn_pair_probe/probe_api.h"',
                                  '#include "' + (probe.parent / "ffn_pair_probe/probe_api.h").as_posix() + '"')
    out.mkdir(parents=True, exist_ok=True)
    (out / "timing.cuh").write_bytes(text.encode("utf-8"))
    (out / "main.cu").write_bytes(generated_main.encode("utf-8"))
    print(json.dumps({"shared_harness_sha256": sha(raw), "original_main_sha256": sha(original_main),
                      "generated_main_sha256": sha(generated_main.encode("utf-8")),
                      "generated_timing_sha256": sha(text.encode("utf-8")),
                      "repetitions": 32, "warmup_samples": 4, "blocks": 3,
                      "device_arithmetic_changed": False, "selected_cubin_verification_required": True}))


if __name__ == "__main__":
    main()
