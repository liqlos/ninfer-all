import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import pathlib
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=pathlib.Path, required=True)
    parser.add_argument("--probe-root", type=pathlib.Path, required=True)
    parser.add_argument("--output-root", type=pathlib.Path, required=True)
    args = parser.parse_args()
    helper = args.probe_root.parent / "q4_compact_a_probe/generate_timing.py"
    spec = importlib.util.spec_from_file_location("timing_host", helper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    previous = sys.argv
    captured = io.StringIO()
    try:
        sys.argv = [str(helper), "--source-root", str(args.source_root),
                    "--probe-root", str(args.probe_root), "--output-root", str(args.output_root)]
        with contextlib.redirect_stdout(captured):
            module.main()
    finally:
        sys.argv = previous
    metadata = json.loads(captured.getvalue())
    path = args.output_root / "timing.cuh"
    data = path.read_text(encoding="utf-8")
    old = "const std::array<int, 8> rows{0, 1, 7, kRows / 4, kRows / 3, kRows / 2, kRows - 2, kRows - 1};"
    new = "const std::array<int, 16> rows{0, 1, 7, 15, 16, 31, 32, 63, 64, 127, 128, kRows / 4, kRows / 3, kRows / 2, kRows - 2, kRows - 1};"
    assert data.count(old) == 1
    data = data.replace(old, new)
    old = 'independent_fp64_samples\\\":40'
    assert data.count(old) == 1
    data = data.replace(old, 'independent_fp64_samples\\\":80')
    path.write_bytes(data.encode("utf-8"))
    metadata.update(independent_fp64_samples=80,
                    boundary_rows=[15, 16, 31, 32, 63, 64, 127, 128],
                    generated_timing_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    host_generator_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
                    timing_helper_sha256=hashlib.sha256(helper.read_bytes()).hexdigest())
    print(json.dumps(metadata))


if __name__ == "__main__":
    main()
