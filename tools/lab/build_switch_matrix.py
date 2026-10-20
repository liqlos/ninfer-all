import ctypes
import hashlib
import json
import pathlib
import re
import subprocess
import sys
import time
import uuid

import profile_long_context as lab


ROOT = lab.ROOT
SOURCE = ROOT / "source"
OUT = ROOT / "results/switch-matrix"
BUILDER = "ninfer-builder"
MUTEXES = [
    "NInferV17RuntimeScreen", "NInferV14RuntimeMeasurement",
    "NInferV15ExportedOperatorSeries", "NInferV14CpuVariantPreparation",
    "NInferJonathanSpecShort20261018",
]
VARIANTS = {"w": (False, False), "q4": (True, False),
            "q6": (False, True), "both": (True, True)}


def main():
    arm, tag = sys.argv[1:]
    assert arm in VARIANTS and re.fullmatch(r"[a-z0-9-]+", tag)
    OUT.mkdir(exist_ok=True)
    receipt = OUT / (tag + "-build.json")
    assert not receipt.exists()
    k = ctypes.WinDLL("kernel32")
    k.CreateMutexW.restype = ctypes.c_void_p
    k.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    k.ReleaseMutex.argtypes = [ctypes.c_void_p]
    k.CloseHandle.argtypes = [ctypes.c_void_p]
    handles = []
    reservation = None
    record = {"arm": arm, "started_at": lab.now(), "phases_seconds": {}}
    config = lab.sha(ROOT / "broker/config.json")
    try:
        for name in MUTEXES:
            h = k.CreateMutexW(None, False, "Local\\" + name)
            if k.WaitForSingleObject(h, 0) not in (0, 128):
                k.CloseHandle(h)
                raise RuntimeError("Laboratory mutex busy: " + name)
            handles.append(h)
        record["admission"] = lab.admission()
        lab.OWNER = "ninfer-switch-matrix-build"
        reservation = uuid.uuid4().hex
        granted = lab.api("/v1/photo-gpu-reservations",
                          {"owner": lab.OWNER, "reservation_id": reservation,
                           "ttl_seconds": 3600}, "POST", 90)
        assert granted["reservation_id"] == reservation and granted["state"] == "active"
        record["reservation_id"] = reservation
        record["owned_admission"] = lab.admission(True)
        builder = json.loads(lab.docker("inspect", BUILDER).stdout)[0]
        assert builder["Image"] == "sha256:501c92a90c3d85adc616455a6783bb5ae8b82008f469cf8dc3d6da538f75475b"
        assert not builder["HostConfig"].get("DeviceRequests")
        record["builder_id"] = builder["Id"]
        top = lab.docker("top", BUILDER, "-eo", "pid,args").stdout.splitlines()[1:]
        assert [s.split(None, 1)[1] for s in top if s.strip()] == ["sleep infinity"], top
        head = lab.run(["git", "-C", str(SOURCE), "rev-parse", "HEAD"]).stdout.strip()
        files = lab.run(["git", "-C", str(SOURCE), "ls-files", "-z"]).stdout.split("\0")
        files = [s for s in files if s]
        hashes = {f: lab.sha(SOURCE / f) for f in files}
        record["source_head"] = head
        record["source_manifest_sha256"] = hashlib.sha256(
            json.dumps(hashes, sort_keys=True).encode()).hexdigest()
        record["source_files"] = hashes
        sync = """
import sys,json,pathlib,hashlib
expected=json.load(sys.stdin)
source=pathlib.Path('/lab/source')
destination=pathlib.Path('/src')
changed=[]
extras={}
for path in destination.rglob('*'):
    if not path.is_file() or '.git' in path.relative_to(destination).parts:
        continue
    name=path.relative_to(destination).as_posix()
    if name not in expected:
        assert name.startswith('provenance/') or '__pycache__' in path.parts,name
        extras[name]=hashlib.sha256(path.read_bytes()).hexdigest()
for name,sha in expected.items():
    src=(source/name).resolve()
    dst=(destination/name).resolve()
    assert src.is_relative_to(source) and dst.is_relative_to(destination)
    data=src.read_bytes()
    assert hashlib.sha256(data).hexdigest()==sha,name
    if not dst.is_file() or dst.read_bytes()!=data:
        dst.parent.mkdir(parents=True,exist_ok=True)
        dst.write_bytes(data)
        changed.append(name)
for name,sha in expected.items():
    assert hashlib.sha256((destination/name).read_bytes()).hexdigest()==sha,name
print(json.dumps({'changed':changed,'nonbuild_extras':extras}))
"""
        started = time.monotonic()
        p = subprocess.run(
            ["wsl.exe", "-d", "Ubuntu-24.04", "--exec", "/usr/bin/docker",
             "exec", "-i", BUILDER, "python3", "-c", sync],
            input=json.dumps(hashes), capture_output=True, text=True,
            creationflags=subprocess.CREATE_NO_WINDOW, timeout=180, check=True)
        record["source_sync"] = json.loads(p.stdout)
        record["phases_seconds"]["sync"] = time.monotonic() - started
        options = set()
        for f in files:
            if f.endswith(".cmake") or pathlib.PurePosixPath(f).name == "CMakeLists.txt":
                options.update(re.findall(r"option\(\s*(NINFER_\w+)",
                                          (SOURCE / f).read_text(encoding="utf-8")))
        on = {"NINFER_BUILD_APPS", "NINFER_BUILD_BENCHMARKS",
              "NINFER_GGUF_Q4_COOPERATIVE_DECODE",
              "NINFER_GGUF_Q6_COOPERATIVE_DECODE", "NINFER_GGUF_MMQ_STORE_FUSION"}
        if VARIANTS[arm][0]:
            on.add("NINFER_GGUF_DP4A_SHORT_CHAINS")
        if VARIANTS[arm][1]:
            on.add("NINFER_GGUF_Q6_MMQ_A_X4")
        assert on <= options, on - options
        flags = ["-D" + name + "=" + ("ON" if name in on else "OFF")
                 for name in sorted(options)]
        record["options"] = {name: name in on for name in sorted(options)}
        started = time.monotonic()
        p = lab.docker("exec", BUILDER, "cmake", "-S", "/src", "-B", "/build",
                       "-DCMAKE_CUDA_ARCHITECTURES=86", "-DBUILD_TESTING=ON",
                       "-DCMAKE_BUILD_TYPE=Release", *flags, timeout=180, check=False)
        (OUT / (tag + "-configure.log")).write_text(p.stdout + p.stderr, encoding="utf-8")
        assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
        record["phases_seconds"]["configure"] = time.monotonic() - started
        cache = lab.docker("exec", BUILDER, "cat", "/build/CMakeCache.txt").stdout
        actual = dict(re.findall(r"^(NINFER_\w+):BOOL=(ON|OFF)$", cache, re.M))
        assert all(actual[name] == ("ON" if name in on else "OFF") for name in options)
        assert not {name for name, value in actual.items() if value == "ON"} - on
        ninja = lab.docker("exec", BUILDER, "cat", "/build/build.ninja").stdout
        seeds = ["ninfer-ggml_vec_q4_k", "ninfer-ggml_mmq_q6_k"]
        assert all(ninja.count("--frandom-seed=" + seed) == 1 for seed in seeds)
        record["cuda_symbol_seeds"] = seeds
        effective = set(re.findall(r"(?:-D|/D)(NINFER_\w+)(?:=\S+)?", ninja))
        permitted = on | {"NINFER_SM8X_COMPAT", "NINFER_PYTHON_EXECUTABLE",
                          "NINFER_SOURCE_DIR", "NINFER_TEST_LINK_WRAP"}
        if VARIANTS[arm][1]:
            permitted.add("NINFER_GGUF_Q6_MMQ_A_X4_TEST")
        assert not effective - permitted, sorted(effective - permitted)
        assert not re.search(r"-UNINFER_", ninja)
        record["effective_ninja_defines"] = sorted(effective)
        started = time.monotonic()
        print("Building " + arm + " with explicit W/Q4/Q6 flags; source synchronized.", flush=True)
        command = ["wsl.exe", "-d", "Ubuntu-24.04", "--exec", "/usr/bin/docker",
                   "exec", BUILDER, "timeout", "--kill-after=15s", "2400s",
                   "cmake", "--build", "/build", "--target",
                   "ninfer-serve", "ninfer_tests", "-j", "4"]
        with (OUT / (tag + "-build.log")).open("x", encoding="utf-8") as log:
            p = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                               creationflags=subprocess.CREATE_NO_WINDOW)
        assert p.returncode == 0, "Build failed; inspect log"
        record["phases_seconds"]["build"] = time.monotonic() - started
        lab.admission(True)
        assert all(lab.sha(SOURCE / f) == sha for f, sha in hashes.items())
        started = time.monotonic()
        exported = {}
        for name, path in [("ninfer-serve", "/build/apps/ninfer-serve"),
                           ("ninfer_tests", "/build/tests/ninfer_tests"),
                           ("CMakeCache.txt", "/build/CMakeCache.txt"),
                           ("build.ninja", "/build/build.ninja")]:
            expected = lab.docker("exec", BUILDER, "sha256sum", path).stdout.split()[0]
            target = ROOT / "artifacts/experimental" / name
            temporary = target.with_suffix(target.suffix + ".new")
            assert not temporary.exists()
            lab.docker("cp", BUILDER + ":" + path,
                       lab.WSL_ROOT + "/artifacts/experimental/" + temporary.name, timeout=120)
            assert lab.sha(temporary) == expected
            temporary.replace(target)
            exported[name] = expected
        record["phases_seconds"]["export"] = time.monotonic() - started
        record["export_sha256"] = exported
        record["status"] = "BUILT_EXPORTED_NOT_GPU_QUALIFIED"
        (ROOT / "artifacts/experimental/build-receipt.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8")
        (ROOT / "artifacts/experimental/export-manifest.json").write_text(
            json.dumps({"schema": "switch-matrix-export-v1", "arm": arm,
                        "source_manifest_sha256": record["source_manifest_sha256"],
                        "files": exported}, indent=2), encoding="utf-8")
        print(json.dumps({"status": record["status"], "phases": record["phases_seconds"],
                          "export": exported}), flush=True)
    except BaseException as exc:
        record["status"] = "INCOMPLETE"
        record["failure"] = type(exc).__name__ + ": " + str(exc)
        raise
    finally:
        if reservation:
            try:
                top = lab.docker("top", BUILDER, "-eo", "pid,args").stdout.splitlines()[1:]
                assert [s.split(None, 1)[1] for s in top if s.strip()] == ["sleep infinity"]
                released = lab.api("/v1/photo-gpu-reservations/" + reservation,
                                   {"owner": lab.OWNER}, "DELETE")
                assert released["reservation_id"] == reservation and released["state"] == "released"
                record["reservation_released"] = True
            except BaseException as exc:
                record["cleanup_failure"] = str(exc)
        record["production_config_unchanged"] = lab.sha(ROOT / "broker/config.json") == config
        record["finished_at"] = lab.now()
        with receipt.open("x", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
        for h in reversed(handles):
            k.ReleaseMutex(h)
            k.CloseHandle(h)
    if record.get("cleanup_failure"):
        raise RuntimeError(record["cleanup_failure"])


if __name__ == "__main__":
    main()
