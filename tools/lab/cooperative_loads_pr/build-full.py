import ctypes, hashlib, json, subprocess, time
from pathlib import Path
ROOT=Path(r"C:\Users\aleksei\Projects\NInfer")
OUT=ROOT/"results/cooperative-loads-pr"
VEC=ROOT/"source/src/ops/linear/gguf/ggml_bridge_vec.cuh"
PREFIX=["wsl.exe","-d","Ubuntu-24.04","--cd","/","--exec","docker","exec","ninfer-builder"]
record={"status":"BUILDING_FULL","commands":[],"production_config_sha256":hashlib.sha256((ROOT/"broker/config.json").read_bytes()).hexdigest()}
def save(): (OUT/"full-build.json").write_text(json.dumps(record,indent=2))
def run(args,label):
    start=time.monotonic()
    with (OUT/(label+".log")).open("wb") as log:
        p=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
    record["commands"].append({"label":label,"exit":p.returncode,"seconds":time.monotonic()-start});save()
    if p.returncode: raise RuntimeError(label+" failed")
k=ctypes.WinDLL("kernel32",use_last_error=True)
k.CreateMutexW.argtypes=[ctypes.c_void_p,ctypes.c_bool,ctypes.c_wchar_p];k.CreateMutexW.restype=ctypes.c_void_p
k.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_uint32];k.ReleaseMutex.argtypes=[ctypes.c_void_p]
h=k.CreateMutexW(None,False,"Local\\NInferV14CpuVariantPreparation")
if not h or k.WaitForSingleObject(h,0) not in (0,128):raise RuntimeError("another build owns cache")
try:
    for arm in ("control","candidate"):
        if arm=="control":
            VEC.write_bytes(subprocess.check_output(["git","-C",str(ROOT/"source"),"show","56aa5f5ba5b13cbc5eba52d34e1c92bc34078d01:src/ops/linear/gguf/ggml_bridge_vec.cuh"],creationflags=subprocess.CREATE_NO_WINDOW))
        else: VEC.write_bytes(subprocess.check_output(["git","-C",str(ROOT/"source"),"show","b42619ae6a552b5072ad05e3e893e87d8dc92c32:src/ops/linear/gguf/ggml_bridge_vec.cuh"],creationflags=subprocess.CREATE_NO_WINDOW))
        run(PREFIX+["cmake","-S","/lab/source","-B","/build","-G","Ninja","-DCMAKE_BUILD_TYPE=Release","-DCMAKE_CUDA_ARCHITECTURES=86","-DNINFER_BUILD_APPS=ON","-DBUILD_TESTING=ON","-DNINFER_BUILD_BENCHMARKS=ON","-DPR_COOP="+("ON" if arm=="candidate" else "OFF"),"-DCMAKE_PROJECT_INCLUDE=/lab/results/cooperative-loads-pr/full-target.cmake"],arm+"-full-configure")
        run(PREFIX+["cmake","--build","/build","--target","ninfer_bench","ninfer","ninfer_pr_check","-j","8"],arm+"-full-compile")
        run(PREFIX+["cp","/build/ninfer_pr_check","/lab/results/cooperative-loads-pr/bin/"+arm],arm+"-copy")
        run(PREFIX+["cp","/build/bench/ninfer_bench","/lab/results/cooperative-loads-pr/bin/"+arm+"-decode"],arm+"-decode-copy")
        run(PREFIX+["cp","/build/apps/ninfer","/lab/results/cooperative-loads-pr/bin/"+arm+"-cli"],arm+"-cli-copy")
        record[arm]={"sha256":hashlib.sha256((OUT/"bin"/arm).read_bytes()).hexdigest(),"bytes":(OUT/"bin"/arm).stat().st_size};save()
    record["status"]="PASS_FULL_BUILDS"
except BaseException as e:
    record.update(status="FAILED",error=str(e))
finally:
    expected=subprocess.check_output(["git","-C",str(ROOT/"source"),"show","b42619ae6a552b5072ad05e3e893e87d8dc92c32:src/ops/linear/gguf/ggml_bridge_vec.cuh"],creationflags=subprocess.CREATE_NO_WINDOW)
    if VEC.read_bytes()!=expected: VEC.write_bytes(expected)
    k.ReleaseMutex(h);save()
