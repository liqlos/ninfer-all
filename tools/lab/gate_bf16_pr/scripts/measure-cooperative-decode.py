import ctypes, hashlib, json, re, runpy, shutil, subprocess, sys, time, uuid
from pathlib import Path
H=runpy.run_path(str(Path(__file__).with_name("qualify-bf16-store-gpu.py")))
OWNER="ninfer-cooperative-decode-pr"
H["admission"].__globals__["OWNER"]=OWNER
ROOT,WSL,IMAGE=(H[k] for k in ("ROOT","WSL","IMAGE"))
api,run,admission=(H[k] for k in ("api","run","admission"))
OUT=ROOT/"results/cooperative-loads-pr"
MOUNT="/mnt/c/Users/aleksei/Projects/NInfer/results/cooperative-loads-pr"
class Memory(ctypes.Structure):
    _fields_=[("length",ctypes.c_ulong),("load",ctypes.c_ulong)]+[(x,ctypes.c_ulonglong) for x in ("total_phys","avail_phys","total_page","avail_page","total_virtual","avail_virtual","extended")]
def resources(reservation=None):
    m=Memory();m.length=ctypes.sizeof(m)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):raise RuntimeError("memory telemetry unavailable")
    deadline=time.monotonic()+60
    while m.avail_page<40*1024**3:
        if time.monotonic()>=deadline:raise RuntimeError("full model host guard")
        time.sleep(3)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):raise RuntimeError("memory telemetry unavailable")
    if shutil.disk_usage("C:/").free<8*1024**3:raise RuntimeError("full model disk guard")
    code="Get-CimInstance Win32_PerfFormattedData_GPUPerformanceCounters_GPUProcessMemory | Where-Object {$_.DedicatedUsage -gt 64MB} | ForEach-Object { if($_.Name -match 'pid_(\\d+)_') { $p=Get-Process -Id ([int]$Matches[1]) -ErrorAction SilentlyContinue; $display=($p.ProcessName -eq 'dwm') -or ($p.Path -ieq 'C:\\Windows\\Explorer.EXE') -or ($p.Path -ieq 'C:\\Program Files\\WindowsApps\\OpenAI.Codex_26.1016.8622.0_x64__2p2nqsd0c76g0\\app\\ChatGPT.exe'); if((-not $display) -or $_.DedicatedUsage -gt 256MB) { $p.ProcessName } } }"
    result=run(["powershell.exe","-NoProfile","-NonInteractive","-Command",code])
    if result:raise RuntimeError("foreign GPU work: "+result)
    s=admission(reservation)
    if s["capacity_estimate"][0]["available_llm_budget_bytes"]<23000000000:raise RuntimeError("full model GPU budget insufficient")
    return {"free_commit":m.avail_page,"admission":s}
def main():
    mode,tag=sys.argv[1:]
    receipt=OUT/(tag+".json")
    if receipt.exists():raise RuntimeError("preserve prior series")
    build=json.loads((OUT/"full-build.json").read_text())
    if build["status"]!="PASS_FULL_BUILDS":raise RuntimeError("full build incomplete")
    orders=["ABBA","BAAB"]*(4 if mode=="confirm" else 1)
    if mode=="cli":orders=["AB"]
    points=["4096,256"]
    record={"status":"RUNNING","runs":[],"protocol":{"orders":orders,"prompt_gen":points,"repetitions":1,"warmup":1,"max_context":65536,"prefill_chunk":1024,"kv":"int8","spec":"mtp3","ngram_draft_tokens":0,"scope":"public Engine native benchmark; load excluded","primary":"decode_seconds ratio control/candidate; exploratory screen requires >0.3 percent speed gain and both block ratios positive before fresh fixed 8-block confirmation"}}
    config_sha=hashlib.sha256((ROOT/"broker/config.json").read_bytes()).hexdigest()
    k=ctypes.WinDLL("kernel32",use_last_error=True)
    k.CreateMutexW.argtypes=[ctypes.c_void_p,ctypes.c_bool,ctypes.c_wchar_p];k.CreateMutexW.restype=ctypes.c_void_p
    k.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_uint32];k.ReleaseMutex.argtypes=[ctypes.c_void_p];k.CloseHandle.argtypes=[ctypes.c_void_p]
    locks=[];reservation=None;containers=[];cli_tokens=None
    def save():receipt.write_text(json.dumps(record,indent=2))
    try:
        for name in ("Local\\NInferV17RuntimeScreen","Local\\NInferV14RuntimeMeasurement","Local\\NInferV15ExportedOperatorSeries","Local\\NInferV14CpuVariantPreparation"):
            h=k.CreateMutexW(None,False,name)
            if not h or k.WaitForSingleObject(h,0) not in (0,128):raise RuntimeError("another lab worker owns "+name)
            locks.append(h)
        record["admission_before"]=resources()
        suffix="-cli" if mode=="cli" else "-decode"
        binding=json.loads((OUT/"binding.json").read_text())
        record["binaries"]={a:{"sha256":hashlib.sha256((OUT/"bin"/(a+suffix)).read_bytes()).hexdigest(),"bytes":(OUT/"bin"/(a+suffix)).stat().st_size} for a in ("control","candidate")}
        if any(record["binaries"][a]!=binding["binaries"][a+suffix] for a in record["binaries"]):raise RuntimeError("exported binary binding changed")
        record["model"]=json.loads((OUT/"model-compatibility.json").read_text())
        if record["model"]["status"]!="PASS":raise RuntimeError("model metadata not qualified")
        record["corpus_sha256"]=hashlib.sha256((ROOT/"source/bench/fixtures/bench_corpus.ids").read_bytes()).hexdigest()
        model_mount=MOUNT+"/model"
        if mode!="cli":
            model_mount="/root/ninfer-stage/cooperative-pr-model"
            staging="import pathlib,shutil,hashlib,os; p=pathlib.Path('/root/ninfer-stage/cooperative-pr-model'); p.mkdir(exist_ok=True); src=pathlib.Path('"+MOUNT+"/model/qwen38-jonathan-q4km-proposal-vision-q.ninfer'); dst=p/src.name; expected='"+record["model"]["qualified_sha256"]+"'; assert src.exists() or dst.exists(); shutil.copyfile(src,dst) if not dst.exists() else None; h=hashlib.sha256(); f=dst.open('rb'); [h.update(x) for x in iter(lambda:f.read(8*1024*1024),b'')]; os.posix_fadvise(f.fileno(),0,0,os.POSIX_FADV_DONTNEED); f.close(); assert h.hexdigest()==expected; dst.chmod(0o444); print(h.hexdigest())"
            record["staged_model_sha256"]=run(WSL[:-1]+["python3","-c",staging],timeout=300)
            record["runtime_model_path"]=model_mount
            local_model=OUT/"model/qwen38-jonathan-q4km-proposal-vision-q.ninfer"
            if local_model.exists():local_model.unlink()
        reservation=uuid.uuid4().hex
        grant=api("/v1/photo-gpu-reservations","POST",{"owner":OWNER,"reservation_id":reservation,"ttl_seconds":3600})
        if grant["reservation_id"]!=reservation or grant["state"]!="active":raise RuntimeError("reservation mismatch")
        record["reservation"]=grant;save()
        for block,order in enumerate(orders):
            for position,choice in enumerate(order):
                arm="control" if choice=="A" else "candidate"
                if record["runs"]: time.sleep(6)
                resources(reservation)
                label=f"{tag}-b{block}-p{position}-{arm}"
                name="ninfer-coop-"+uuid.uuid4().hex[:12]
                argv=WSL+["run","-d","--name",name,"--network","none","--gpus","all","--ipc","host","--label","io.codex.ninfer-gpu=true","--label","io.codex.ninfer-owner="+OWNER,
                    "--mount","type=bind,source="+MOUNT+"/bin,target=/bin-pr,readonly",
                    "--mount","type=bind,source="+MOUNT+",target=/results",
                    "--mount","type=bind,source="+model_mount+",target=/model,readonly",
                    "--mount","type=bind,source=/mnt/c/Users/aleksei/Projects/NInfer/source/bench/fixtures/bench_corpus.ids,target=/corpus.ids,readonly",
                    "--entrypoint","/bin-pr/"+arm+"-decode",IMAGE,
                    "--weights","/model/qwen38-jonathan-q4km-proposal-vision-q.ninfer","--corpus","/corpus.ids","--prompt-gen",";".join(points),"--repetitions","1","--warmup","1","--max-ctx","65536","--prefill-chunk","1024","--kv-dtype","int8","--spec","mtp","--draft-tokens","3","--ngram-draft-tokens","0","--output","json","--output-file","/results/"+label+".json"]
                if mode=="cli":
                    at=argv.index("--entrypoint")
                    argv=argv[:at]+["--entrypoint","/bin-pr/"+arm+"-cli",IMAGE,"/model/qwen38-jonathan-q4km-proposal-vision-q.ninfer",
                        "--prompt","Write a detailed explanation of how a computer cache works, with examples.",
                        "--max-context","65536","--kv-capacity","65536","--prefill-chunk","1024","--kv-dtype","int8","--spec","mtp","--draft-tokens","3","--ngram-draft-tokens","0","--max-new","128","--temperature","0","--seed","173","--no-thinking","--raw-output","--print-token-ids"]
                start=time.monotonic();container=run(argv);containers.append(container)
                code=run(WSL+["wait",container],timeout=1500)
                log=run(WSL+["logs",container]);(OUT/(label+".log")).write_text(log)
                run(WSL+["rm",container]);containers.remove(container)
                if mode!="cli":
                    release="import os,subprocess; assert not subprocess.check_output(['docker','ps','-q','--filter','label=io.codex.ninfer-gpu=true']).strip(); p='/root/ninfer-stage/cooperative-pr-model/qwen38-jonathan-q4km-proposal-vision-q.ninfer'; assert os.path.realpath(p)==p; f=os.open(p,os.O_RDONLY); assert os.fstat(f).st_size==17695430144; os.posix_fadvise(f,0,0,os.POSIX_FADV_DONTNEED); os.close(f)"
                    run(WSL[:-1]+["python3","-c",release])
                if code!="0":raise RuntimeError(label+" failed: "+log[-1500:])
                if mode=="cli":
                    match=re.search(r"^tokens\s+generated ids\s+([0-9 ]+)\s*$",log,re.M)
                    if not match:raise RuntimeError("CLI did not expose token IDs")
                    ids=tuple(map(int,match.group(1).split()))
                    if not ids or (cli_tokens is not None and ids!=cli_tokens):raise RuntimeError("CLI token IDs differ")
                    cli_tokens=ids
                    record["runs"].append({"arm":arm,"label":label,"generated_tokens":len(ids),"token_sha256":hashlib.sha256(json.dumps(ids).encode()).hexdigest()})
                    save();continue
                result=json.loads((OUT/(label+".json")).read_text())
                if len(result["tests"]) != 1:
                    raise RuntimeError("unexpected test count")
                test=result["tests"][0]
                if test["n_prompt"]!=4096 or test["n_gen"]!=256 or len(test["reps"])!=1:
                    raise RuntimeError("decode workload mismatch")
                rep=test["reps"][0]
                signature={"generated_output_tokens":rep["generated_output_tokens"],"decode_output_tokens":rep["decode_output_tokens"],"decode_engine_tokens":rep["decode_engine_tokens"],"speculative":rep["speculative"]}
                if record["runs"] and signature != record["work_signature"]:
                    raise RuntimeError("MTP work changed between arms/runs")
                record["work_signature"]=signature
                record["runs"].append({"arm":arm,"block":block,"position":position,"label":label,"wall_seconds":time.monotonic()-start,"report":result});save()
        record["status"]="PASS"
    except BaseException as e:
        record.update(status="FAILED",error=str(e));raise
    finally:
        errors=[]
        for container in containers:
            try:run(WSL+["rm","-f",container])
            except Exception as e:errors.append(str(e))
        if reservation:
            try:record["release"]=api("/v1/photo-gpu-reservations/"+reservation,"DELETE",{"owner":OWNER})
            except Exception as e:errors.append(str(e))
        record["cleanup_errors"]=errors
        record["production_config_unchanged"]=hashlib.sha256((ROOT/"broker/config.json").read_bytes()).hexdigest()==config_sha
        save()
        for h in reversed(locks):k.ReleaseMutex(h);k.CloseHandle(h)
if __name__=="__main__":main()
