import ctypes,hashlib,json,runpy,subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parents[2];out=root/'results/gate-bf16-pr';arm="control"
receipt=out/'control-full-build.json';assert not receipt.exists()
h=runpy.run_path(str(root/'scripts/measure-cooperative-decode.py'));h['resources']()
k=ctypes.WinDLL('kernel32',use_last_error=True);k.CreateMutexW.argtypes=[ctypes.c_void_p,ctypes.c_bool,ctypes.c_wchar_p];k.CreateMutexW.restype=ctypes.c_void_p;k.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_uint32];k.ReleaseMutex.argtypes=[ctypes.c_void_p];k.CloseHandle.argtypes=[ctypes.c_void_p]
locks=[]
for name in ('Local\\NInferV17RuntimeScreen','Local\\NInferV14RuntimeMeasurement','Local\\NInferV15ExportedOperatorSeries','Local\\NInferV14CpuVariantPreparation'):
 v=k.CreateMutexW(None,False,name)
 if not v or k.WaitForSingleObject(v,0) not in (0,128):raise RuntimeError('another lab worker')
 locks.append(v)
r={'status':'BUILDING','arm':arm,'commands':[],'base':h['run'](['git','-C',str(root/'source'),'rev-parse','HEAD'])}
prefix=['wsl.exe','-d','Ubuntu-24.04','--cd','/','--exec','docker','exec','ninfer-builder']
def run(args,label):
 start=time.monotonic()
 with (out/(label+'.log')).open('wb') as log:p=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
 r['commands'].append({'label':label,'exit':p.returncode,'seconds':time.monotonic()-start})
 receipt.write_text(json.dumps(r,indent=2),encoding='utf-8')
 if p.returncode:raise RuntimeError(label+' failed')
try:
 run(prefix+['cmake','-S','/lab/source','-B','/build','-G','Ninja','-DCMAKE_BUILD_TYPE=Release','-DCMAKE_CUDA_ARCHITECTURES=86','-DNINFER_BUILD_APPS=ON','-DBUILD_TESTING=ON','-DNINFER_BUILD_BENCHMARKS=ON','-DCMAKE_PROJECT_INCLUDE=/lab/results/gate-bf16-pr/targets.cmake'],arm+'-configure')
 targets=['ninfer_bench','ninfer']
 run(prefix+['cmake','--build','/build','--target',*targets,'-j','8'],arm+'-compile')
 paths={}
 paths.update({'-cli':'/build/apps/ninfer','-decode':'/build/bench/ninfer_bench'})
 r['binaries']={}
 for suffix,source in paths.items():
  run(prefix+['cp',source,'/lab/results/gate-bf16-pr/bin/'+arm+suffix],arm+suffix+'-export')
  p=out/'bin'/(arm+suffix);r['binaries'][arm+suffix]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
 r['status']='PASS_BUILD'
except BaseException as e:r.update(status='FAILED',error=str(e));raise
finally:
 receipt.write_text(json.dumps(r,indent=2),encoding='utf-8')
 for v in reversed(locks):k.ReleaseMutex(v);k.CloseHandle(v)
