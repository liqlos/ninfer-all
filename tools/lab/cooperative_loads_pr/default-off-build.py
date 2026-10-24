import ctypes,hashlib,json,runpy,subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parents[2];out=root/'results/cooperative-loads-pr'
h=runpy.run_path(str(root/'scripts/measure-cooperative-decode.py'))
h['resources']()
k=ctypes.WinDLL('kernel32',use_last_error=True);k.CreateMutexW.argtypes=[ctypes.c_void_p,ctypes.c_bool,ctypes.c_wchar_p];k.CreateMutexW.restype=ctypes.c_void_p;k.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_uint32];k.ReleaseMutex.argtypes=[ctypes.c_void_p];k.CloseHandle.argtypes=[ctypes.c_void_p]
lock=k.CreateMutexW(None,False,'Local\\NInferV14CpuVariantPreparation')
if not lock or k.WaitForSingleObject(lock,0) not in (0,128):raise RuntimeError('another CPU worker')
receipt=out/'default-off-build.json'
assert not receipt.exists()
r={'status':'RUNNING','commands':[]}
prefix=['wsl.exe','-d','Ubuntu-24.04','--cd','/','--exec','docker','exec','ninfer-builder']
def run(args,label):
 start=time.monotonic()
 with (out/(label+'.log')).open('wb') as log:p=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
 r['commands'].append({'label':label,'exit':p.returncode,'seconds':time.monotonic()-start})
 if p.returncode:raise RuntimeError(label+' failed')
try:
 run(prefix+['cmake','-S','/lab/source','-B','/build','-G','Ninja','-DCMAKE_BUILD_TYPE=Release','-DCMAKE_CUDA_ARCHITECTURES=86','-UNINFER_GGUF_COOPERATIVE_LOADS','-DPR_COOP=ON','-DCMAKE_PROJECT_INCLUDE=/lab/results/cooperative-loads-pr/default-target.cmake'],'default-off-configure')
 cache=h['run'](prefix+['cat','/build/CMakeCache.txt'])
 assert 'NINFER_GGUF_COOPERATIVE_LOADS:BOOL=OFF' in cache
 r['cmake_option']='OFF from empty option cache'
 run(prefix+['cmake','--build','/build','--target','ninfer_pr_check','-j','8'],'default-off-compile')
 run(prefix+['cp','/build/ninfer_pr_check','/lab/results/cooperative-loads-pr/bin/default-off'],'default-off-export')
 p=out/'bin/default-off';r['binary']={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
 r['status']='PASS_OFF_BUILD'
except BaseException as e:r.update(status='FAILED',error=str(e));raise
finally:
 receipt.write_text(json.dumps(r,indent=2));k.ReleaseMutex(lock);k.CloseHandle(lock)
subprocess.run([sys.executable,'-B',str(root/'scripts/qualify-cooperative-loads.py'),'tests','default-off-gates','default-off'],cwd=root,check=True,creationflags=subprocess.CREATE_NO_WINDOW)
