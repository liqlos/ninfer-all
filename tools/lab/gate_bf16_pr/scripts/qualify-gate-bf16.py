import csv,ctypes,hashlib,json,runpy,sqlite3,subprocess,sys,time,uuid
from pathlib import Path
root=Path(__file__).resolve().parents[1];out=root/'results/gate-bf16-pr';mount='/mnt/c/Users/aleksei/Projects/NInfer/results/gate-bf16-pr'
h=runpy.run_path(str(root/'scripts/qualify-bf16-store-gpu.py'));owner='ninfer-gate-bf16-pr';h['admission'].__globals__['OWNER']=owner
api,run,admission,WSL,IMAGE=(h[k] for k in ('api','run','admission','WSL','IMAGE'))
mode,tag=sys.argv[1:];receipt=out/(tag+'.json');assert not receipt.exists()
info={a:json.loads((out/(a+'-build.json')).read_text()) for a in ('control','candidate')}
assert all(x['status']=='PASS_BUILD' for x in info.values())
assert info['control']['base']==info['candidate']['base']
for a in info:assert hashlib.sha256((out/'bin'/a).read_bytes()).hexdigest()==info[a]['binaries'][a]['sha256']
k=ctypes.WinDLL('kernel32',use_last_error=True);k.CreateMutexW.argtypes=[ctypes.c_void_p,ctypes.c_bool,ctypes.c_wchar_p];k.CreateMutexW.restype=ctypes.c_void_p;k.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_uint32];k.ReleaseMutex.argtypes=[ctypes.c_void_p];k.CloseHandle.argtypes=[ctypes.c_void_p]
locks=[];containers=[];reservation=None
r={'status':'RUNNING','mode':mode,'runs':[],'builds':info,'production_config_sha256':hashlib.sha256((root/'broker/config.json').read_bytes()).hexdigest()}
def save():receipt.write_text(json.dumps(r,indent=2),encoding='utf-8')
def execute(arm,label,args,profile=False):
 admission(reservation)
 name='ninfer-gate-'+uuid.uuid4().hex[:12]
 mounts=['--mount','type=bind,source='+mount+'/bin,target=/bin-pr,readonly','--mount','type=bind,source='+mount+',target=/results']
 argv=['--entrypoint','/bin-pr/'+arm,IMAGE,*args]
 if profile:
  mounts+=['--mount','type=bind,source=/mnt/c/Users/aleksei/Projects/NInfer/tools/nsys,target=/nsys/target-linux-x64,readonly']
  argv=['--entrypoint','/bin/bash',IMAGE,'-c','ln -s /nsys/target-linux-x64/nsys /tmp/ninfer-nsys; exec "$@"','profile','/tmp/ninfer-nsys','profile','--trace=cuda','--sample=none','--cpuctxsw=none','-o','/results/'+label,'/bin-pr/'+arm,*args]
 container=run(WSL+['run','-d','--name',name,'--network','none','--gpus','all','--label','io.codex.ninfer-gpu=true','--label','io.codex.ninfer-owner='+owner,*mounts,*argv]);containers.append(container)
 code=run(WSL+['wait',container],timeout=900);log=run(WSL+['logs',container]);(out/(label+'.log')).write_text(log,encoding='utf-8')
 run(WSL+['rm',container]);containers.remove(container)
 if code!='0':raise RuntimeError(label+' failed: '+log[-1200:])
 return {'arm':arm,'label':label,'exit':code}
try:
 for name in ('Local\\NInferV17RuntimeScreen','Local\\NInferV14RuntimeMeasurement','Local\\NInferV15ExportedOperatorSeries','Local\\NInferV14CpuVariantPreparation'):
  v=k.CreateMutexW(None,False,name)
  if not v or k.WaitForSingleObject(v,0) not in (0,128):raise RuntimeError('another lab worker')
  locks.append(v)
 r['admission_before']=admission()
 res=runpy.run_path(str(root/'scripts/measure-cooperative-decode.py'))
 res['resources'].__globals__['admission']=admission
 res['resources']()
 reservation=uuid.uuid4().hex;r['reservation']=api('/v1/photo-gpu-reservations','POST',{'owner':owner,'reservation_id':reservation,'ttl_seconds':3600})
 assert r['reservation']['state']=='active' and r['reservation']['reservation_id']==reservation;save()
 if mode=='gates':
  for arm in ('control','candidate'):
   for command in ('oracle','store'):
    r['runs'].append(execute(arm,tag+'-'+arm+'-'+command,[command]));save()
 elif mode=='profile':
  r['profiles']=[]
  for arm in ('control','candidate'):
   label=tag+'-'+arm
   row=execute(arm,label,['swiglu','--t-sweep','1024','--warmup','30','--profile'],True)
   prefix='/lab/results/gate-bf16-pr/'+label
   run(WSL+['exec','ninfer-builder','/opt/nvidia/nsight-compute/2025.4.1/host/linux-desktop-glibc_2_11_3-x64/QdstrmImporter','-i',prefix+'.qdstrm','-o',prefix+'.nsys-rep'],timeout=180)
   run(WSL+['exec','ninfer-builder','/lab/tools/nsys/nsys','export','--type=sqlite','--output='+prefix+'.sqlite',prefix+'.nsys-rep'],timeout=180)
   db=sqlite3.connect(out/(label+'.sqlite'))
   row['kernels']=[dict(name=n,nanoseconds=ns,registers=regs,grid=grid,block=block) for n,ns,regs,grid,block in db.execute('SELECT s.value,k.end-k.start,k.registersPerThread,k.gridX,k.blockX*k.blockY*k.blockZ FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName ORDER BY k.start')]
   db.close();r['profiles'].append(row);save()
   for suffix in ('.qdstrm','.nsys-rep','.sqlite'):(out/(label+suffix)).unlink()
 else:
  orders=['ABBA','BAAB']*(4 if mode=='confirm' else 1)
  r['protocol']={'orders':orders,'t':[16,128,512,1024],'warmup':30,'repeat':150,'execution':'eager','flush_mib':256,'primary':'public linear_swiglu T1024 speed ratio; other T are secondary/regression screens','scope':'public GGUF_Q4_K linear_swiglu, stacked parent N34816,K5120 -> N17408,includes packing,both projections,activation/final store','no_profiler_in_timing':True};save()
  for block,order in enumerate(orders):
   for position,choice in enumerate(order):
    arm='control' if choice=='A' else 'candidate';label=f'{tag}-b{block}-p{position}-{arm}'
    if r['runs']:time.sleep(2)
    row=execute(arm,label,['swiglu','--t-sweep','16,128,512,1024','--warmup','30','--repeat','150','--execution','eager','--csv-out','/results/'+label+'.csv'])
    with (out/(label+'.csv')).open(newline='') as f:data=list(csv.DictReader(f))
    assert [int(x['t']) for x in data]==[16,128,512,1024]
    row.update(block=block,position=position,points=data);r['runs'].append(row);save()
 r['status']='PASS'
except BaseException as e:r.update(status='FAILED',error=str(e));raise
finally:
 errors=[]
 for container in containers:
  try:run(WSL+['rm','-f',container])
  except Exception as e:errors.append(str(e))
 if reservation:
  try:r['release']=api('/v1/photo-gpu-reservations/'+reservation,'DELETE',{'owner':owner})
  except Exception as e:errors.append(str(e))
 r['cleanup_errors']=errors;r['production_config_unchanged']=hashlib.sha256((root/'broker/config.json').read_bytes()).hexdigest()==r['production_config_sha256'];save()
 for v in reversed(locks):k.ReleaseMutex(v);k.CloseHandle(v)
print(r['status'],len(r['runs']),flush=True)
