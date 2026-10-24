import hashlib,json,shutil,sys,urllib.request
from pathlib import Path
sys.path.insert(0,'source')
from tools.artifact.reader import Artifact
root=Path.cwd();out=root/'results/cooperative-loads-pr';src=root/'models/stable/qwen38-jonathan-q4km-proposal-vision-q.ninfer'
with Artifact(src) as a:
  obj=a.by_id['resource/text/tokenizer.json'];old=json.loads(a.read_object(obj.id));begin=a.payload_offset+obj.offset;size=obj.bytes
with urllib.request.urlopen('https://huggingface.co/Qwen/Qwen3.8-27B/resolve/main/tokenizer.json',timeout=60) as r:
  raw=r.read();official=json.loads(raw)
assert hashlib.sha256(raw).hexdigest()=='0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3'
assert old['model']['vocab']==official['model']['vocab']
normalize=lambda x:[' '.join(v) if isinstance(v,list) else v for v in x]
assert normalize(old['model']['merges'])==normalize(official['model']['merges'])
replacement=json.loads(json.dumps(old));replacement['pre_tokenizer']=official['pre_tokenizer']
changed=json.dumps(replacement,ensure_ascii=False,separators=(',',':')).encode()
assert len(changed)<=size
folder=out/'model';folder.mkdir(exist_ok=True);dst=folder/src.name
assert not dst.exists()
sha=hashlib.sha256();before=hashlib.sha256();after=hashlib.sha256()
with src.open('rb') as f,dst.open('wb') as g:
  offset=0
  while data:=f.read(8*1024*1024):
    sha.update(data);g.write(data);offset+=len(data)
assert sha.hexdigest()=='781fcd7da81eb72a123091054ec9b2bd163798a374a127b0f90f68dae3edbf27'
with dst.open('r+b') as f:f.seek(begin);f.write(changed+b' '*(size-len(changed)))
with src.open('rb') as f,dst.open('rb') as g:
  offset=0
  while x:=f.read(8*1024*1024):
    y=g.read(len(x));end=offset+len(x)
    if offset<begin:assert x[:min(begin-offset,len(x))]==y[:min(begin-offset,len(x))]
    if end>begin+size:
      start=max(0,begin+size-offset);assert x[start:]==y[start:]
    after.update(y);offset=end
r={'status':'PASS','original_sha256':sha.hexdigest(),'qualified_sha256':after.hexdigest(),'bytes':dst.stat().st_size,'official_tokenizer_sha256':hashlib.sha256(raw).hexdigest(),'change':'lab copy only: pre_tokenizer metadata replaced from official Qwen/Qwen3.8-27B; vocabulary and normalized merges match; all bytes outside tokenizer resource proven identical; no weights/quantization/KV changes','path':str(dst)}
(out/'model-compatibility.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
