import hashlib,json,sys,uuid
from pathlib import Path
sys.path.insert(0,'source')
from tools.artifact.reader import Artifact
from tools.artifact.framing import HEADER,MAGIC
out=Path('results/cooperative-loads-pr');src=Path('models/stable/qwen38-jonathan-q4km-proposal-vision-q.ninfer');dst=out/'model'/src.name
r=json.loads((out/'model-compatibility.json').read_text());r['previous_metadata_sha256']=r['qualified_sha256']
with Artifact(src) as a:
 directory=a.directory.to_json();offset=a.payload_offset
 tokenizer=json.loads(a.read_object('resource/text/tokenizer.json'));config=json.loads(a.read_object('resource/text/tokenizer_config.json'))
 oldrange=a.by_id['resource/text/tokenizer.json'].bytes
 official=json.loads((out/'official-tokenizer.json').read_text())
 tokenizer['pre_tokenizer']=official['pre_tokenizer']
 config['added_tokens_decoder']={str(x['id']):{k:v for k,v in x.items() if k!='id'} for x in tokenizer['added_tokens']}
 dumps=lambda x:json.dumps(x,ensure_ascii=False,separators=(',',':')).encode()
 tokenbytes=dumps(tokenizer);configbytes=dumps(config)
 assert len(tokenbytes)+len(configbytes)<oldrange
 for obj in directory['objects']:
  if obj['id']=='resource/text/tokenizer.json':obj['bytes']=len(tokenbytes)
  if obj['id']=='resource/text/tokenizer_config.json':obj.update(offset=len(tokenbytes),bytes=len(configbytes))
 rawdir=dumps(directory);assert HEADER.size+len(rawdir)<=offset
 with dst.open('r+b') as f:
  f.write(HEADER.pack(MAGIC,len(rawdir),uuid.uuid4().bytes));f.write(rawdir);f.write(b' '*(offset-HEADER.size-len(rawdir)))
  f.write(tokenbytes);f.write(configbytes);f.write(b' '*(oldrange-len(tokenbytes)-len(configbytes)))
with Artifact(dst) as lab,Artifact(src) as original:
 assert [(o.id,o.shape,o.format,o.layout,o.offset,o.bytes) for o in lab.objects if o.kind=='tensor']==[(o.id,o.shape,o.format,o.layout,o.offset,o.bytes) for o in original.objects if o.kind=='tensor']
 assert lab.directory.components==original.directory.components
 assert lab.directory.bindings==original.directory.bindings
 assert lab.directory.uses==original.directory.uses
 h=hashlib.sha256()
 with src.open('rb') as f,dst.open('rb') as g:
  position=0
  while x:=f.read(8*1024*1024):
   y=g.read(len(x));h.update(y);end=position+len(x)
   if end>offset+oldrange:
    start=max(0,offset+oldrange-position);assert x[start:]==y[start:]
   position=end
r.update(qualified_sha256=h.hexdigest(),change='Lab copy only: official pre_tokenizer metadata and added_tokens_decoder derived exactly from embedded added_tokens. Header/resource offsets updated in existing reserve; every byte after tokenizer resource, all tensor records/bindings/Uses/configs and tensor data unchanged. Production artifact untouched.')
(out/'model-compatibility.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
