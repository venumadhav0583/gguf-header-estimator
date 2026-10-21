import json, subprocess, itertools
models=['qwen2-0.5b','tinyllama-1.1b','llama-3.2-1b','qwen2-1.5b','gemma-2-2b','phi-3-mini']
out=[]
for m,kv,c,mm in itertools.product(models,['f16','q8_0'],[512,2048,8192],['--no-mmap','--mmap']):
    cmd=['./gguf-parser','-m',f'models/{m}.Q4_K_M.gguf','-c',str(c),'--ngl','0',mm,'--ctk',kv,'--ctv',kv,'--no-flash-attention','--json','--skip-tokenizer','--skip-architecture','--skip-metadata']
    d=json.loads(subprocess.run(cmd,capture_output=True,text=True).stdout)['estimate']
    it=d['items'][0]
    out.append(dict(model=m,kv_type=kv,ctx=c,mode=mm,ram_uma=it['ram']['uma'],ram_nonuma=it['ram']['nonuma'],vram_uma=it['vrams'][0]['uma'],vram_nonuma=it['vrams'][0]['nonuma']))
json.dump(dict(tool='gguf-parser-go v0.26.4',flags='-c C --ngl 0 --ctk/--ctv KV --no-flash-attention, default -b 2048 -ub 512, default platform footprint 150,250 MiB',results=out),open('analysis/ggufparser_estimates.json','w'),indent=1)
print(len(out))
