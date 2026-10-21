import json, subprocess, os, sys
sys.path.insert(0,'.'); from run_rss import run
B='llama.cpp/build/bin'; M=['qwen2-0.5b','tinyllama-1.1b','llama-3.2-1b','qwen2-1.5b','gemma-2-2b','phi-3-mini']
corpus=open('fill_corpus.txt').read()
pang=' '.join(['The quick brown fox jumps over the lazy dog.']*20)
out='results_grid2.jsonl'
done=set()
if os.path.exists(out):
    for l in open(out): r=json.loads(l); done.add((r['exp'],r['model'],r['ctx']))
def ntok(m,text):
    open('/tmp/p.txt','w').write(text)
    o=subprocess.run([f'{B}/llama-tokenize','-m',f'models/{m}.Q4_K_M.gguf','-f','/tmp/p.txt','--show-count','--log-disable'],capture_output=True,text=True).stdout
    return int(o.strip().split()[-1])
def fill(m,target):
    lo,hi=0,len(corpus)
    # ratio guess then bisection on chars
    r=ntok(m,corpus[:20000])/20000; n=int(target/r)
    lo,hi=int(n*0.8),min(int(n*1.2),len(corpus))
    for _ in range(14):
        mid=(lo+hi)//2
        if ntok(m,corpus[:mid])<=target: lo=mid
        else: hi=mid
    return corpus[:lo]
def go(exp,m,C,prompt_text,load_mode='none'):
    if (exp,m,C) in done: return
    pf=f'prompts/{exp}_{m}_{C}.txt'; os.makedirs('prompts',exist_ok=True); open(pf,'w').write(prompt_text)
    args=[f'{B}/llama-completion','-m',f'models/{m}.Q4_K_M.gguf','-c',str(C),'-ctk','f16','-ctv','f16','-fa','off','-fit','off','--load-mode',load_mode,'-t','2','-no-cnv','-f',pf,'-n','16','--no-warmup','-lv','4','--temp','0','-b','2048','-ub','512']
    r=run(args,f'logs/{exp}_{m}_{C}.log'); r.update(exp=exp,model=m,ctx=C,kv_type='f16',load_mode=load_mode)
    open(out,'a').write(json.dumps(r)+'\n'); print(exp,m,C,r['peak_rss_bytes']/2**30,r['n_prompt_tokens'],round(r['wall_s']),flush=True)
for m in M:
    for C in [512,2048,8192]: go('short',m,C,pang)
for m in M: go('mmap_short',m,2048,pang,'mmap')
for m in M:
    for C in [2048,8192]: go('filled',m,C,fill(m,C-64))
print('ALL_DONE',flush=True)
