import json, math, statistics as st
from scipy import stats
G=2**30; U=512
hdr={h['file'].split('.Q4')[0]:h for h in json.load(open('analysis/header_audit.json'))}
VOC={'gemma-2-2b':256000,'llama-3.2-1b':128256,'phi-3-mini':32064,'qwen2-0.5b':151936,'qwen2-1.5b':151936,'tinyllama-1.1b':32000}
M=['qwen2-0.5b','tinyllama-1.1b','llama-3.2-1b','qwen2-1.5b','gemma-2-2b','phi-3-mini']
rows=[json.loads(l) for l in open('results_grid2.jsonl')]
def geo(m):
    f=hdr[m]['fields']; nh=f['attention.head_count']; nkv=f['attention.head_count_kv']; L=f['block_count']; d=f['embedding_length']
    return nh,nkv,L,d,f.get('attention.key_length',d//nh),f.get('attention.value_length',d//nh),f.get('attention.sliding_window')
def kv(m,C,window_aware=True,b=2.0):
    nh,nkv,L,d,dk,dv,W=geo(m); per=b*nkv*(dk+dv)
    if window_aware and m.startswith('gemma-2') and W:
        return per*(L//2*C + (L-L//2)*min(C,W+U))
    return per*L*C
def ws(m,C):
    nh=geo(m)[0]; return 4*U*max(VOC[m], nh*C)
def mets(pairs):
    ape=[abs(p-a)/a*100 for p,a in pairs]; se=[(p-a)/a*100 for p,a in pairs]
    return dict(n=len(pairs),mape=st.mean(ape),max_ape=max(ape),worst_under=min(se),worst_over=max(se))
W=lambda m: hdr[m]['file_size']/G
out={}
# KV check against llama.cpp log
out['kv_check']=[dict(m=r['model'],C=r['ctx'],log_MiB=sum(r['kv_MiB'])/2,pred_MiB=kv(r['model'],r['ctx'])/2**20,pred_naive_MiB=kv(r['model'],r['ctx'],False)/2**20) for r in rows if r['exp']=='short']
out['ws_check']=[dict(m=r['model'],C=r['ctx'],log_MiB=r['compute_buf_MiB'][0],pred_MiB=ws(r['model'],r['ctx'])/2**20) for r in rows if r['exp']=='short']
def lomo(sub, predfn):
    pairs=[]; betas={}
    for m in M:
        tr=[r for r in sub if r['model']!=m]; te=[r for r in sub if r['model']==m]
        if not te: continue
        b=st.mean([r['peak_rss_bytes']/G-predfn(r) for r in tr]); betas[m]=b
        pairs+=[(predfn(r)+b, r['peak_rss_bytes']/G) for r in te]
    return pairs,betas
v1=lambda r: W(r['model'])+kv(r['model'],r['ctx'])/G
v2=lambda r: W(r['model'])+kv(r['model'],r['ctx'])/G+ws(r['model'],r['ctx'])/G
for exp in ['short','filled']:
    sub=[r for r in rows if r['exp']==exp]
    for name,fn in [('v1',v1),('v2',v2)]:
        p,b=lomo(sub,fn); out[f'{exp}_{name}_lomo']=dict(**mets(p),betas=b)
# v1 calibrated on short runs then applied to filled (what the submitted estimator would do)
short=[r for r in rows if r['exp']=='short']; filled=[r for r in rows if r['exp']=='filled']
b1=st.mean([r['peak_rss_bytes']/G-v1(r) for r in short])
out['v1_short_calibrated_on_filled']=mets([(v1(r)+b1,r['peak_rss_bytes']/G) for r in filled])
out['filled_vs_short']=[dict(m=r['model'],C=r['ctx'],filled=r['peak_rss_bytes']/G,short=[s for s in short if s['model']==r['model'] and s['ctx']==r['ctx']][0]['peak_rss_bytes']/G) for r in filled]
for x in out['filled_vs_short']: x['increase_pct']=(x['filled']/x['short']-1)*100
mm=[r for r in rows if r['exp']=='mmap_short']
out['mmap_vs_nommap_2048']=[dict(m=r['model'],mmap=r['peak_rss_bytes']/G,nommap=[s for s in short if s['model']==r['model'] and s['ctx']==2048][0]['peak_rss_bytes']/G) for r in mm]
for x in out['mmap_vs_nommap_2048']: x['increase_pct']=(x['mmap']/x['nommap']-1)*100
out['prefill_tps']=[(r['model'],r['ctx'],r['prompt_tps']) for r in filled]
json.dump(out,open('analysis/grid2_analysis.json','w'),indent=1)
for k,v in out.items(): print(k, json.dumps(v,default=lambda x: round(x,3))[:900])
