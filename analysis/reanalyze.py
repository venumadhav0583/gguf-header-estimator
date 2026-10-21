import json, math, random, statistics as st
from scipy import stats
G=1024**3
meas=json.load(open('e1_original/measurements_36cfg.json'))
hdr={h['file'].split('.Q4')[0]:h for h in json.load(open('analysis/header_audit.json'))}
gp={(r['model'],r['kv_type'],r['ctx'],r['mode']):r for r in json.load(open('analysis/ggufparser_estimates.json'))['results']}
models=['qwen2-0.5b','tinyllama-1.1b','llama-3.2-1b','qwen2-1.5b','gemma-2-2b','phi-3-mini']
P={m:(hdr[m]['file_size']*8/4.65)/1e9 for m in models}  # same file-size-derived P as original analyze.py
def geo(m):
    f=hdr[m]['fields']; nh=f['attention.head_count']; nkv=f['attention.head_count_kv']; L=f['block_count']; d=f['embedding_length']
    return nh,nkv,L,d,f.get('attention.key_length',d//nh),f.get('attention.value_length',d//nh)
BYTES_OLD={'f16':2.0,'q8_0':1.0}; BYTES_NEW={'f16':2.0,'q8_0':34/32}
def kv(m,kvt,C,variant):
    nh,nkv,L,d,dk,dv=geo(m)
    if variant=='orig': return 2*BYTES_OLD[kvt]*nkv*(d/nh)*L*C
    if variant=='fullheads_orig': return 2*BYTES_OLD[kvt]*nh*(d/nh)*L*C
    if variant=='new': return BYTES_NEW[kvt]*nkv*(dk+dv)*L*C
    if variant=='new_fullheads': return BYTES_NEW[kvt]*nh*(dk+dv)*L*C
    if variant in ('swa','swa_fullheads'):
        h = nh if variant=='swa_fullheads' else nkv
        W=hdr[m]['fields'].get('attention.sliding_window')
        if m.startswith('gemma-2') and W: return BYTES_NEW[kvt]*h*(dk+dv)*(L//2*C+(L-L//2)*min(C,W+512))
        return BYTES_NEW[kvt]*h*(dk+dv)*L*C
for r in meas: r['meas']=r['peak_rss_max_bytes']/G
def fit(rows,variant):
    x=[P[r['model']] for r in rows]; y=[r['meas']-hdr[r['model']]['file_size']/G-kv(r['model'],r['kv_type'],r['ctx'],variant)/G for r in rows]
    s=stats.linregress(x,y); return s.slope,s.intercept
def pred(r,variant,a,b,kvvariant=None):
    return hdr[r['model']]['file_size']/G+kv(r['model'],r['kv_type'],r['ctx'],kvvariant or variant)/G+a*P[r['model']]+b
def metrics(pairs):
    ape=[abs(p-a)/a*100 for p,a in pairs]; se=[(p-a)/a*100 for p,a in pairs]
    random.seed(0); bs=sorted(st.mean(random.choices(ape,k=len(ape))) for _ in range(2000))
    return dict(mape=st.mean(ape),ci=(bs[50],bs[1949]),max_ape=max(ape),worst_under=min(se),worst_over=max(se),
                r=stats.pearsonr([p for p,_ in pairs],[a for _,a in pairs])[0],rmse=math.sqrt(st.mean([(p-a)**2 for p,a in pairs])),n=len(pairs),
                n_under=sum(1 for x in se if x<0))
R={}
for v in ['orig','new','swa']:
    a,b=fit(meas,v); R[v+'_insample']=dict(alpha=a,beta=b,**metrics([(pred(r,v,a,b),r['meas']) for r in meas]))
    pairs=[];coef={}
    for m in models:
        tr=[r for r in meas if r['model']!=m]; a2,b2=fit(tr,v); coef[m]=(a2,b2)
        pairs+= [(pred(r,v,a2,b2),r['meas']) for r in meas if r['model']==m]
    R[v+'_lomo']=dict(coef=coef,**metrics(pairs))
    # per-model LOMO
    R[v+'_lomo_permodel']={m:metrics([(pred(r,v,*coef[m]),r['meas']) for r in meas if r['model']==m])['mape'] for m in models}
# Ablations on corrected estimator, LOMO coefficients
a,b=R['swa_insample']['alpha'],R['swa_insample']['beta']
gqa=[r for r in meas if r['model']!='phi-3-mini']
abl={}
abl['no_overhead']=metrics([(pred(r,'swa',0,0),r['meas']) for r in meas])['mape']
abl['full_heads_gqa']=metrics([(pred(r,'swa',a,b,'swa_fullheads'),r['meas']) for r in gqa])['mape']
abl['full_heads_gqa_8k']=metrics([(pred(r,'swa',a,b,'swa_fullheads'),r['meas']) for r in gqa if r['ctx']==8192])['mape']
abl['kv_always_f16']=metrics([(hdr[r['model']]['file_size']/G+kv(r['model'],'f16',r['ctx'],'swa')/G+a*P[r['model']]+b,r['meas']) for r in meas])['mape']
abl['weights_only']=metrics([(hdr[r['model']]['file_size']/G,r['meas']) for r in meas])
R['ablation_new']=abl
# gguf-parser baseline
for mode in ['--no-mmap','--mmap']:
    R['ggufparser'+mode]=metrics([(gp[(r['model'],r['kv_type'],r['ctx'],mode)]['ram_uma']/G,r['meas']) for r in meas])
# KV slope per model (FP16, 512->8192)
sl={}
for m in models:
    rr={r['ctx']:r['peak_rss_max_bytes'] for r in meas if r['model']==m and r['kv_type']=='f16'}
    ms=(rr[8192]-rr[512])/(8192-512)/2**20; po=kv(m,'f16',1,'orig')/2**20; pn=(kv(m,'f16',8192,'swa')-kv(m,'f16',512,'swa'))/(8192-512)/2**20
    sl[m]=dict(meas=ms,pred_orig=po,pred_new=pn,ratio_new=ms/pn)
R['kv_slope_MiB_per_tok']=sl
# Q8 vs F16 KV reduction measured
red=[]
for m in models:
    for C in [2048,8192]:
        f=[r for r in meas if r['model']==m and r['ctx']==C and r['kv_type']=='f16'][0]['peak_rss_max_bytes']
        q=[r for r in meas if r['model']==m and r['ctx']==C and r['kv_type']=='q8_0'][0]['peak_rss_max_bytes']
        red.append(dict(m=m,C=C,saved_MiB=(f-q)/2**20,pred_saved_MiB=(kv(m,'f16',C,'swa')-kv(m,'q8_0',C,'swa'))/2**20))
R['q8_savings']=red
# required safety margin from LOMO
v='new'; worst=R['swa_lomo']['worst_under']; R['lomo_margin_needed_pct']=-worst
json.dump(R,open('analysis/reanalysis.json','w'),indent=1,default=float)
for k,vv in R.items(): print(k, json.dumps(vv,default=lambda x: round(x,4) if isinstance(x,float) else str(x))[:600])
