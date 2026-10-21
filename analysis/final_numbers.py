"""Collect every number quoted in the camera-ready (E1 = submitted grid, E2 = replication)."""
import json, random, statistics as st, numpy as np
R=json.load(open('analysis/reanalysis.json'))
exec(open('analysis/analyze_grid2.py').read().split("out={}")[0])
G=2**30
# ---------- E1 cluster bootstrap (resample models) for LOMO APEs of corrected estimator
meas=json.load(open('e1_original/measurements_36cfg.json'))
import importlib.util
spec=importlib.util.spec_from_file_location('ra','analysis/reanalyze.py')
src=open('analysis/reanalyze.py').read().split("R={}")[0]
ns={}; exec(src,ns)
def lomo_apes(variant):
    out={}
    for m in ns['models']:
        tr=[r for r in ns['meas'] if r['model']!=m]; a,b=ns['fit'](tr,variant)
        out[m]=[((ns['pred'](r,variant,a,b)-r['meas'])/r['meas']*100, r) for r in ns['meas'] if r['model']==m]
    return out
def cluster_ci(per):
    ms=list(per); random.seed(1); bs=[]
    for _ in range(4000):
        pick=random.choices(ms,k=len(ms)); bs.append(st.mean([abs(e) for m in pick for e,_ in per[m]]))
    bs.sort(); return bs[int(.025*len(bs))], bs[int(.975*len(bs))-1]
F={}
for v in ['orig','swa']:
    per=lomo_apes(v); F[f'E1_{v}_lomo_cluster_ci']=cluster_ci(per)
    if v=='swa':
        errs=[(e,r) for m in per for e,r in per[m]]
        # decision accuracy across budgets 0.40..6.00 GiB step 0.01, with margin 0 and 5%
        for marg in [0,0.04,0.05]:
            tot=ok=false_fit=0
            for B in np.arange(0.40,6.0,0.01):
                for e,r in errs:
                    pred=r['meas']*(1+e/100)*(1+marg); fits_pred=pred<=B; fits_true=r['meas']<=B
                    tot+=1; ok+= fits_pred==fits_true; false_fit+= fits_pred and not fits_true
            F[f'E1_decision_margin{marg}']=dict(acc=ok/tot,false_fit=false_fit,pairs=tot)
# ---------- E2
filled=[r for r in rows if r['exp']=='filled']; short=[r for r in rows if r['exp']=='short']
A=lambda r: 4*U*geo(r['model'])[0]*r['ctx']/G
base=lambda r: W(r['model'])+kv(r['model'],r['ctx'])/G
def fitgb(tr):
    X=np.array([[A(r),1] for r in tr]); y=np.array([r['peak_rss_bytes']/G-base(r) for r in tr]); return np.linalg.lstsq(X,y,rcond=None)[0]
per={}; co={}
for m in M:
    tr=[r for r in filled if r['model']!=m]; g,b=fitgb(tr); co[m]=(g,b)
    per[m]=[((base(r)+g*A(r)+b-r['peak_rss_bytes']/G)/(r['peak_rss_bytes']/G)*100, r) for r in filled if r['model']==m]
e=[x for m in per for x,_ in per[m]]
F['E2_filled_v2_lomo']=dict(n=len(e),mape=st.mean(map(abs,e)),worst_under=min(e),worst_over=max(e),cluster_ci=cluster_ci(per),gamma_range=(min(c[0] for c in co.values()),max(c[0] for c in co.values())),beta_range=(min(c[1] for c in co.values()),max(c[1] for c in co.values())))
F['E2_filled_v2_insample']=list(fitgb(filled))
# short-calibrated reserved model applied to filled
b1=st.mean([r['peak_rss_bytes']/G-base(r) for r in short])
e1=[(base(r)+b1-r['peak_rss_bytes']/G)/(r['peak_rss_bytes']/G)*100 for r in filled]
F['E2_v1_on_filled']=dict(mape=st.mean(map(abs,e1)),worst_under=min(e1),beta=b1)
# E2 short v1 LOMO
pe=[]
for m in M:
    tr=[r for r in short if r['model']!=m]; b=st.mean([r['peak_rss_bytes']/G-base(r) for r in tr])
    pe+=[(base(r)+b-r['peak_rss_bytes']/G)/(r['peak_rss_bytes']/G)*100 for r in short if r['model']==m]
F['E2_short_v1_lomo']=dict(n=len(pe),mape=st.mean(map(abs,pe)),worst_under=min(pe),max_ape=max(map(abs,pe)))
# KV log agreement
F['E2_kv_log_match']=all(abs(sum(r['kv_MiB'])/2-kv(r['model'],r['ctx'])/2**20)<0.01 for r in short)
F['E2_table']=[dict(m=r['model'],C=r['ctx'],short=[s for s in short if s['model']==r['model'] and s['ctx']==r['ctx']][0]['peak_rss_bytes']/G,filled=r['peak_rss_bytes']/G,v1=base(r)+b1,v2=base(r)+co[r['model']][0]*A(r)+co[r['model']][1]) for r in filled]
mm=[r for r in rows if r['exp']=='mmap_short']
F['E2_mmap_increase_pct']=[(r['model'],(r['peak_rss_bytes']/[s for s in short if s['model']==r['model'] and s['ctx']==2048][0]['peak_rss_bytes']-1)*100) for r in mm]
# E1 vs E2 short replication agreement (FP16 only)
d=[]
for s in short:
    o=[x for x in meas if x['model']==s['model'] and x['ctx']==s['ctx'] and x['kv_type']=='f16'][0]
    d.append((s['peak_rss_bytes']-o['peak_rss_max_bytes'])/o['peak_rss_max_bytes']*100)
F['E1_E2_short_diff_pct']=dict(mean_abs=st.mean(map(abs,d)),max_abs=max(map(abs,d)))
json.dump(F,open('analysis/final_numbers.json','w'),indent=1,default=float)
for k,v in F.items(): print(k, json.dumps(v,default=lambda x: round(float(x),4) if not isinstance(x,dict) else str(x))[:700])
