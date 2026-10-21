import json, statistics as st, numpy as np
exec(open('analysis/analyze_grid2.py').read().split("out={}")[0])
filled=[r for r in rows if r['exp']=='filled']
A=lambda r: 4*U*geo(r['model'])[0]*r['ctx']/G
base=lambda r: W(r['model'])+kv(r['model'],r['ctx'])/G
def fit(tr):
    X=np.array([[A(r),1] for r in tr]); y=np.array([r['peak_rss_bytes']/G-base(r) for r in tr]); return np.linalg.lstsq(X,y,rcond=None)[0]
pairs=[];co={}
for m in M:
    tr=[r for r in filled if r['model']!=m]; te=[r for r in filled if r['model']==m]
    if not te: continue
    g,b=fit(tr); co[m]=(g,b); pairs+=[(base(r)+g*A(r)+b, r['peak_rss_bytes']/G, m, r['ctx']) for r in te]
g,b=fit(filled)
print('insample gamma,beta',g,b)
for p in pairs: print(p[2],p[3],round(p[0],3),round(p[1],3),round((p[0]-p[1])/p[1]*100,2))
ape=[abs(p-a)/a*100 for p,a,_,_ in pairs]; print('LOMO MAPE',st.mean(ape),'max',max(ape),'worst under',min((p-a)/a*100 for p,a,_,_ in pairs)); print(co)
