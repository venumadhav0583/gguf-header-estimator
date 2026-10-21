import json
F=json.load(open('analysis/final_numbers.json'))
names={'qwen2-0.5b':'Qwen2-0.5B','qwen2-1.5b':'Qwen2-1.5B','tinyllama-1.1b':'TinyLlama-1.1B','llama-3.2-1b':'LLaMA-3.2-1B','gemma-2-2b':'Gemma-2-2B','phi-3-mini':'Phi-3-mini'}
order=list(names)
t=[r for r in F['E2_table'] if r['C']==8192]; t.sort(key=lambda r: order.index(r['m']))
inc=[(r['filled']/r['short']-1)*100 for r in t]
L=F['E2_filled_v2_lomo']; g,b=F['E2_filled_v2_insample']
G=2**30; A=4*512*12*8192/G
q=[r for r in F['E2_table'] if r['m']=='qwen2-1.5b' and r['C']==8192][0]
qpred=0.918+0.219+g*A+b
rows=''.join(f"{names[r['m']]} & {r['short']:.3f} & {r['filled']:.3f} & {(r['filled']/r['short']-1)*100:.0f} & {r['v1']:.3f} & {r['v2']:.3f} \\\\\n" for r in t)
M=dict(FilledMaxIncrease=f"{max(inc):.0f}",FilledMinIncreaseEight=f"{min(inc):.0f}",FilledMAPE=f"{L['mape']:.2f}",
 FilledCI=f"{L['cluster_ci'][0]:.2f}--{L['cluster_ci'][1]:.2f}\\%",FilledWorstUnder=f"$-{abs(L['worst_under']):.2f}$",
 VoneWorstUnder=f"{abs(F['E2_v1_on_filled']['worst_under']):.1f}",GammaLo=f"{L['gamma_range'][0]:.2f}",GammaHi=f"{L['gamma_range'][1]:.2f}",
 BetaLo=f"{L['beta_range'][0]:.3f}",BetaHi=f"{L['beta_range'][1]:.3f}",GammaIn=f"{g:.2f}",BetaIn=f"{b:.3f}",
 AttQwen=f"{g*A:.3f}",QwenFilledPred=f"{qpred:.3f}",QwenFilledMeas=f"{q['filled']:.3f}",FilledN=str(L['n']))
out=''.join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k,v in M.items())+"\\newcommand{\\FilledRows}{%\n"+rows+"}\n"
open('paper/e2numbers.tex','w').write(out); print(out)
json.dump(dict(gamma=g,beta=b),open('coefficients_filled.json','w'))
