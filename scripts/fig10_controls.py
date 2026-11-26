"""Fig. 10 of the revised manuscript: where the class information lies (SD). Reads revision/runs.csv
written by notebook 05. Usage: python scripts/fig10_controls.py <revision/runs.csv> <out_prefix>"""
import sys, numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt
from scipy import stats
mpl.rcParams.update({'font.size':9,'axes.titlesize':9,'axes.labelsize':9,'xtick.labelsize':7.5,'ytick.labelsize':8,
                     'legend.fontsize':7.5,'savefig.dpi':300,'axes.spines.top':False,'axes.spines.right':False,
                     'pdf.fonttype':42,'ps.fonttype':42})
R=pd.read_csv(sys.argv[1]); out=sys.argv[2]
def q(exp,reg,var,seeds=None):
    x=R[(R.exp==exp)&(R.regime==reg)&(R.variant==var)&(R.model=='w25k')]
    return x if seeds is None else x[x.seed.isin(seeds)]
def ci(x):
    s=x.groupby('subject').bal_acc.mean(); h=stats.t.ppf(.975,len(s)-1)*s.std(ddof=1)/np.sqrt(len(s)); return s.mean(),h
A=[('full epoch',q('E1','sd','base',[0,1,2]),'#2ecc71'),('central\nelectrodes',q('E7','sd','central'),'#2ecc71'),
   ('low-pass\n1 Hz',q('E9','sd','lp1'),'#2ecc71'),('pre-onset\nonly',q('E9','sd','pre'),'#e67e22'),
   ('high-pass\n0.5 Hz',q('E9','sd','hp0.5'),'#95a5a6'),('high-pass\n1 Hz',q('E7','sd','hp1'),'#95a5a6'),
   ('linear\ndetrend',q('E9','sd','dt'),'#95a5a6'),('baseline-\ncorrected',q('E9','sd','bc'),'#95a5a6'),
   ('permuted\nlabels',R[(R.exp=='E6')&(R.regime=='sd')],'#bdc3c7')]
B=[('full epoch',q('E4','sd_cv_group','base'),'#2ecc71'),('pre-onset\nonly',q('E9','sd_cv_group','pre'),'#e67e22'),
   ('baseline-\ncorrected',q('E9','sd_cv_group','bc'),'#95a5a6'),('high-pass 1 Hz,\ncentral',q('E7','sd_cv_group','hp1_central'),'#95a5a6')]
fig,ax=plt.subplots(1,2,figsize=(11,3.5),gridspec_kw={'width_ratios':[9,4.2]})
for a,items,title in [(ax[0],A,'(a) held-out split (SD, mean of 3 seeded runs; permuted: 10 runs)'),
                      (ax[1],B,'(b) instruction-block 5-fold CV (SD)')]:
    m=[ci(x) for _,x,_ in items]
    a.bar(range(len(items)),[v for v,_ in m],yerr=[h for _,h in m],capsize=3,width=0.62,
          color=[c for *_,c in items],edgecolor='k',lw=.6,alpha=.9)
    for i,(v,h) in enumerate(m): a.text(i,v+h+1.0,f'{v:.1f}',ha='center',fontsize=7.5,fontweight='bold')
    a.axhline(20,ls=':',c='gray',lw=1,label='chance (20%)')
    a.set_xticks(range(len(items))); a.set_xticklabels([n for n,*_ in items]); a.set_ylim(0,62)
    a.set_title(title); a.legend(loc='upper right',frameon=False)
ax[0].set_ylabel('SD balanced accuracy (%)')
plt.tight_layout(); fig.savefig(out+'.pdf',bbox_inches='tight'); fig.savefig(out+'.png',bbox_inches='tight',dpi=150)
for n,x,_ in A+B: print(n.replace('\n',' '), [round(v,2) for v in ci(x)])
