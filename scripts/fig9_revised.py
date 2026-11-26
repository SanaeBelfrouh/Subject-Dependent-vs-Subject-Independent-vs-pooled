# Fig. 9 of the revised manuscript: the same measured values as notebook 03 (27.9, 49.7, 44.5, 33.5) with
# clearer annotations. Usage: python scripts/fig9_revised.py <out_prefix>
import sys, matplotlib as mpl, matplotlib.pyplot as plt
mpl.rcParams.update({'font.size':9,'axes.titlesize':9,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,
                     'legend.fontsize':7.5,'savefig.dpi':300,'axes.spines.top':False,'axes.spines.right':False,
                     'pdf.fonttype':42,'ps.fonttype':42})
GAP2D_SD, FLATTEN_SD, LINEAR_PROBE = 27.9, 49.73, 44.5
BAND_PROBE = {'broadband\nmean (ours)':44.5, 'gamma\npower':33.5}
fig,ax=plt.subplots(1,2,figsize=(11,3.4))
v=[GAP2D_SD,FLATTEN_SD]
b=ax[0].bar([0,1],v,width=0.55,color=['#e74c3c','#2ecc71'],alpha=.88,edgecolor='k',lw=.7)
ax[0].set_xticks([0,1]); ax[0].set_xticklabels(['Global spatial\npooling (GAP2D)','Flatten\n(ours)'])
ax[0].axhline(20,ls=':',c='gray',label='chance (20%)')
ax[0].axhline(LINEAR_PROBE,ls='--',c='k',lw=1,label=f'linear probe on the same maps ({LINEAR_PROBE}%)')
for r,x in zip(b,v): ax[0].text(r.get_x()+r.get_width()/2,x+1.1,f'{x:.1f}',ha='center',fontweight='bold',fontsize=8)
# difference shown as a bracket to the right of the flatten bar, clear of both bars
xb=1.42
ax[0].annotate('',xy=(xb,v[1]),xytext=(xb,v[0]),arrowprops=dict(arrowstyle='<->',color='k',lw=.9))
ax[0].plot([0.275,xb],[v[0],v[0]],ls=(0,(2,2)),c='#e74c3c',lw=.8,zorder=0)
ax[0].text(xb+0.05,(v[0]+v[1])/2,f'+{v[1]-v[0]:.1f} pp\n(flatten $-$ GAP2D)',va='center',fontsize=8)
ax[0].set_xlim(-0.5,2.15)
ax[0].set_ylabel('SD balanced accuracy (%)'); ax[0].set_ylim(0,62); ax[0].legend(loc='upper left',frameon=False)
ax[0].set_title('(a) spatial position carries the signal')
names=list(BAND_PROBE); pv=[BAND_PROBE[k] for k in names]
b=ax[1].bar(names,pv,width=0.55,color=['#2ecc71','#bdc3c7'],alpha=.88,edgecolor='k',lw=.7)
ax[1].axhline(20,ls=':',c='gray',label='chance (20%)')
for r,x in zip(b,pv): ax[1].text(r.get_x()+r.get_width()/2,x+.8,f'{x:.1f}',ha='center',fontsize=8)
ax[1].set_ylabel('linear-probe accuracy (%)'); ax[1].set_ylim(0,52); ax[1].legend(frameon=False)
ax[1].set_title('(b) feature summary: broadband mean vs band power')
plt.tight_layout(); fig.savefig(sys.argv[1]+'.pdf',bbox_inches='tight'); fig.savefig(sys.argv[1]+'.png',bbox_inches='tight',dpi=150)
