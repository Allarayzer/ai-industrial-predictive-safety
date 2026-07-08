from __future__ import annotations
import pathlib, json
import numpy as np,pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef,f1_score,recall_score
from scipy.stats import wilcoxon
HERE=pathlib.Path(__file__).resolve(); ROOT=HERE.parents[2] if HERE.parent.parent.name=='q1_extension' else HERE.parents[1]; EXT=ROOT/'q1_extension' if (ROOT/'q1_extension').exists() else ROOT; OUT=EXT/'results'

def metric(y,p):
 y=np.asarray(y,int);p=np.asarray(p,int);n=y==0
 return {'mcc':matthews_corrcoef(y,p),'f1':f1_score(y,p,zero_division=0),'recall':recall_score(y,p,zero_division=0),'far':np.mean(p[n]) if n.any() else np.nan,'cost':(10*np.sum((y==1)&(p==0))+np.sum((y==0)&(p==1)))/len(y)}
# SF-OGD engine metrics and bootstrap
sf=pd.read_csv(OUT/'cmapss_sfogd_trace.csv')
rows=[]
for (sub,u),g in sf.groupby(['subset','unit']):rows.append({'subset':sub,'unit':u,**metric(g.label,g.pred)})
em=pd.DataFrame(rows);em.to_csv(OUT/'cmapss_sfogd_engine_metrics.csv',index=False)
rng=np.random.default_rng(923); boots=[]
for sub,g in em.groupby('subset'):
 vals={k:[] for k in ['mcc','far','recall','cost']}
 for _ in range(5000):
  z=g.iloc[rng.integers(0,len(g),len(g))]
  for k in vals:vals[k].append(z[k].mean())
 row={'subset':sub,'method':'SF-OGD','n_engines':len(g)}
 for k,v in vals.items():row[k+'_macro']=g[k].mean();row[k+'_lo']=np.quantile(v,.025);row[k+'_hi']=np.quantile(v,.975)
 boots.append(row)
pd.DataFrame(boots).to_csv(OUT/'cmapss_sfogd_engine_bootstrap.csv',index=False)
# paired stats channel failure
cf=pd.read_csv(OUT/'channel_failure_results.csv')
stat=[]
focus=cf[cf.method=='Governed escalation'].set_index('seed')
for comp in ['Static','Scheduled','ACI','Quantile tracking','SF-OGD']:
 c=cf[cf.method==comp].set_index('seed')
 for met in ['mcc','post_mcc','expected_cost']:
  d=focus[met]-c[met]
  try:res=wilcoxon(d,zero_method='wilcox',alternative='two-sided')
  except:res=type('R',(object,),{'statistic':np.nan,'pvalue':np.nan})()
  stat.append({'focus':'Governed escalation','comparator':comp,'metric':met,'mean_difference':d.mean(),'median_difference':d.median(),'wilcoxon_W':res.statistic,'p_value':res.pvalue})
pd.DataFrame(stat).to_csv(OUT/'channel_failure_paired_statistics.csv',index=False)
# Figure 1 architecture
fig,ax=plt.subplots(figsize=(12.6,7.2));ax.axis('off')
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
def box(x,y,w,h,text,fc='white',bold=False,ec='black'):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.012',fc=fc,ec=ec,lw=1.4))
 ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=10.2,weight='bold' if bold else None)
def ar(x1,y1,x2,y2,ls='-',rad=0,lw=1.4):
 ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=13,lw=lw,linestyle=ls,connectionstyle=f'arc3,rad={rad}',color='black'))
ax.text(.5,.97,'Governed online adaptation for asynchronous predictive-safety scores',ha='center',fontsize=14,weight='bold')
# main decision path
box(.27,.84,.40,.075,'Asynchronous sensor / model events',(.85,.92,.97,1),True)
box(.27,.72,.40,.075,'Freshness-aware channel cache')
box(.03,.56,.20,.09,'Anomaly risk')
box(.27,.56,.20,.09,'RUL horizon risk')
box(.51,.56,.20,.09,'Supervised failure risk')
box(.19,.40,.44,.085,'Dynamic simplex fusion',(.96,.88,.94,1),True)
box(.19,.25,.44,.085,'Online threshold policy',(.84,.92,.98,1),True)
box(.19,.09,.44,.085,'Operator alert / abstention',(.86,.94,.90,1),True)
# governance column, kept separate to prevent visual ambiguity
box(.76,.68,.21,.09,'Delayed outcomes\nnormal + event labels',(.95,.95,.95,1))
box(.76,.48,.21,.11,'Governance monitor\nFAR, ranking, channel loss',(.98,.93,.82,1),True)
box(.76,.27,.21,.11,'Escalation decision\nthreshold / weights / review',(.98,.93,.82,1),True)
# forward path
ar(.47,.84,.47,.795)
for x in [.13,.37,.61]: ar(.47,.72,x,.65)
for x in [.13,.37,.61]: ar(x,.56,.41,.485)
ar(.41,.40,.41,.335); ar(.41,.25,.41,.175)
# governance path and bounded actions
ar(.865,.68,.865,.59,'--')
ar(.865,.48,.865,.38,'--')
ar(.76,.325,.63,.442,'--',-.06)
ar(.76,.315,.63,.292,'--',.03)
ar(.76,.29,.63,.135,'--',.05)
# labels
ax.text(.50,.675,'latest valid score + freshness',fontsize=8.7,style='italic',ha='center')
ax.text(.865,.62,'delayed feedback',fontsize=8.7,style='italic',ha='center')
ax.set_xlim(0,1);ax.set_ylim(0,1)
plt.tight_layout();plt.savefig(OUT/'figure1_governed_architecture.png',dpi=260,bbox_inches='tight');plt.close()
# Figure 2 controlled location shift
old=pd.read_csv(ROOT/'revision'/'results'/'controlled_summary.csv');sfm=pd.read_csv(OUT/'controlled_sfogd_summary.csv');ctrl=pd.concat([old,sfm],ignore_index=True)
methods=['Static matched','Scheduled rolling','ACI','Rolling ACI','Quantile tracking','Guarded full','SF-OGD']
scens=['none','sudden','gradual','recurring']
fig,axes=plt.subplots(1,3,figsize=(15,4.8))
for ax,met,title in zip(axes,['mcc_mean','far_post_mean','cost_mean'],['MCC','Post-onset FAR','Expected cost (10:1)']):
 p=ctrl[ctrl.method.isin(methods)].pivot(index='scenario',columns='method',values=met).reindex(scens)
 p.plot(kind='bar',ax=ax,width=.82);ax.set_title(title);ax.set_xlabel('');ax.tick_params(axis='x',rotation=0)
 if met=='far_post_mean':ax.axhline(.05,ls='--',lw=1)
 if ax is not axes[-1]:ax.get_legend().remove()
axes[-1].legend(fontsize=7,loc='upper left',bbox_to_anchor=(1.01,1))
fig.suptitle('Controlled location-shift replay (10 seeds)',weight='bold');plt.tight_layout();plt.savefig(OUT/'figure2_controlled_location_shift.png',dpi=240,bbox_inches='tight');plt.close()
# Figure 3 channel reliability shift
cs=pd.read_csv(OUT/'channel_failure_summary.csv').set_index('method')
order=['Static','Scheduled','ACI','Quantile tracking','SF-OGD','Governed escalation'];cs=cs.reindex(order)
fig,axes=plt.subplots(1,3,figsize=(14,4.6))
for ax,met,title in zip(axes,['post_mcc_mean','cost_mean','weight_updates_mean'],['Post-shift MCC','Expected cost (10:1)','Mean weight updates']):
 cs[met].plot(kind='bar',ax=ax);ax.set_title(title);ax.set_xlabel('');ax.tick_params(axis='x',rotation=35,labelsize=8)
fig.suptitle('Channel-reliability shift: threshold-only adaptation cannot restore lost ranking',weight='bold');plt.tight_layout();plt.savefig(OUT/'figure3_channel_reliability_shift.png',dpi=240,bbox_inches='tight');plt.close()
# Figure 4 C-MAPSS
cb=pd.read_csv(ROOT/'revision'/'results'/'cmapss_engine_bootstrap.csv')
sfb=pd.read_csv(OUT/'cmapss_sfogd_engine_bootstrap.csv')
# normalize names
sfb=sfb.rename(columns={'mcc_macro':'mcc_macro','far_macro':'far_macro','recall_macro':'recall_macro','cost_macro':'expected_cost_macro'})
cb2=pd.concat([cb,sfb],ignore_index=True,sort=False)
keep=['Static matched','ACI','Quantile tracking','Guarded full','SF-OGD']
fig,axes=plt.subplots(1,3,figsize=(14.5,4.8))
for ax,met,title in zip(axes,['mcc_macro','far_macro','expected_cost_macro'],['Engine-macro MCC','Engine-macro FAR','Engine-macro cost']):
 p=cb2[cb2.method.isin(keep)].pivot(index='subset',columns='method',values=met)
 p.plot(kind='bar',ax=ax,width=.82);ax.set_title(title);ax.set_xlabel('');ax.tick_params(axis='x',rotation=0)
 if met=='far_macro':ax.axhline(.05,ls='--',lw=1)
 if ax is not axes[-1]:ax.get_legend().remove()
axes[-1].legend(fontsize=7,loc='upper left',bbox_to_anchor=(1.01,1))
fig.suptitle('Engine-disjoint NASA C-MAPSS simulated run-to-failure replay',weight='bold');plt.tight_layout();plt.savefig(OUT/'figure4_cmapss.png',dpi=240,bbox_inches='tight');plt.close()
# Figure 5 battery
bs=pd.read_csv(OUT/'battery_summary.csv').set_index('method').reindex(order)
fig,axes=plt.subplots(1,3,figsize=(14,4.7))
for ax,met,title in zip(axes,['mcc_mean','far_mean','cost_mean'],['Cell-macro MCC','Cell-macro FAR','Cell-macro cost']):
 bs[met].plot(kind='bar',ax=ax);ax.set_title(title);ax.set_xlabel('');ax.tick_params(axis='x',rotation=35,labelsize=8)
 if met=='far_mean':ax.axhline(.05,ls='--',lw=1)
fig.suptitle('Physical NASA Li-ion battery cycle replay (four held-out cells)',weight='bold');plt.tight_layout();plt.savefig(OUT/'figure5_battery.png',dpi=240,bbox_inches='tight');plt.close()
print('done')
