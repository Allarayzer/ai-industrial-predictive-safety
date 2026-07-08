import pandas as pd, numpy as np, pathlib
from sklearn.metrics import matthews_corrcoef, f1_score, recall_score
OUT=pathlib.Path('/mnt/data/P4_7of10_revision/results')
d=pd.read_csv(OUT/'cmapss_crossfit_trace.csv',usecols=['subset','unit','label','method','pred'])
rows=[]
for (subset,method,unit),g in d.groupby(['subset','method','unit'],sort=False):
    y=g.label.to_numpy(int); p=g.pred.to_numpy(int); normal=y==0
    rows.append({'subset':subset,'method':method,'unit':unit,
                 'mcc':matthews_corrcoef(y,p),'f1':f1_score(y,p,zero_division=0),'recall':recall_score(y,p,zero_division=0),
                 'far':float(p[normal].mean()) if normal.any() else np.nan,
                 'expected_cost':float((10*np.sum((y==1)&(p==0))+np.sum((y==0)&(p==1)))/len(y))})
e=pd.DataFrame(rows);e.to_csv(OUT/'cmapss_engine_metrics.csv',index=False)
rng=np.random.default_rng(20260708); summary=[]
for (subset,method),g in e.groupby(['subset','method'],sort=False):
    row={'subset':subset,'method':method,'n_engines':len(g)}
    for metric in ['mcc','f1','recall','far','expected_cost']:
        x=g[metric].to_numpy(float); x=x[np.isfinite(x)]
        idx=rng.integers(0,len(x),size=(5000,len(x)))
        bm=x[idx].mean(axis=1)
        row[metric+'_macro']=float(x.mean());row[metric+'_lo']=float(np.quantile(bm,.025));row[metric+'_hi']=float(np.quantile(bm,.975))
    summary.append(row)
s=pd.DataFrame(summary);s.to_csv(OUT/'cmapss_engine_bootstrap.csv',index=False)
focus='Guarded full'; comps=['Static matched','Scheduled rolling','ACI','Rolling ACI','AgACI-style','Quantile tracking']
diffs=[]
for subset,gs in e.groupby('subset'):
    wide={m:g.set_index('unit') for m,g in gs.groupby('method')}
    for comp in comps:
        if focus not in wide or comp not in wide:continue
        units=wide[focus].index.intersection(wide[comp].index)
        row={'subset':subset,'focus':focus,'comparator':comp,'n_engines':len(units)}
        for metric in ['mcc','far','expected_cost']:
            x=(wide[focus].loc[units,metric]-wide[comp].loc[units,metric]).to_numpy(float)
            idx=rng.integers(0,len(x),size=(5000,len(x)));bm=x[idx].mean(axis=1)
            row['delta_'+metric]=float(x.mean());row['delta_'+metric+'_lo']=float(np.quantile(bm,.025));row['delta_'+metric+'_hi']=float(np.quantile(bm,.975))
        diffs.append(row)
pd.DataFrame(diffs).to_csv(OUT/'cmapss_engine_paired_bootstrap.csv',index=False)
print(s[s.method.isin(['Static matched','Scheduled rolling','ACI','Rolling ACI','AgACI-style','Quantile tracking','Guarded full'])].round(4).to_string(index=False))
