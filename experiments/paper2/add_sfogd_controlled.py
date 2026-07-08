import sys,pathlib,pandas as pd,numpy as np
HERE=pathlib.Path(__file__).resolve(); ROOT=HERE.parents[2] if HERE.parent.parent.name=='q1_extension' else HERE.parents[1]
EXT=ROOT/'q1_extension' if (ROOT/'q1_extension').exists() else ROOT
sys.path.insert(0,str(ROOT/'revision'/'code'))
import run_p4_revision_benchmark as old
sys.path.insert(0,str(HERE.parent))
from run_q1_extension import run_threshold,metrics,local_far_error,conformal_q
out=[]
for scenario in ['none','sudden','gradual','recurring']:
  for seed in range(10):
    d=old.build_controlled(seed,scenario)
    normal=d.labels[:d.calibration_end]==0
    ref=d.async_score[:d.calibration_end][normal]
    y=d.labels[d.calibration_end:];s=d.async_score[d.calibration_end:]
    q=conformal_q(ref)
    p,t,u=run_threshold('SF-OGD',s,y,q,delay=120)
    m=metrics(y,p,s);post=max(0,d.drift_onset-d.calibration_end);pn=y[post:]==0
    m['far_post']=float(np.mean(p[post:][pn]));m['lfe100']=local_far_error(y,p,100)
    out.append({'scenario':scenario,'seed':seed,'method':'SF-OGD','updates':u,**m})
r=pd.DataFrame(out);r.to_csv(EXT/'results'/'controlled_sfogd_results.csv',index=False)
s=r.groupby(['scenario','method']).agg(mcc_mean=('mcc','mean'),mcc_sd=('mcc','std'),far_mean=('far','mean'),far_post_mean=('far_post','mean'),cost_mean=('expected_cost','mean'),lfe100_mean=('lfe100','mean'),updates_mean=('updates','mean')).reset_index();s.to_csv(EXT/'results'/'controlled_sfogd_summary.csv',index=False)
print(s.round(4).to_string(index=False))
