from __future__ import annotations
import math, json, pathlib, hashlib
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.ensemble import IsolationForest, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import matthews_corrcoef, f1_score, precision_score, recall_score, roc_auc_score
from scipy.optimize import minimize

HERE=pathlib.Path(__file__).resolve()
ROOT=HERE.parents[2] if HERE.parent.parent.name=='q1_extension' else HERE.parents[1]
EXT=ROOT/'q1_extension' if (ROOT/'q1_extension').exists() else ROOT
OUT=EXT/'results'; OUT.mkdir(parents=True,exist_ok=True)
DATA=EXT/'data'/'nasa_battery_discharge.csv'
ALPHA=.05


def conformal_q(x, alpha=ALPHA):
    x=np.sort(np.asarray(x,float)[np.isfinite(x)])
    if len(x)<20: return float(np.quantile(x,1-alpha))
    rank=min(len(x)-1,max(0,math.ceil((len(x)+1)*(1-alpha))-1))
    return float(x[rank])

def metrics(y,p,s):
    y=np.asarray(y,int);p=np.asarray(p,int);s=np.asarray(s,float); normal=y==0
    try: auc=float(roc_auc_score(y,s))
    except: auc=float('nan')
    return dict(mcc=float(matthews_corrcoef(y,p)),f1=float(f1_score(y,p,zero_division=0)),
                precision=float(precision_score(y,p,zero_division=0)),recall=float(recall_score(y,p,zero_division=0)),
                far=float(np.mean(p[normal])) if normal.any() else np.nan,
                expected_cost=float((10*np.sum((y==1)&(p==0))+np.sum((y==0)&(p==1)))/len(y)),roc_auc=auc)

def local_far_error(y,p,window=50,alpha=ALPHA):
    idx=np.flatnonzero(np.asarray(y)==0); z=np.asarray(p)[idx]
    if len(z)<window: return np.nan
    vals=[abs(float(np.mean(z[i:i+window]))-alpha) for i in range(len(z)-window+1)]
    return float(max(vals))

def optimize_weights(comp,y,old=None,shrink=.02):
    comp=np.asarray(comp,float); y=np.asarray(y,float)
    if len(np.unique(y))<2: return np.array([1/3]*3) if old is None else old.copy()
    old=np.array([1/3]*3) if old is None else np.asarray(old,float)
    def obj(w):
        r=np.clip(comp@w,1e-6,1-1e-6)
        loss=-np.mean(10*y*np.log(r)+(1-y)*np.log(1-r))
        return loss+shrink*np.sum((w-old)**2)
    res=minimize(obj,old,method='SLSQP',bounds=[(0,1)]*3,constraints={'type':'eq','fun':lambda w:w.sum()-1},options={'maxiter':300})
    w=res.x if res.success else old
    w=np.clip(w,0,1); return w/w.sum()



def optimize_weights_grid(comp,y,old=None,step=.1,shrink=.05):
    comp=np.asarray(comp,float);y=np.asarray(y,int);old=np.ones(3)/3 if old is None else np.asarray(old,float)
    best=old.copy();bestv=float('inf')
    vals=np.arange(0,1+1e-9,step)
    for a in vals:
        for b in vals:
            c=1-a-b
            if c < -1e-9: continue
            w=np.array([a,b,max(0,c)])
            score=comp@w; pred=score>0.5
            v=(10*np.sum((y==1)&(~pred))+np.sum((y==0)&pred))/len(y)+shrink*np.sum((w-old)**2)
            if v<bestv:bestv=v;best=w
    return best/best.sum()

def delivered_cache(comp,seed,cad=(1,4,2),miss=(.0,.05,.05),max_age=(2,8,4)):
    rng=np.random.default_rng(seed);n=len(comp); latest=np.zeros((n,3));avail=np.zeros((n,3),bool)
    cur=comp[0].copy();last=np.zeros(3,int)
    for i in range(n):
        for j in range(3):
            if i%cad[j]==0 and rng.random()>miss[j]:cur[j]=comp[i,j];last[j]=i
        latest[i]=cur;avail[i]=[(i-last[j])<=max_age[j] for j in range(3)]
    return latest,avail

def fused_from_cache(latest,avail,w):
    ww=avail*w[None,:]; den=ww.sum(1); den[den==0]=1
    return (latest*ww).sum(1)/den

def run_threshold(method,scores,y,q0,delay=2,alpha=ALPHA,interval=25,buffer=60):
    n=len(scores); p=np.zeros(n,int); th=np.full(n,q0,float); normals=[]; q=q0; updates=0
    alpha_t=alpha; gsum=0.; D=1.; eta=D/math.sqrt(3)
    for i in range(n):
        th[i]=q;p[i]=int(scores[i]>q)
        r=i-delay
        if r>=0 and y[r]==0:
            x=float(scores[r]);err=int(x>th[r]);normals.append(x);normals=normals[-buffer:]
            if method=='Scheduled' and i%interval==0 and len(normals)>=20:
                q=conformal_q(normals,alpha);updates+=1
            elif method=='ACI':
                alpha_t=float(np.clip(alpha_t+.01*(alpha-err),1/61,.5))
                ref=np.sort(np.asarray(normals if len(normals)>=20 else [q0]))
                q=float(np.quantile(ref,1-alpha_t));updates+=1
            elif method=='Quantile tracking':
                q=float(np.clip(q+.015*(err-alpha),0,1));updates+=1
            elif method=='SF-OGD':
                grad=alpha-err;gsum+=grad*grad
                q=float(np.clip(q-eta*grad/max(math.sqrt(gsum),1e-12),0,1));updates+=1
    return p,th,updates


def aggregate_battery():
    d=pd.read_csv(DATA)
    rows=[]
    for (b,c),g in d.groupby(['Battery','id_cycle'],sort=True):
        t=g.Time.to_numpy(float);v=g.Voltage_measured.to_numpy(float);cur=g.Current_measured.to_numpy(float);temp=g.Temperature_measured.to_numpy(float)
        order=np.argsort(t);t=t[order];v=v[order];cur=cur[order];temp=temp[order]
        dt=np.diff(t,prepend=t[0]);
        rows.append(dict(Battery=b,cycle=int(c),capacity=float(g.Capacity.iloc[0]),duration=float(np.max(t)),
                         v_mean=float(v.mean()),v_min=float(v.min()),v_max=float(v.max()),v_std=float(v.std()),
                         v_early=float(v[:max(3,len(v)//10)].mean()),v_late=float(v[-max(3,len(v)//10):].mean()),
                         i_mean=float(cur.mean()),i_std=float(cur.std()),temp_mean=float(temp.mean()),temp_max=float(temp.max()),
                         temp_rise=float(temp.max()-temp[0]),energy_proxy=float(np.sum(np.abs(v*cur)*dt)/3600)))
    z=pd.DataFrame(rows).sort_values(['Battery','cycle']).reset_index(drop=True)
    for col in ['capacity','duration','v_mean','v_min','v_std','v_late','temp_mean','temp_max','temp_rise','energy_proxy']:
        z['lag_'+col]=z.groupby('Battery')[col].shift(1)
        z['delta_'+col]=z.groupby('Battery')[col].diff()
    z['age']=z.groupby('Battery').cumcount().astype(float)
    z['age_sqrt']=np.sqrt(z.age)
    z=z.dropna().reset_index(drop=True)
    max_cycle=z.groupby('Battery').cycle.transform('max')
    z['rul']=(max_cycle-z.cycle).clip(lower=0)
    z['label']=(z.rul<=20).astype(int)
    return z

FEATURES=['age','age_sqrt','lag_capacity','delta_capacity','duration','lag_duration','delta_duration',
          'v_mean','v_min','v_std','v_late','lag_v_mean','delta_v_mean','temp_mean','temp_max','temp_rise','lag_temp_mean','energy_proxy','lag_energy_proxy','delta_energy_proxy']

@dataclass
class BatteryModels:
    scaler:StandardScaler; iso:IsolationForest; rul:GradientBoostingRegressor; risk:LogisticRegression; healthy_ref:np.ndarray

def fit_models(train):
    X=train[FEATURES].to_numpy(float);sc=StandardScaler().fit(X);Z=sc.transform(X)
    healthy=train.rul.to_numpy()>40
    iso=IsolationForest(n_estimators=300,contamination='auto',random_state=11).fit(Z[healthy])
    raw=-iso.score_samples(Z[healthy]);ref=np.sort(raw)
    rul=GradientBoostingRegressor(n_estimators=150,max_depth=2,learning_rate=.035,loss='huber',random_state=12).fit(Z,train.rul)
    risk=LogisticRegression(max_iter=2000,class_weight={0:1,1:10},C=.5,random_state=13).fit(Z,train.label)
    return BatteryModels(sc,iso,rul,risk,ref)

def predict_models(m,d):
    Z=m.scaler.transform(d[FEATURES].to_numpy(float));raw=-m.iso.score_samples(Z)
    anom=np.searchsorted(m.healthy_ref,raw,side='right')/len(m.healthy_ref)
    rh=np.clip(m.rul.predict(Z),0,200); rr=expit((20-rh)/5)
    risk=m.risk.predict_proba(Z)[:,1]
    return np.column_stack([anom,rr,risk])

FOLDS=[('B0005','B0006',['B0007','B0018']),('B0006','B0007',['B0005','B0018']),('B0007','B0018',['B0005','B0006']),('B0018','B0005',['B0006','B0007'])]

def run_battery():
    d=aggregate_battery(); rows=[];trace=[];diag=[]
    for fi,(testb,calb,trainbs) in enumerate(FOLDS):
        tr=d[d.Battery.isin(trainbs)];cal=d[d.Battery==calb];te=d[d.Battery==testb]
        m=fit_models(tr);cc=predict_models(m,cal);tc=predict_models(m,te)
        w=optimize_weights(cc,cal.label.to_numpy())
        cl,ca=delivered_cache(cc,100+fi);tl,ta=delivered_cache(tc,200+fi)
        cs=fused_from_cache(cl,ca,w);ts=fused_from_cache(tl,ta,w)
        ref=cs[cal.label.to_numpy()==0];q0=conformal_q(ref)
        y=te.label.to_numpy(int)
        for meth in ['Static','Scheduled','ACI','Quantile tracking','SF-OGD']:
            if meth=='Static':p=(ts>q0).astype(int);th=np.full(len(ts),q0);upd=0
            else:p,th,upd=run_threshold(meth,ts,y,q0)
            mm=metrics(y,p,ts);mm['lfe50']=local_far_error(y,p,50)
            rows.append(dict(fold=fi,test_battery=testb,calibration_battery=calb,train_batteries='+'.join(trainbs),method=meth,updates=upd,**mm))
            trace.append(pd.DataFrame(dict(fold=fi,Battery=testb,cycle=te.cycle.to_numpy(),label=y,score=ts,pred=p,threshold=th,method=meth)))
        # Governed escalation: threshold + supervised weight review with chronological holdout.
        latest,avail=tl,ta; curw=w.copy();q=q0;p=np.zeros(len(y),int);th=np.zeros(len(y));w_hist=[];buf=[];w_updates=0;t_updates=0;escalations=0
        for i in range(len(y)):
            ww=curw*avail[i]; ww=ww/ww.sum() if ww.sum()>0 else np.ones(3)/3
            score=float(latest[i]@ww);th[i]=q;p[i]=score>q;w_hist.append(curw.copy())
            r=i-2
            if r>=0:
                buf.append((tc[r].copy(),int(y[r])));buf=buf[-70:]
            if i>0 and i%12==0 and len(buf)>=35:
                B=np.array([x for x,_ in buf]);Y=np.array([yy for _,yy in buf])
                split=max(24,int(.7*len(B))); Bt,Yt=B[:split],Y[:split];Bv,Yv=B[split:],Y[split:]
                if len(np.unique(Yt))==2 and len(np.unique(Yv))==2:
                    old_s=Bv@curw;old_p=(old_s>q).astype(int);old_cost=metrics(Yv,old_p,old_s)['expected_cost']
                    cand=optimize_weights_grid(Bt,Yt,curw,step=.1,shrink=.08);new_s=Bv@cand
                    normals=Bv[Yv==0]@cand
                    if len(normals)>=5:
                        candq=conformal_q(normals)
                        new_p=(new_s>candq).astype(int);new_cost=metrics(Yv,new_p,new_s)['expected_cost']
                        aucs=[]
                        for j in range(3):
                            try:aucs.append(roc_auc_score(Yv,Bv[:,j]))
                            except:aucs.append(.5)
                        try:fauc=roc_auc_score(Yv,old_s)
                        except:fauc=.5
                        if max(aucs)<.56: escalations+=1
                        elif new_cost<=old_cost*.95 and np.sum(abs(cand-curw))<=1.2:
                            curw=cand;q=candq;w_updates+=1;t_updates+=1
                # threshold-only adjustment if FAR too high and ranking still useful
                normal_scores=np.array([x@curw for x,yy in buf if yy==0])
                if len(normal_scores)>=25:
                    candq=conformal_q(normal_scores)
                    if abs(candq-q)/max(abs(q),.05)<.6:q=candq;t_updates+=1
        score_dyn=np.array([latest[i]@(w_hist[i]*avail[i]/max((w_hist[i]*avail[i]).sum(),1e-12)) for i in range(len(y))])
        mm=metrics(y,p,score_dyn);mm['lfe50']=local_far_error(y,p,50)
        rows.append(dict(fold=fi,test_battery=testb,calibration_battery=calb,train_batteries='+'.join(trainbs),method='Governed escalation',updates=t_updates,weight_updates=w_updates,model_escalations=escalations,**mm))
        trace.append(pd.DataFrame(dict(fold=fi,Battery=testb,cycle=te.cycle.to_numpy(),label=y,score=score_dyn,pred=p,threshold=th,method='Governed escalation')))
        diag.append(dict(fold=fi,test=testb,cal=calb,w_anomaly=w[0],w_rul=w[1],w_risk=w[2],n_train=len(tr),n_cal=len(cal),n_test=len(te)))
    R=pd.DataFrame(rows);T=pd.concat(trace,ignore_index=True);D=pd.DataFrame(diag)
    R.to_csv(OUT/'battery_fold_results.csv',index=False);T.to_csv(OUT/'battery_trace.csv',index=False);D.to_csv(OUT/'battery_diagnostics.csv',index=False)
    S=R.groupby('method').agg(mcc_mean=('mcc','mean'),mcc_sd=('mcc','std'),far_mean=('far','mean'),recall_mean=('recall','mean'),cost_mean=('expected_cost','mean'),auc_mean=('roc_auc','mean'),lfe50_mean=('lfe50','mean'),updates_mean=('updates','mean')).reset_index()
    S.to_csv(OUT/'battery_summary.csv',index=False)
    return d,R,T,D,S


def add_sfogd_to_cmapss():
    tr=pd.read_csv(ROOT/'revision'/'results'/'cmapss_crossfit_trace.csv')
    base=tr[tr.method=='Static matched'].copy();out=[]
    for (sub,fold),g in base.groupby(['subset','fold']):
        g=g.sort_values(['unit','cycle']).copy();q0=float(g.threshold.iloc[0]);y=g.label.to_numpy(int);s=g.score.to_numpy(float)
        p,th,upd=run_threshold('SF-OGD',s,y,q0,delay=40)
        g['method']='SF-OGD';g['pred']=p;g['threshold']=th;out.append(g)
    sf=pd.concat(out,ignore_index=True);sf.to_csv(OUT/'cmapss_sfogd_trace.csv',index=False)
    rows=[]
    for (sub,fold),g in sf.groupby(['subset','fold']):rows.append(dict(subset=sub,fold=fold,method='SF-OGD',updates=int((g.label==0).sum()),**metrics(g.label,g.pred,g.score),lfe100=local_far_error(g.label,g.pred,100)))
    r=pd.DataFrame(rows);r.to_csv(OUT/'cmapss_sfogd_fold_results.csv',index=False)
    s=r.groupby('subset').agg(mcc_mean=('mcc','mean'),mcc_sd=('mcc','std'),far_mean=('far','mean'),recall_mean=('recall','mean'),cost_mean=('expected_cost','mean'),lfe100_mean=('lfe100','mean')).reset_index();s.to_csv(OUT/'cmapss_sfogd_summary.csv',index=False)
    return r,s


def run_channel_failure_controlled():
    # Dedicated ranking-drift stress test: channel 3 loses discrimination after onset.
    rngs=range(10);rows=[]
    for seed in rngs:
        rng=np.random.default_rng(seed);n=10000;cal=2000;onset=5000;t=np.arange(n)
        failures=np.arange(2600,n-150,750)+rng.integers(-80,81,size=len(np.arange(2600,n-150,750)))
        dist=np.full(n,np.inf);nxt=np.inf;fs=set(map(int,failures))
        for i in range(n-1,-1,-1):
            if i in fs:nxt=i
            dist[i]=nxt-i
        y=((dist>=0)&(dist<=140)).astype(int);prox=np.clip(1-dist/140,0,1);prox[~np.isfinite(dist)]=0
        c1=expit(-3+2.3*prox+rng.normal(0,.38,n));c2=expit(-3.2+4.0*prox+rng.normal(0,.30,n));c3=expit(-3+4.8*prox+rng.normal(0,.28,n))
        # post-onset supervised channel becomes weak/noisy and biased upward
        c3[onset:]=expit(-1.8+.5*prox[onset:]+rng.normal(0,.65,n-onset))
        comp=np.c_[c1,c2,c3];w=optimize_weights(comp[:cal],y[:cal]);q0=conformal_q((comp[:cal]@w)[y[:cal]==0]);
        score=comp[cal:]@w;yy=y[cal:]
        for meth in ['Static','Scheduled','ACI','Quantile tracking','SF-OGD']:
            if meth=='Static':p=(score>q0).astype(int);th=np.full(len(score),q0);upd=0
            else:p,th,upd=run_threshold(meth,score,yy,q0,delay=80,interval=900,buffer=900)
            m=metrics(yy,p,score);m['post_mcc']=metrics(yy[onset-cal:],p[onset-cal:],score[onset-cal:])['mcc'];m['lfe100']=local_far_error(yy,p,100)
            rows.append(dict(seed=seed,method=meth,updates=upd,weight_updates=0,**m))
        # governed controller uses labels to reweight when ranking degrades
        curw=w.copy();q=q0;p=np.zeros(len(yy),int);s_dyn=np.zeros(len(yy));th=np.zeros(len(yy));buf=[];wu=tu=esc=0
        for i in range(len(yy)):
            s_dyn[i]=comp[cal+i]@curw;th[i]=q;p[i]=s_dyn[i]>q
            r=i-80
            if r>=0:buf.append((comp[cal+r].copy(),int(yy[r])));buf=buf[-1000:]
            if i>0 and i%100==0 and len(buf)>=400:
                B=np.array([x for x,_ in buf]);Y=np.array([z for _,z in buf]);sp=700;Bt,Yt=B[:sp],Y[:sp];Bv,Yv=B[sp:],Y[sp:]
                if len(np.unique(Yt))==2 and len(np.unique(Yv))==2:
                    old=Bv@curw;oldp=(old>q);om=metrics(Yv,oldp,old)
                    aucs=[roc_auc_score(Yv,Bv[:,j]) for j in range(3)];fauc=roc_auc_score(Yv,old)
                    bestj=int(np.argmax(aucs))
                    if max(aucs)<.58:
                        esc+=1
                    else:
                        # Reliability-shift proposal: strongly favor the best preserved channel,
                        # then validate on the chronological holdout.
                        target=np.zeros(3);target[bestj]=1.0
                        cand=.15*curw+.85*target
                        cand=cand/cand.sum()
                        ns=Bv@cand;norm=ns[Yv==0]
                        if len(norm)>=40:
                            nq=conformal_q(norm);nm=metrics(Yv,ns>nq,ns)
                            if (max(aucs)>fauc+.03 and (nm['expected_cost']<om['expected_cost']*.98 or nm['mcc']>om['mcc']+.02)):
                                curw=cand;q=nq;wu+=1;tu+=1
                normal=np.array([x@curw for x,z in buf if z==0])
                if len(normal)>200:
                    nq=conformal_q(normal)
                    if abs(nq-q)<.4:q=nq;tu+=1
        m=metrics(yy,p,s_dyn);m['post_mcc']=metrics(yy[onset-cal:],p[onset-cal:],s_dyn[onset-cal:])['mcc'];m['lfe100']=local_far_error(yy,p,100)
        rows.append(dict(seed=seed,method='Governed escalation',updates=tu,weight_updates=wu,model_escalations=esc,**m))
    R=pd.DataFrame(rows);R.to_csv(OUT/'channel_failure_results.csv',index=False)
    S=R.groupby('method').agg(mcc_mean=('mcc','mean'),post_mcc_mean=('post_mcc','mean'),far_mean=('far','mean'),cost_mean=('expected_cost','mean'),lfe100_mean=('lfe100','mean'),updates_mean=('updates','mean'),weight_updates_mean=('weight_updates','mean')).reset_index();S.to_csv(OUT/'channel_failure_summary.csv',index=False)
    return R,S


def main():
    d,R,T,D,S=run_battery(); sf,sfs=add_sfogd_to_cmapss(); cr,cs=run_channel_failure_controlled()
    meta={'created':'2026-07-08','battery_source':'NASA Li-ion Battery Aging Dataset, discharge-cycle CSV transport mirror','battery_sha256':hashlib.sha256(DATA.read_bytes()).hexdigest(),'battery_primary_folds':FOLDS,'event_horizon_cycles':20,'notes':['C-MAPSS remains simulated','battery data are physical laboratory cycles','SF-OGD is a bounded one-sided adaptation of Algorithm 2 to upper alarm thresholds','governed escalation is evaluated as a decision policy, not claimed to carry a conformal theorem']}
    (OUT/'metadata.json').write_text(json.dumps(meta,indent=2))
    print('\nBATTERY\n',S.round(4).to_string(index=False));print('\nSF C-MAPSS\n',sfs.round(4).to_string(index=False));print('\nCHANNEL FAILURE\n',cs.round(4).to_string(index=False))
if __name__=='__main__':main()
