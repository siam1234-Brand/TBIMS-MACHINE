"""Round-2 reviewer items.
  1. ML-vs-Cox uncertainty from REPEATED SPLITS (not fixed-test bootstrap)
  2. Extended penalty grid so the optimum is interior
  3. LOS-proxy test: what survives after conditioning on predicted length of stay
  4. Fine-Gray subdistribution models vs cause-specific models
Run from a directory containing Form1.csv.
"""
import numpy as np, pandas as pd, warnings, json
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sksurv.metrics import concordance_index_censored
from lifelines import CoxPHFitter
from scipy.stats import spearmanr
import xgboost as xgb

SEED = 42; np.random.seed(SEED); OUT = {}

# ---------------------------------------------------------------- build data
f1 = pd.read_csv("Form1.csv", low_memory=False)
GEN = [66,77,88,99,666,777,888,999,6666,9999]
def num(c, sent=GEN):
    s = pd.to_numeric(f1[c], errors="coerce"); return s.mask(s.isin(sent))
d = pd.DataFrame({"Mod1Id": f1.Mod1Id})
age = pd.to_numeric(f1.AGENoPHI, errors="coerce")
d["age"] = age.where(age.between(0,88)); d["age_topcoded"]=(age==777).astype(int)
d.loc[age==777,"age"]=89
d["male"]=f1.SexF.map({1:0,2:1}); d["edu_years"]=num("EduYears"); d["inj_year"]=num("INJYEAR",[9999])
gcs=pd.to_numeric(f1.GCSTot,errors="coerce")
d["gcs"]=gcs.mask(gcs.isin([77,88,999])); d["gcs_sedated"]=gcs.isin([77,88]).astype(int)
tfc=pd.to_numeric(f1.TFCDays,errors="coerce")
d["tfc_days"]=tfc.mask(tfc.isin([7777,9999])); d["tfc_never"]=(tfc==7777).astype(int)
d["sci"]=num("SCI")
d["days_to_acute"]=num("DAYStoACUTEadm",[9999]); d["days_to_rehab"]=num("DAYStoREHABadm",[9999])
d["rehab_dc_day"]=num("DAYStoREHABdc",[9999]); d["rehab_los"]=d.rehab_dc_day-d.days_to_rehab
for a,b in [("FIMTOTA","fim_tot_adm"),("FIMMOTA","fim_mot_adm"),("FIMCOGA","fim_cog_adm"),("DRSa","drs_adm")]:
    d[b]=num(a)
d["cause"]=f1.Cause.mask(f1.Cause==999); d["payor"]=f1.RehabPay1.mask(f1.RehabPay1.isin([55,999]))
d["emp_preinj"]=f1.Emp1.mask(f1.Emp1.isin([666,777,888,999])); d["res_preinj"]=f1.ResInj.mask(f1.ResInj==999)
pta=pd.to_numeric(f1.PTADays,errors="coerce")
d["event"]=(~pta.isin([8888,9999]) & pta.notna()).astype(int)
d["duration"]=np.where(d.event==1,pta,d.rehab_dc_day)
d.loc[pta==9999,["event","duration"]]=np.nan
df=d[d.duration.notna()&(d.duration>0)].reset_index(drop=True); df["event"]=df.event.astype(int)

FEAT=["age","age_topcoded","male","edu_years","inj_year","gcs","gcs_sedated","tfc_days",
      "tfc_never","sci","days_to_acute","days_to_rehab","fim_tot_adm","fim_mot_adm",
      "fim_cog_adm","drs_adm","cause","payor","emp_preinj","res_preinj"]
TOP=["days_to_rehab","drs_adm","fim_cog_adm","tfc_days","fim_tot_adm","gcs",
     "fim_mot_adm","inj_year","age","cause","gcs_sedated","edu_years"]
T=df.duration.values; E=df.event.values.astype(bool); X=df[FEAT].astype(float)
def prep(itr, ite):
    im=SimpleImputer(strategy="median").fit(X.iloc[itr])
    s_=StandardScaler().fit(im.transform(X.iloc[itr]))
    return (pd.DataFrame(s_.transform(im.transform(X.iloc[itr])),columns=FEAT),
            pd.DataFrame(s_.transform(im.transform(X.iloc[ite])),columns=FEAT))
def C(ev,t,p): return concordance_index_censored(ev,t,p)[0]
def fit_xgb(Xd,t,e,**kw):
    p=dict(objective="survival:cox",n_estimators=300,learning_rate=.05,max_depth=3,
           subsample=.8,colsample_bytree=.8,reg_lambda=1.,random_state=SEED); p.update(kw)
    return xgb.XGBRegressor(**p).fit(Xd,np.where(e,t,-t))

# ============================================================ ITEM 2 first
print("="*72); print("ITEM 2 -- EXTENDED PENALTY GRID (is the optimum interior?)")
print("="*72)
itr,ite=train_test_split(np.arange(len(df)),test_size=.2,random_state=SEED,stratify=E)
Xtr,Xte=prep(itr,ite)
GRID=[1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1.0]
sb=np.random.RandomState(SEED).choice(len(itr),4000,replace=False)
scores={}
for pen in GRID:
    tr=Xtr[TOP].copy(); tr["T"]=T[itr]; tr["E"]=E[itr].astype(int)
    m=CoxPHFitter(penalizer=pen).fit(tr,"T","E")
    scores[pen]=C(E[itr][sb],T[itr][sb],m.predict_partial_hazard(Xtr[TOP].iloc[sb]).values)
for p_,v in scores.items(): print(f"   penalizer {p_:<8g} C={v:.4f}")
best_pen=max(scores,key=scores.get)
interior = GRID.index(best_pen) not in (0,len(GRID)-1)
print(f"  best penalizer = {best_pen:g} | interior optimum: {interior}")
OUT["penalty"]={"grid":GRID,"scores":{str(k):float(v) for k,v in scores.items()},
                "best":float(best_pen),"interior":bool(interior)}

# ============================================================ ITEM 1
print("\n"+"="*72); print("ITEM 1 -- REPEATED SPLITS (honest uncertainty for XGB - Cox)")
print("="*72)
R=20; rows=[]
for r in range(R):
    a,b=train_test_split(np.arange(len(df)),test_size=.2,random_state=1000+r,stratify=E)
    Xa,Xb=prep(a,b)
    tr=Xa[TOP].copy(); tr["T"]=T[a]; tr["E"]=E[a].astype(int)
    cox=CoxPHFitter(penalizer=best_pen).fit(tr,"T","E")
    c_cox=C(E[b],T[b],cox.predict_partial_hazard(Xb[TOP]).values)
    c_xgb=C(E[b],T[b],fit_xgb(Xa[TOP],T[a],E[a]).predict(Xb[TOP]))
    rows.append((c_cox,c_xgb,c_xgb-c_cox))
    print(f"   split {r+1:2d}: Cox {c_cox:.4f}  XGB {c_xgb:.4f}  diff {c_xgb-c_cox:+.4f}")
arr=np.array(rows); dif=arr[:,2]
lo,hi=np.percentile(dif,[2.5,97.5])
from scipy.stats import ttest_1samp, wilcoxon
tp=ttest_1samp(dif,0).pvalue; wp=wilcoxon(dif).pvalue
print(f"\n  Cox  mean C = {arr[:,0].mean():.4f} (sd {arr[:,0].std():.4f})")
print(f"  XGB  mean C = {arr[:,1].mean():.4f} (sd {arr[:,1].std():.4f})")
print(f"  diff mean   = {dif.mean():+.4f}  95% range [{lo:+.4f},{hi:+.4f}]"
      f"  t-test p={tp:.2e}  Wilcoxon p={wp:.2e}")
print(f"  -> zero excluded: {'YES' if lo>0 or hi<0 else 'NO'}")
OUT["repeated"]={"R":R,"cox":float(arr[:,0].mean()),"xgb":float(arr[:,1].mean()),
  "cox_sd":float(arr[:,0].std()),"xgb_sd":float(arr[:,1].std()),"diff":float(dif.mean()),
  "lo":float(lo),"hi":float(hi),"t_p":float(tp),"wilcoxon_p":float(wp),
  "excludes_zero":bool(lo>0 or hi<0)}

# ============================================================ ITEM 3
print("\n"+"="*72); print("ITEM 3 -- IS THIS A LENGTH-OF-STAY PROXY?")
print("="*72)
los=df.rehab_los.values
mask=~np.isnan(los)
los_tr=[i for i in itr if mask[i]]; los_te=[i for i in ite if mask[i]]
los_model=xgb.XGBRegressor(n_estimators=300,learning_rate=.05,max_depth=3,subsample=.8,
    colsample_bytree=.8,reg_lambda=1.,random_state=SEED).fit(
    Xtr[TOP].loc[[list(itr).index(i) for i in los_tr]], np.log(np.clip(los[los_tr],1,None)))
pos=[list(ite).index(i) for i in los_te]
pred_los=los_model.predict(Xte[TOP].iloc[pos])
r_pta=fit_xgb(Xtr[TOP],T[itr],E[itr]).predict(Xte[TOP])[pos]
Ete=E[ite][pos]; Tte=T[ite][pos]
c_raw=C(Ete,Tte,r_pta)
print(f"  PTA model C on this subset            : {c_raw:.3f}")
print(f"  C of predicted-LOS alone as a score   : {C(Ete,Tte,-pred_los):.3f}")
print(f"  Spearman(PTA risk, predicted LOS)     : {spearmanr(r_pta,pred_los).statistic:+.3f}")
# residualise the PTA score on predicted LOS
b1=np.polyfit(pred_los,r_pta,1); resid=r_pta-np.polyval(b1,pred_los)
print(f"  C of PTA score residualised on pred LOS: {C(Ete,Tte,resid):.3f}")
# within-stratum concordance (5 strata of predicted LOS)
q=pd.qcut(pred_los,5,labels=False); strat=[]
for k in range(5):
    m_=q==k
    if E[ite][pos][m_].sum()>20: strat.append(C(Ete[m_],Tte[m_],r_pta[m_]))
print(f"  Mean within-stratum C (5 LOS strata)  : {np.mean(strat):.3f}  (per stratum: "
      + ", ".join(f"{v:.3f}" for v in strat) + ")")
OUT["los_proxy"]={"c_raw":float(c_raw),"c_predlos_alone":float(C(Ete,Tte,-pred_los)),
  "rho":float(spearmanr(r_pta,pred_los).statistic),"c_residualised":float(C(Ete,Tte,resid)),
  "c_within_strata":float(np.mean(strat)),"strata":[float(v) for v in strat]}

# ============================================================ ITEM 4
print("\n"+"="*72); print("ITEM 4 -- FINE-GRAY SUBDISTRIBUTION vs CAUSE-SPECIFIC")
print("="*72)
# No administrative censoring here: every subject either emerges (event 1) or is
# discharged still in PTA (competing event 2). The Fine-Gray subdistribution risk
# set therefore retains competing-event subjects indefinitely, which is obtained
# by setting their time to beyond the last observed event and marking them censored.
TMAX=T.max()+1
T_sd=np.where(E,T,TMAX); E_sd=E.copy()          # subdistribution recoding
def eval_pair(a,b):
    Xa,Xb=prep(a,b)
    cs=fit_xgb(Xa[TOP],T[a],E[a])                       # cause-specific
    sd=fit_xgb(Xa[TOP],T_sd[a],E_sd[a])                 # subdistribution (Fine-Gray)
    r_cs=cs.predict(Xb[TOP]); r_sd=sd.predict(Xb[TOP])
    # evaluate both on the SUBDISTRIBUTION scale: who emerges before discharge
    return (C(E_sd[b],T_sd[b],r_cs), C(E_sd[b],T_sd[b],r_sd),
            C(E[b],T[b],r_cs),       C(E[b],T[b],r_sd))
res=[]
for r in range(10):
    a,b=train_test_split(np.arange(len(df)),test_size=.2,random_state=2000+r,stratify=E)
    res.append(eval_pair(a,b))
res=np.array(res)
print(f"  Evaluated on SUBDISTRIBUTION scale (emergence before discharge):")
print(f"    cause-specific model : {res[:,0].mean():.4f} (sd {res[:,0].std():.4f})")
print(f"    Fine-Gray model      : {res[:,1].mean():.4f} (sd {res[:,1].std():.4f})")
dsd=res[:,1]-res[:,0]
print(f"    difference           : {dsd.mean():+.4f}  [{np.percentile(dsd,2.5):+.4f},"
      f"{np.percentile(dsd,97.5):+.4f}]  p={ttest_1samp(dsd,0).pvalue:.2e}")
print(f"  Evaluated on CAUSE-SPECIFIC scale:")
print(f"    cause-specific model : {res[:,2].mean():.4f}")
print(f"    Fine-Gray model      : {res[:,3].mean():.4f}")
OUT["finegray"]={"sd_cause":float(res[:,0].mean()),"sd_fg":float(res[:,1].mean()),
  "sd_diff":float(dsd.mean()),"sd_lo":float(np.percentile(dsd,2.5)),
  "sd_hi":float(np.percentile(dsd,97.5)),"sd_p":float(ttest_1samp(dsd,0).pvalue),
  "cs_cause":float(res[:,2].mean()),"cs_fg":float(res[:,3].mean())}

json.dump(OUT,open("round2_analyses.json","w"),indent=1)
print("\nSaved round2_analyses.json")
