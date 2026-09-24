"""
camera_ready_checks.py -- numbers added for the camera-ready version, computed with the
ORIGINAL pipeline of this repository (same cleaning, features, top-12, seeds and models as
tbims_paper_complete.py / round2_fixes.py / round3_fixes.py). Run from a directory containing
Form1.csv.  Writes camera_ready_checks.json.

  1. risk-quartile Kaplan-Meier on the primary split (completes Sec. IV-D)
  2. standard (unnormalised) IPCW median next to the weighted Kaplan-Meier of emerged participants
  3. 20 repeated splits: Cox vs XGBoost, length-of-stay contrast, within-stratum concordance of
     BOTH scores (the stay score is the control), nested LRT with a convergence check, and the
     Nadeau-Bengio corrected resampled t-test
  4. landmark sensitivity: participants still in PTA after the rehabilitation-admission day
  5. descriptive facts used in the text (GCS missingness, prior emergence, sentinel counts)
"""
import json, warnings
import numpy as np, pandas as pd
from scipy.stats import t as tdist, chi2, spearmanr
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sksurv.metrics import concordance_index_censored
from lifelines import CoxPHFitter, KaplanMeierFitter
import xgboost as xgb
warnings.filterwarnings("ignore")
SEED = 42; OUT = {}

# ------------------------------------------------ data: identical to round2/round3_fixes.py
f1 = pd.read_csv("Form1.csv", low_memory=False)
GEN = [66,77,88,99,666,777,888,999,6666,9999]
def num(c, sent=GEN):
    s = pd.to_numeric(f1[c], errors="coerce"); return s.mask(s.isin(sent))
d = pd.DataFrame({"Mod1Id": f1.Mod1Id})
age = pd.to_numeric(f1.AGENoPHI, errors="coerce")
d["age"] = age.where(age.between(0,88)); d["age_topcoded"]=(age==777).astype(int); d.loc[age==777,"age"]=89
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
pta=pd.to_numeric(f1.PTADays,errors="coerce"); d["pta_raw"]=pta; d["gcs_raw"]=gcs
d["event"]=(~pta.isin([8888,9999]) & pta.notna()).astype(int)
d["duration"]=np.where(d.event==1,pta,d.rehab_dc_day)
d.loc[pta==9999,["event","duration"]]=np.nan
df=d[d.duration.notna()&(d.duration>0)].reset_index(drop=True); df["event"]=df.event.astype(int)
FEAT=["age","age_topcoded","male","edu_years","inj_year","gcs","gcs_sedated","tfc_days","tfc_never","sci",
      "days_to_acute","days_to_rehab","fim_tot_adm","fim_mot_adm","fim_cog_adm","drs_adm","cause","payor",
      "emp_preinj","res_preinj"]
TOP=["days_to_rehab","drs_adm","fim_cog_adm","tfc_days","fim_tot_adm","gcs","fim_mot_adm","inj_year","age",
     "cause","gcs_sedated","edu_years"]
BEST_PEN = 3e-5                       # the penalty used in round2_fixes.py
T=df.duration.values.astype(float); E=df.event.values.astype(bool); X=df[FEAT].astype(float)
def C(ev,t,p): return concordance_index_censored(np.asarray(ev,bool),np.asarray(t,float),np.asarray(p,float))[0]
def prep(Xd, a, b):
    im=SimpleImputer(strategy="median").fit(Xd.iloc[a]); s_=StandardScaler().fit(im.transform(Xd.iloc[a]))
    f=lambda i: pd.DataFrame(s_.transform(im.transform(Xd.iloc[i])),columns=Xd.columns)
    return f(a), f(b)
P=dict(n_estimators=300,learning_rate=.05,max_depth=3,subsample=.8,colsample_bytree=.8,reg_lambda=1.,random_state=SEED,n_jobs=1)
def fit_xgb(Xd,t,e): return xgb.XGBRegressor(objective="survival:cox",**P).fit(Xd,np.where(e,t,-t))
def corrected_t(v, ntr, nte):
    v=np.asarray(v,float); R=len(v); tt=v.mean()/np.sqrt((1/R+nte/ntr)*v.var(ddof=1))
    return float(tt), float(2*tdist.sf(abs(tt),R-1))

# ------------------------------------------------ 5. descriptive facts
ev=df.event.values==1; adm=df.days_to_rehab.values; P_=df.pta_raw.values
OUT["facts"]={"n":len(df),"events":int(ev.sum()),
  "emerged_before_adm":int((ev&(P_<adm)).sum()),"emerged_on_adm":int((ev&(P_==adm)).sum()),
  "tfc_after_adm":int((df.tfc_days>df.days_to_rehab).sum()),
  "gcs_missing":int(df.gcs.isna().sum()),"gcs_sedated_intubated":int(df.gcs_sedated.sum()),
  "gcs_missing_pct_emerged":float(100*df.gcs[ev].isna().mean()),"gcs_missing_pct_censored":float(100*df.gcs[~ev].isna().mean()),
  "fimtot_valid_66_99_masked":int(pd.to_numeric(f1.FIMTOTA,errors="coerce").isin([66,77,88,99]).sum()),
  "fimmot_valid_66_88_masked":int(pd.to_numeric(f1.FIMMOTA,errors="coerce").isin([66,77,88]).sum())}

# ------------------------------------------------ 1. quartiles on the primary split (seed 42)
itr,ite=train_test_split(np.arange(len(df)),test_size=.2,random_state=SEED,stratify=E)
Xtr,Xte=prep(X,itr,ite)
r_te=fit_xgb(Xtr[TOP],T[itr],E[itr]).predict(Xte[TOP])
q=pd.qcut(r_te,4,labels=False); quart={}
for k,lab in [(3,"Q1_fastest"),(2,"Q2"),(1,"Q3"),(0,"Q4_slowest")]:
    m=q==k; km=KaplanMeierFitter().fit(T[ite][m],E[ite][m])
    quart[lab]={"s60":float(km.survival_function_at_times(60).values[0]),
                "s180":float(km.survival_function_at_times(180).values[0]),"median":float(km.median_survival_time_)}
OUT["quartiles"]=quart

# ------------------------------------------------ 2. IPCW: same censoring model as CELL 15
Xall=pd.DataFrame(StandardScaler().fit(SimpleImputer(strategy="median").fit(X.iloc[itr]).transform(X.iloc[itr]))
                  .transform(SimpleImputer(strategy="median").fit(X.iloc[itr]).transform(X)),columns=FEAT)[TOP]
cd_=Xall.copy(); cd_["T"]=T; cd_["Ecens"]=(1-E).astype(int)
Gt=CoxPHFitter(penalizer=0.1).fit(cd_,"T","Ecens").predict_survival_function(Xall)
G_at_T=np.array([np.interp(T[i],Gt.index.values,Gt.iloc[:,i].values) for i in range(len(T))])
w=np.where(E,1.0/np.clip(G_at_T,0.02,1.0),0.0)
wkm=KaplanMeierFitter().fit(T[E],np.ones(E.sum()),weights=w[E]).median_survival_time_
o=np.argsort(T); F=np.cumsum(w[o])/len(T); ht=float(T[o][np.searchsorted(F,0.5)])
OUT["ipcw"]={"weighted_km_of_emerged":float(wkm),"standard_unnormalised":ht,"estimated_mass":float(F[-1]),
             "km":float(KaplanMeierFitter().fit(T,E).median_survival_time_),"complete_case":float(np.median(T[E])),
             "km_still_in_pta_at_last_time":float(KaplanMeierFitter().fit(T,E).survival_function_.iloc[-1,0])}

# ------------------------------------------------ 3/4. repeated splits
def one_split(Tv,Ev,Xv,LOS,a,b):
    Xa,Xb=prep(Xv,a,b); A_,B_=Xa[TOP],Xb[TOP]
    tr=A_.copy(); tr["T"]=Tv[a]; tr["E"]=Ev[a].astype(int)
    r_c=CoxPHFitter(penalizer=BEST_PEN).fit(tr,"T","E").predict_partial_hazard(B_).values
    r_x=fit_xgb(A_,Tv[a],Ev[a]).predict(B_)
    ok=~np.isnan(LOS[a])
    pl=xgb.XGBRegressor(**P).fit(A_[ok],np.log(np.clip(LOS[a][ok],1,None))).predict(B_)
    Eb,Tb=Ev[b],Tv[b]; out={"cox":C(Eb,Tb,r_c),"xgb":C(Eb,Tb,r_x),"los":C(Eb,Tb,-pl)}
    out["diff"]=out["xgb"]-out["cox"]; out["inc"]=out["xgb"]-out["los"]
    for nq in (5,10):
        qq=pd.qcut(pl,nq,labels=False,duplicates="drop"); vp,vl=[],[]
        for k in np.unique(qq):
            m=qq==k
            if Eb[m].sum()>15: vp.append(C(Eb[m],Tb[m],r_x[m])); vl.append(C(Eb[m],Tb[m],-pl[m]))
        out[f"q{nq}_pta"]=float(np.mean(vp)); out[f"q{nq}_los"]=float(np.mean(vl))
    z=lambda v:(v-v.mean())/v.std()
    dd=pd.DataFrame({"pl":z(pl),"rp":z(r_x),"T":Tb,"E":Eb.astype(int)})
    m0=CoxPHFitter().fit(dd[["pl","T","E"]],"T","E"); m1=CoxPHFitter().fit(dd[["pl","rp","T","E"]],"T","E")
    lr=2*(m1.log_likelihood_-m0.log_likelihood_)
    out["lr"]=float(lr); out["lr_valid"]=bool(lr>0); out["lr_p"]=float(chi2.sf(lr,1)) if lr>0 else None
    out["rho"]=float(spearmanr(r_x,pl).statistic)
    return out, len(a), len(b)
def summarise(rows, ntr, nte):
    s={k:{"mean":float(np.mean([r[k] for r in rows])),"lo":float(np.percentile([r[k] for r in rows],2.5)),
          "hi":float(np.percentile([r[k] for r in rows],97.5))} for k in rows[0] if k not in ("lr_valid","lr_p")}
    s["diff_t"]=corrected_t([r["diff"] for r in rows],ntr,nte); s["inc_t"]=corrected_t([r["inc"] for r in rows],ntr,nte)
    valid=[r for r in rows if r["lr_valid"]]
    s["lr_valid_splits"]=len(valid); s["lr_sig_among_valid"]=int(sum(r["lr_p"]<.05 for r in valid))
    s["lr_mean_valid"]=float(np.mean([r["lr"] for r in valid])); s["share"]=(s["los"]["mean"]-.5)/(s["xgb"]["mean"]-.5)
    return s
LOSv=df.rehab_los.values.astype(float); rows=[]
for r in range(1000,1020):
    a,b=train_test_split(np.arange(len(df)),test_size=.2,random_state=r,stratify=E)
    o_,ntr,nte=one_split(T,E,X,LOSv,a,b); rows.append(o_); print("R",r,{k:round(v,3) for k,v in o_.items() if isinstance(v,float)},flush=True)
OUT["repeated"]=summarise(rows,ntr,nte); OUT["repeated_rows"]=rows
# landmark: still in PTA after the admission day, time from admission
keep=(~ev | (P_>adm)) & ~np.isnan(adm)
Tl=np.where(ev,P_-adm,df.rehab_dc_day.values-adm).astype(float)
late=ev&(P_>df.rehab_dc_day.values); El=ev.copy(); El[late]=False; Tl[late]=(df.rehab_dc_day.values-adm)[late]
keep&=Tl>0; idx=np.where(keep)[0]
TL,EL,XL,LL=Tl[idx],El[idx],X.iloc[idx].reset_index(drop=True),LOSv[idx]
rows_l=[]
for r in range(1000,1020):
    a,b=train_test_split(np.arange(len(idx)),test_size=.2,random_state=r,stratify=EL)
    o_,ntr,nte=one_split(TL,EL,XL,LL,a,b); rows_l.append(o_); print("L",r,round(o_["xgb"],3),round(o_["los"],3),flush=True)
OUT["landmark"]={"n":int(len(idx)),"events":int(EL.sum()),**summarise(rows_l,ntr,nte)}

# ------------------------------------------------ 6. coding checks on the primary split (seed 42)
def c_primary(Xd):
    a_,b_=prep(Xd,itr,ite); return C(E[ite],T[ite],fit_xgb(a_[TOP],T[itr],E[itr]).predict(b_[TOP]))
X_fim=X.copy()
for raw,col,sent in [("FIMTOTA","fim_tot_adm",[9999]),("FIMMOTA","fim_mot_adm",[999])]:
    v=pd.to_numeric(f1.set_index("Mod1Id").loc[df.Mod1Id,raw],errors="coerce").values
    X_fim[col]=np.where(np.isin(v,sent),np.nan,v)
X_tfc=X_fim.copy(); tv=X_tfc["tfc_days"].values; av=df.days_to_rehab.values
X_tfc["tfc_days"]=np.where(np.isnan(tv),np.nan,np.minimum(tv,av))
OUT["coding_checks"]={"original":c_primary(X),"fim_codes_fixed":c_primary(X_fim),"plus_tfc_truncated":c_primary(X_tfc)}
print("coding checks",OUT["coding_checks"])
json.dump(OUT,open("camera_ready_checks.json","w"),indent=1)
print("saved camera_ready_checks.json")
