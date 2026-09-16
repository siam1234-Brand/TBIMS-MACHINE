"""Round-3 reviewer items.
  3. Confidence interval on the LOS-conditional estimate across the SAME 20 splits
  4. Resolve the 0.489 (linear residualisation) vs 0.762 (quintile) discrepancy
     using decile stratification and a nested likelihood-ratio test
  + the incremental concordance attributable to modelling PTA rather than LOS
Run from a directory containing Form1.csv.
"""
import numpy as np, pandas as pd, warnings, json
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sksurv.metrics import concordance_index_censored
from lifelines import CoxPHFitter
from scipy.stats import spearmanr, chi2, ttest_1samp
import xgboost as xgb

SEED = 42; np.random.seed(SEED); OUT = {}

f1 = pd.read_csv("Form1.csv", low_memory=False)
GEN = [66,77,88,99,666,777,888,999,6666,9999]
def num(c, sent=GEN):
    s = pd.to_numeric(f1[c], errors="coerce"); return s.mask(s.isin(sent))
d = pd.DataFrame({"Mod1Id": f1.Mod1Id})
age = pd.to_numeric(f1.AGENoPHI, errors="coerce")
d["age"]=age.where(age.between(0,88)); d["age_topcoded"]=(age==777).astype(int)
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
LOS=df.rehab_los.values
def C(ev,t,p): return concordance_index_censored(ev,t,p)[0]
def gb(Xd,y,**kw):
    p=dict(n_estimators=300,learning_rate=.05,max_depth=3,subsample=.8,
           colsample_bytree=.8,reg_lambda=1.,random_state=SEED); p.update(kw)
    return xgb.XGBRegressor(**p).fit(Xd,y)

print("="*74)
print("ROUND 3 -- LOS SEPARABILITY ACROSS 20 SPLITS (same seeds as Round 2)")
print("="*74)
rows=[]
for r in range(20):
    a,b=train_test_split(np.arange(len(df)),test_size=.2,random_state=1000+r,stratify=E)
    im=SimpleImputer(strategy="median").fit(X.iloc[a])
    s_=StandardScaler().fit(im.transform(X.iloc[a]))
    Xa=pd.DataFrame(s_.transform(im.transform(X.iloc[a])),columns=FEAT)[TOP]
    Xb=pd.DataFrame(s_.transform(im.transform(X.iloc[b])),columns=FEAT)[TOP]
    # PTA survival model
    r_pta=xgb.XGBRegressor(objective="survival:cox",n_estimators=300,learning_rate=.05,
        max_depth=3,subsample=.8,colsample_bytree=.8,reg_lambda=1.,
        random_state=SEED).fit(Xa,np.where(E[a],T[a],-T[a])).predict(Xb)
    # LOS model (train rows with observed LOS)
    ok_a=~np.isnan(LOS[a]); ok_b=~np.isnan(LOS[b])
    los_m=gb(Xa[ok_a],np.log(np.clip(LOS[a][ok_a],1,None)))
    p_los=los_m.predict(Xb)
    Eb,Tb=E[b],T[b]
    c_pta=C(Eb,Tb,r_pta); c_los=C(Eb,Tb,-p_los); inc=c_pta-c_los
    # stratified conditional concordance
    def strat_C(nq):
        q=pd.qcut(p_los,nq,labels=False,duplicates="drop"); vals=[]
        for k in np.unique(q):
            m=q==k
            if Eb[m].sum()>15: vals.append(C(Eb[m],Tb[m],r_pta[m]))
        return float(np.mean(vals))
    c_q5, c_q10 = strat_C(5), strat_C(10)
    # nested likelihood-ratio test on the test split
    dd=pd.DataFrame({"pl":(p_los-p_los.mean())/p_los.std(),
                     "rp":(r_pta-r_pta.mean())/r_pta.std(),
                     "T":Tb,"E":Eb.astype(int)})
    m0=CoxPHFitter().fit(dd[["pl","T","E"]],"T","E")
    m1=CoxPHFitter().fit(dd[["pl","rp","T","E"]],"T","E")
    lr=2*(m1.log_likelihood_-m0.log_likelihood_); p_lr=chi2.sf(lr,1)
    rows.append((c_pta,c_los,inc,c_q5,c_q10,lr,p_lr,
                 spearmanr(r_pta,p_los).statistic))
    print(f"  split {r+1:2d}: PTA {c_pta:.4f} LOS {c_los:.4f} inc {inc:+.4f} "
          f"| Q5 {c_q5:.4f} Q10 {c_q10:.4f} | LR {lr:8.1f} p={p_lr:.1e}")
A=np.array(rows)
def ci(v): return v.mean(), np.percentile(v,2.5), np.percentile(v,97.5)
names=["PTA model C","LOS-only C","increment","within-quintile C","within-decile C"]
print("\n  quantity                 mean    2.5%    97.5%")
for i,n in enumerate(names):
    m,lo,hi=ci(A[:,i]); print(f"  {n:22s} {m:7.4f} {lo:7.4f} {hi:7.4f}")
print(f"  Spearman(PTA,LOS)      {A[:,7].mean():7.3f} {np.percentile(A[:,7],2.5):7.3f} {np.percentile(A[:,7],97.5):7.3f}")
print(f"\n  Nested LR statistic: mean {A[:,5].mean():.1f}; max p across splits {A[:,6].max():.2e}")
print(f"  -> PTA score adds information beyond predicted LOS in "
      f"{int((A[:,6]<0.05).sum())}/20 splits")
print(f"\n  increment mean {A[:,2].mean():+.4f}  t-test p={ttest_1samp(A[:,2],0).pvalue:.2e}")

OUT["los_r3"]={
 "c_pta":float(A[:,0].mean()),"c_los":float(A[:,1].mean()),
 "inc":float(A[:,2].mean()),"inc_lo":float(np.percentile(A[:,2],2.5)),
 "inc_hi":float(np.percentile(A[:,2],97.5)),
 "q5":float(A[:,3].mean()),"q5_lo":float(np.percentile(A[:,3],2.5)),
 "q5_hi":float(np.percentile(A[:,3],97.5)),
 "q10":float(A[:,4].mean()),"q10_lo":float(np.percentile(A[:,4],2.5)),
 "q10_hi":float(np.percentile(A[:,4],97.5)),
 "lr_mean":float(A[:,5].mean()),"lr_maxp":float(A[:,6].max()),
 "lr_sig_splits":int((A[:,6]<0.05).sum()),
 "rho":float(A[:,7].mean())}
json.dump(OUT,open("round3_analyses.json","w"),indent=1)
print("\nSaved round3_analyses.json")
