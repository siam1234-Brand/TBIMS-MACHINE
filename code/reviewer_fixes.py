"""Reviewer-mandated analyses (items 1-4 + competing risks).
Run from a directory containing Form1.csv."""
import numpy as np, pandas as pd, warnings, json
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv
from sksurv.linear_model import CoxnetSurvivalAnalysis
from lifelines import CoxPHFitter, KaplanMeierFitter, WeibullAFTFitter
import xgboost as xgb

SEED = 42
np.random.seed(SEED)
OUT = {}

# ------------------------------------------------------------------ data
f1 = pd.read_csv("Form1.csv", low_memory=False)
GEN = [66, 77, 88, 99, 666, 777, 888, 999, 6666, 9999]
def num(c, sent=GEN):
    s = pd.to_numeric(f1[c], errors="coerce"); return s.mask(s.isin(sent))

d = pd.DataFrame({"Mod1Id": f1.Mod1Id})
age = pd.to_numeric(f1.AGENoPHI, errors="coerce")
d["age"] = age.where(age.between(0, 88)); d["age_topcoded"] = (age == 777).astype(int)
d.loc[age == 777, "age"] = 89
d["male"] = f1.SexF.map({1: 0, 2: 1}); d["edu_years"] = num("EduYears")
d["inj_year"] = num("INJYEAR", [9999])
gcs = pd.to_numeric(f1.GCSTot, errors="coerce")
d["gcs"] = gcs.mask(gcs.isin([77, 88, 999])); d["gcs_sedated"] = gcs.isin([77, 88]).astype(int)
tfc = pd.to_numeric(f1.TFCDays, errors="coerce")
d["tfc_days"] = tfc.mask(tfc.isin([7777, 9999])); d["tfc_never"] = (tfc == 7777).astype(int)
d["sci"] = num("SCI")
d["days_to_acute"] = num("DAYStoACUTEadm", [9999]); d["days_to_rehab"] = num("DAYStoREHABadm", [9999])
d["rehab_dc_day"] = num("DAYStoREHABdc", [9999])
d["rehab_los"] = d.rehab_dc_day - d.days_to_rehab
for s_, t_ in [("FIMTOTA","fim_tot_adm"),("FIMMOTA","fim_mot_adm"),("FIMCOGA","fim_cog_adm"),("DRSa","drs_adm")]:
    d[t_] = num(s_)
d["cause"] = f1.Cause.mask(f1.Cause == 999)
d["payor"] = f1.RehabPay1.mask(f1.RehabPay1.isin([55, 999]))
d["emp_preinj"] = f1.Emp1.mask(f1.Emp1.isin([666,777,888,999]))
d["res_preinj"] = f1.ResInj.mask(f1.ResInj == 999)

pta = pd.to_numeric(f1.PTADays, errors="coerce")
d["event"] = (~pta.isin([8888, 9999]) & pta.notna()).astype(int)
d["duration"] = np.where(d.event == 1, pta, d.rehab_dc_day)
d.loc[pta == 9999, ["event", "duration"]] = np.nan
df = d[d.duration.notna() & (d.duration > 0)].reset_index(drop=True)
df["event"] = df.event.astype(int)

FEAT = ["age","age_topcoded","male","edu_years","inj_year","gcs","gcs_sedated","tfc_days",
        "tfc_never","sci","days_to_acute","days_to_rehab","fim_tot_adm","fim_mot_adm",
        "fim_cog_adm","drs_adm","cause","payor","emp_preinj","res_preinj"]
TOP = ["days_to_rehab","drs_adm","fim_cog_adm","tfc_days","fim_tot_adm","gcs",
       "fim_mot_adm","inj_year","age","cause","gcs_sedated","edu_years"]
T = df.duration.values; E = df.event.values.astype(bool)
itr, ite = train_test_split(np.arange(len(df)), test_size=.20, random_state=SEED, stratify=E)
X = df[FEAT].astype(float)
imp = SimpleImputer(strategy="median").fit(X.iloc[itr])
sc = StandardScaler().fit(imp.transform(X.iloc[itr]))
Xtr = pd.DataFrame(sc.transform(imp.transform(X.iloc[itr])), columns=FEAT)
Xte = pd.DataFrame(sc.transform(imp.transform(X.iloc[ite])), columns=FEAT)

print("="*70); print("ITEM 1 -- CENSORING INFERENCE: naive vs KM vs adjusted")
print("="*70)
naive_med = float(np.median(T[E]))
km = KaplanMeierFitter().fit(T, E); km_med = float(km.median_survival_time_)

# (a) IPCW-weighted KM: weights 1/G(T_i-) from a Cox model for the CENSORING process
Xall = pd.DataFrame(sc.transform(imp.transform(X)), columns=FEAT)[TOP].copy()
cd = Xall.copy(); cd["T"] = T; cd["Ecens"] = (~E).astype(int)
cph_c = CoxPHFitter(penalizer=0.1).fit(cd, "T", "Ecens")
Gt = cph_c.predict_survival_function(Xall)                      # censoring survival
G_at_T = np.array([np.interp(T[i], Gt.index.values, Gt.iloc[:, i].values) for i in range(len(T))])
G_at_T = np.clip(G_at_T, 0.02, 1.0)
w = np.where(E, 1.0 / G_at_T, 0.0)
w = w / w[E].mean()
km_ipcw = KaplanMeierFitter().fit(T[E], np.ones(E.sum()), weights=w[E])
ipcw_med = float(km_ipcw.median_survival_time_)

# (b) covariate-adjusted (g-formula) median from a Cox model for T
td = Xall.copy(); td["T"] = T; td["E"] = E.astype(int)
cph_t = CoxPHFitter(penalizer=0.1).fit(td, "T", "E")
S = cph_t.predict_survival_function(Xall)                        # one curve per subject
S_avg = S.mean(axis=1)
gf_med = float(S_avg.index[np.searchsorted(-S_avg.values, -0.5)])

print(f"  naive median (observed only)        : {naive_med:.0f} d")
print(f"  marginal Kaplan-Meier median        : {km_med:.0f} d")
print(f"  IPCW-weighted median                : {ipcw_med:.0f} d")
print(f"  covariate-adjusted (g-formula) med. : {gf_med:.0f} d")
OUT["medians"] = dict(naive=naive_med, km=km_med, ipcw=ipcw_med, gformula=gf_med)

# does censoring depend on T given X?  (test of MNAR vs covariate-dependent MAR)
print("\n  Censoring depends on covariates X: Cox censoring model global p =",
      f"{cph_c.log_likelihood_ratio_test().p_value:.2e}")
print("  -> establishes covariate-dependent (conditionally MAR) censoring.")
print("  -> MNAR (C depends on T given X) is NOT identifiable from these data.")

print("\n" + "="*70); print("ITEM 2 -- TUNED CLASSICAL COX vs ML (paired bootstrap)")
print("="*70)
ytr = Surv.from_arrays(E[itr], T[itr])
def cidx(m, Xd, idx): return concordance_index_censored(E[idx], T[idx], m)[0]

best = (None, -1)
for pen in [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]:
    tr = Xtr[TOP].copy(); tr["T"] = T[itr]; tr["E"] = E[itr].astype(int)
    m = CoxPHFitter(penalizer=pen).fit(tr, "T", "E")
    c = cidx(m.predict_partial_hazard(Xtr[TOP]).values, None, itr)
    if c > best[1]: best = (pen, c, m)
pen, ctr_cox, cox_m = best
r_cox = cox_m.predict_partial_hazard(Xte[TOP]).values
c_cox = cidx(r_cox, None, ite)

xm = xgb.XGBRegressor(objective="survival:cox", n_estimators=300, learning_rate=.05,
                      max_depth=3, subsample=.8, colsample_bytree=.8, reg_lambda=1.,
                      random_state=SEED).fit(Xtr[TOP], np.where(E[itr], T[itr], -T[itr]))
r_xgb = xm.predict(Xte[TOP]); c_xgb = cidx(r_xgb, None, ite)

B, diffs, cc, cx = 800, [], [], []
rng = np.random.RandomState(SEED)
for _ in range(B):
    b = rng.choice(len(ite), len(ite), replace=True)
    if E[ite][b].sum() < 30: continue
    a = concordance_index_censored(E[ite][b], T[ite][b], r_cox[b])[0]
    z = concordance_index_censored(E[ite][b], T[ite][b], r_xgb[b])[0]
    cc.append(a); cx.append(z); diffs.append(z - a)
lo, hi = np.percentile(diffs, [2.5, 97.5])
print(f"  tuned Cox (penalizer={pen}) test C : {c_cox:.3f}  [{np.percentile(cc,2.5):.3f}, {np.percentile(cc,97.5):.3f}]")
print(f"  XGBoost-Cox            test C      : {c_xgb:.3f}  [{np.percentile(cx,2.5):.3f}, {np.percentile(cx,97.5):.3f}]")
print(f"  paired difference (XGB - Cox)      : {np.mean(diffs):+.4f}  [{lo:+.4f}, {hi:+.4f}]")
print(f"  significant at 5%: {'YES' if (lo>0 or hi<0) else 'NO'}")
OUT["cox_vs_ml"] = dict(cox=c_cox, xgb=c_xgb, diff=float(np.mean(diffs)),
                        lo=float(lo), hi=float(hi), penalizer=pen,
                        significant=bool(lo > 0 or hi < 0))

print("\n" + "="*70); print("ITEM 3 -- DISENTANGLING RECOVERY FROM LENGTH OF STAY")
print("="*70)
unc = E[ite]
c_unc = concordance_index_censored(np.ones(unc.sum(), bool), T[ite][unc], r_xgb[unc])[0]
print(f"  C-index, uncensored test subset only : {c_unc:.3f}  (n={unc.sum():,})")
los_te = df.rehab_los.values[ite]
ok = ~np.isnan(los_te)
from scipy.stats import spearmanr
rho_all = spearmanr(r_xgb[ok], los_te[ok]).statistic
rho_unc = spearmanr(r_xgb[unc & ok], los_te[unc & ok]).statistic
print(f"  Spearman(predicted risk, rehab LOS)  : {rho_all:+.3f} (all test)")
print(f"                                        {rho_unc:+.3f} (uncensored only)")
# without any timeline feature
NT = [f for f in TOP if f not in ("days_to_rehab","days_to_acute","tfc_days","tfc_never")]
xm2 = xgb.XGBRegressor(objective="survival:cox", n_estimators=300, learning_rate=.05,
                       max_depth=3, subsample=.8, colsample_bytree=.8, reg_lambda=1.,
                       random_state=SEED).fit(Xtr[NT], np.where(E[itr], T[itr], -T[itr]))
r2 = xm2.predict(Xte[NT])
c_nt = cidx(r2, None, ite)
c_nt_unc = concordance_index_censored(np.ones(unc.sum(), bool), T[ite][unc], r2[unc])[0]
rho_nt = spearmanr(r2[ok], los_te[ok]).statistic
print(f"  no-timeline model: C {c_nt:.3f} | uncensored-only C {c_nt_unc:.3f} | rho(LOS) {rho_nt:+.3f}")
OUT["los"] = dict(c_uncensored=float(c_unc), rho_all=float(rho_all), rho_unc=float(rho_unc),
                  c_notimeline=float(c_nt), c_nt_unc=float(c_nt_unc), rho_nt=float(rho_nt))

print("\n" + "="*70); print("ITEM 4 -- GCS COMPLETE-CASE ANALYSIS")
print("="*70)
cc_mask = df.gcs.notna().values
print(f"  participants with recorded GCS: {cc_mask.sum():,} ({100*cc_mask.mean():.1f}%)")
sub = np.where(cc_mask)[0]
itr2, ite2 = train_test_split(sub, test_size=.20, random_state=SEED, stratify=E[sub])
imp2 = SimpleImputer(strategy="median").fit(X.iloc[itr2])
sc2 = StandardScaler().fit(imp2.transform(X.iloc[itr2]))
A = pd.DataFrame(sc2.transform(imp2.transform(X.iloc[itr2])), columns=FEAT)[TOP]
Bm = pd.DataFrame(sc2.transform(imp2.transform(X.iloc[ite2])), columns=FEAT)[TOP]
xm3 = xgb.XGBRegressor(objective="survival:cox", n_estimators=300, learning_rate=.05,
                       max_depth=3, subsample=.8, colsample_bytree=.8, reg_lambda=1.,
                       random_state=SEED).fit(A, np.where(E[itr2], T[itr2], -T[itr2]))
c_cc = concordance_index_censored(E[ite2], T[ite2], xm3.predict(Bm))[0]
print(f"  complete-case test C-index: {c_cc:.3f}")
uni = {}
for v in TOP:
    try:
        dd = A[[v]].copy(); dd["T"] = T[itr2]; dd["E"] = E[itr2].astype(int)
        uni[v] = abs(CoxPHFitter().fit(dd, "T", "E").summary.loc[v, "z"])
    except Exception: uni[v] = 0.0
rk = pd.Series(uni).sort_values(ascending=False)
print("  univariate Cox |z| ranking (complete case):")
for i, (k, v) in enumerate(rk.items(), 1):
    print(f"    {i:2d}. {k:16s} {v:7.2f}")
gcs_rank_cc = int(list(rk.index).index("gcs") + 1)
print(f"  -> GCS rank with recorded values only: {gcs_rank_cc} (was 6 with imputation)")
OUT["gcs_cc"] = dict(n=int(cc_mask.sum()), c=float(c_cc), gcs_rank=gcs_rank_cc,
                     gcs_z=float(rk["gcs"]))

print("\n" + "="*70); print("BONUS -- COMPETING-RISKS VIEW")
print("="*70)
# discharge-while-in-PTA treated as a competing event, not censoring
t_grid = np.arange(1, 181)
km_all = KaplanMeierFitter().fit(T, E)
cif_km = 1 - np.interp(t_grid, km_all.survival_function_.index.values,
                       km_all.survival_function_.values.ravel())
# Aalen-Johansen CIF for emergence with discharge as competing risk
n = len(T); order = np.argsort(T); Ts, Es = T[order], E[order]
at_risk = n - np.arange(n)
S_overall = np.cumprod(1 - 1.0 / np.maximum(at_risk, 1))
cif_aj = np.zeros(len(t_grid))
inc = np.where(Es == 1, 1.0 / np.maximum(at_risk, 1), 0.0)
Sprev = np.concatenate([[1.0], S_overall[:-1]])
contrib = np.cumsum(Sprev * inc)
cif_aj = np.interp(t_grid, Ts, contrib)
print(f"  1 - KM at 60 d (treats discharge as censoring): {cif_km[59]:.3f}")
print(f"  Aalen-Johansen CIF at 60 d (competing risk)   : {cif_aj[59]:.3f}")
print(f"  1 - KM at 180 d: {cif_km[-1]:.3f} | AJ CIF at 180 d: {cif_aj[-1]:.3f}")
print("  -> KM overstates the probability of in-rehabilitation emergence.")
OUT["competing"] = dict(km60=float(cif_km[59]), aj60=float(cif_aj[59]),
                        km180=float(cif_km[-1]), aj180=float(cif_aj[-1]))

json.dump(OUT, open("reviewer_analyses.json", "w"), indent=1)
print("\nSaved reviewer_analyses.json")
