# =============================================================================
#  Cognitive Recovery SPEED Prediction after Traumatic Brain Injury
#  TBIMS National Database public release  --  Form1.csv ONLY
#  Outcome : time from injury to emergence from Post-Traumatic Amnesia (PTA)
#            = a TIME-TO-EVENT outcome with 18% right-censoring
# =============================================================================

# %% CELL 1 --- Install & imports ---------------------------------------------
# Colab:  !pip install lifelines scikit-survival shap xgboost -q
import os, warnings, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxnetSurvivalAnalysis
import xgboost as xgb
from sksurv.metrics import concordance_index_censored, integrated_brier_score
from sksurv.util import Surv
from lifelines import KaplanMeierFitter, CoxPHFitter
import shap

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED)
plt.rcParams.update({"figure.dpi": 120, "font.size": 9, "savefig.bbox": "tight"})
OUT = "results"; os.makedirs(OUT, exist_ok=True)


# %% CELL 2 --- Load Form1.csv (manually uploaded in Colab) -------------------
# from google.colab import files; files.upload()      # pick Form1.csv
CSV = os.environ.get("FORM1_CSV", "Form1.csv")
f1 = pd.read_csv(CSV, low_memory=False)
print(f"Form1 loaded: {f1.shape[0]:,} participants x {f1.shape[1]} variables")
assert f1.shape[1] == 325, "This does not look like Form1.csv (expected 325 columns)"


# %% CELL 3 --- PREPROCESSING 1: sentinel-code cleaning ------------------------
# TBIMS has NO global missing marker. Each variable has its own special codes
# (66=variable did not exist, 77=refused, 88=N/A, 99=unknown, ...).
# Treating them as real numbers is the single biggest error with this dataset.
GEN = [66, 77, 88, 99, 666, 777, 888, 999, 6666, 9999]

def num(col, sent=GEN):
    s = pd.to_numeric(f1[col], errors="coerce")
    return s.mask(s.isin(sent))

d = pd.DataFrame({"Mod1Id": f1["Mod1Id"]})

# demographics
age = pd.to_numeric(f1["AGENoPHI"], errors="coerce")
d["age"] = age.where(age.between(0, 88))
d["age_topcoded"] = (age == 777).astype(int)     # 777 = "89 or older" (HIPAA)
d.loc[age == 777, "age"] = 89
d["male"]      = f1["SexF"].map({1: 0, 2: 1})
d["edu_years"] = num("EduYears")
d["inj_year"]  = num("INJYEAR", [9999])

# injury severity
gcs = pd.to_numeric(f1["GCSTot"], errors="coerce")
d["gcs"]         = gcs.mask(gcs.isin([77, 88, 999]))
d["gcs_sedated"] = gcs.isin([77, 88]).astype(int)   # sedated/intubated = informative
tfc = pd.to_numeric(f1["TFCDays"], errors="coerce")
d["tfc_days"]  = tfc.mask(tfc.isin([7777, 9999]))
d["tfc_never"] = (tfc == 7777).astype(int)          # never followed commands
d["sci"] = num("SCI")

# care timeline
d["days_to_acute"] = num("DAYStoACUTEadm", [9999])
d["days_to_rehab"] = num("DAYStoREHABadm", [9999])
d["rehab_dc_day"]  = num("DAYStoREHABdc",  [9999])

# function at REHAB ADMISSION only (discharge values would leak the outcome)
for s, t in [("FIMTOTA","fim_tot_adm"), ("FIMMOTA","fim_mot_adm"),
             ("FIMCOGA","fim_cog_adm"), ("DRSa","drs_adm")]:
    d[t] = num(s)
# kept for Table 1 description ONLY -- never used as a predictor
for s, t in [("FIMTOTD","fim_tot_dc"), ("FIMCOGD","fim_cog_dc"), ("DRSd","drs_dc")]:
    d[t] = num(s)

# context (categorical)
d["cause"]      = f1["Cause"].mask(f1["Cause"] == 999)
d["payor"]      = f1["RehabPay1"].mask(f1["RehabPay1"].isin([55, 999]))
d["emp_preinj"] = f1["Emp1"].mask(f1["Emp1"].isin([666, 777, 888, 999]))
d["res_preinj"] = f1["ResInj"].mask(f1["ResInj"] == 999)
print("Predictors cleaned:", d.shape)


# %% CELL 4 --- PREPROCESSING 2: build the time-to-event outcome ---------------
# PTADays: days from injury to emergence from PTA
#   8888 = STILL IN PTA at rehab discharge -> RIGHT-CENSORED (not missing!)
#   9999 = unknown                          -> truly missing, dropped
pta = pd.to_numeric(f1["PTADays"], errors="coerce")
d["event"]    = (~pta.isin([8888, 9999]) & pta.notna()).astype(int)
d["duration"] = np.where(d["event"] == 1, pta, d["rehab_dc_day"])
d.loc[pta == 9999, ["event", "duration"]] = np.nan

df = d[d.duration.notna() & (d.duration > 0)].reset_index(drop=True)
print(f"Analytic sample : {len(df):,}")
df["event"] = df["event"].astype(int)
print(f"  emerged       : {df.event.sum():,} ({100*df.event.mean():.1f}%)")
print(f"  right-censored: {(df.event==0).sum():,} ({100*(1-df.event.mean()):.1f}%)")


# %% CELL 5 --- PREPROCESSING 3: feature set, no leakage -----------------------
FEATURES = ["age","age_topcoded","male","edu_years","inj_year","gcs","gcs_sedated",
            "tfc_days","tfc_never","sci","days_to_acute","days_to_rehab",
            "fim_tot_adm","fim_mot_adm","fim_cog_adm","drs_adm",
            "cause","payor","emp_preinj","res_preinj"]
X_raw = df[FEATURES].astype(float)
y_time, y_event = df.duration.values, df.event.values.astype(bool)
print("Missing rate per feature (%):")
print((100*X_raw.isna().mean()).round(1).sort_values(ascending=False).head(8).to_string())


# %% CELL 6 --- TRAIN / TEST SPLIT (stratified on the event indicator) --------
idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.20,
                                  random_state=SEED, stratify=y_event)
# Imputer + scaler fitted ONLY on train  -> prevents information leakage
imp = SimpleImputer(strategy="median").fit(X_raw.iloc[idx_tr])
sc  = StandardScaler().fit(imp.transform(X_raw.iloc[idx_tr]))
Xtr = pd.DataFrame(sc.transform(imp.transform(X_raw.iloc[idx_tr])), columns=FEATURES)
Xte = pd.DataFrame(sc.transform(imp.transform(X_raw.iloc[idx_te])), columns=FEATURES)
ytr = Surv.from_arrays(y_event[idx_tr], y_time[idx_tr])
yte = Surv.from_arrays(y_event[idx_te], y_time[idx_te])
print(f"Train {len(idx_tr):,} | Test {len(idx_te):,} | "
      f"censoring train {100*(1-y_event[idx_tr].mean()):.1f}% test {100*(1-y_event[idx_te].mean()):.1f}%")


# %% CELL 7 --- TABLE 1 : who gets censored?  (justifies survival analysis) ----
T1 = ["age","male","edu_years","gcs","tfc_days","fim_tot_adm","fim_cog_adm",
      "drs_adm","fim_cog_dc","drs_dc","days_to_rehab"]
g1, g0 = df[df.event==1], df[df.event==0]
tab1 = pd.DataFrame({
    "Emerged":  g1[T1].mean(),
    "Censored": g0[T1].mean(),
    "SMD": [(g0[v].mean()-g1[v].mean())/np.sqrt((g0[v].var()+g1[v].var())/2) for v in T1]
}).round(3)
tab1.to_csv(f"{OUT}/table1_censoring.csv")
print("\n=== TABLE 1  (|SMD| > 0.10 = meaningful imbalance) ===")
print(tab1.sort_values("SMD", key=abs, ascending=False).to_string())

# Fit on TRAIN, evaluate on the held-out TEST set only (no in-sample leakage).
cens_clf = HistGradientBoostingClassifier(random_state=SEED).fit(Xtr, 1 - y_event[idx_tr])
p_cens_te = cens_clf.predict_proba(Xte)[:, 1]
auc_c = roc_auc_score(1 - y_event[idx_te], p_cens_te)
print(f"\nAUC for predicting censoring (held-out test): {auc_c:.3f}")
print("-> censoring is strongly predictable from admission data: NOT random (MNAR)")


# %% CELL 8 --- FEATURE IMPORTANCE / SELECTION (4 independent methods) --------
# Method 1: univariate Cox (|z| statistic)
cx = pd.DataFrame(Xtr, columns=FEATURES); cx["T"]=y_time[idx_tr]; cx["E"]=y_event[idx_tr]
uni = {}
for v in FEATURES:
    try:
        c = CoxPHFitter().fit(cx[[v,"T","E"]], "T", "E")
        uni[v] = abs(c.summary.loc[v, "z"])
    except Exception:
        uni[v] = 0.0
m1 = pd.Series(uni).rename("cox_z")

# Method 2: mutual information vs log-time (observed events only)
obs = y_event[idx_tr]
m2 = pd.Series(mutual_info_regression(Xtr[obs], np.log(y_time[idx_tr][obs]),
                                      random_state=SEED), index=FEATURES).rename("mut_info")

# Method 3: Random Survival Forest permutation importance (on a subsample = fast)
sub = np.random.RandomState(SEED).choice(len(Xtr), min(3000, len(Xtr)), replace=False)
Xs, ys_ = Xtr.iloc[sub], Surv.from_arrays(y_event[idx_tr][sub], y_time[idx_tr][sub])
rsf = RandomSurvivalForest(n_estimators=150, min_samples_leaf=15, max_features="sqrt",
                           n_jobs=-1, random_state=SEED).fit(Xs, ys_)
base = rsf.score(Xs, ys_)
perm = {}
for v in FEATURES:
    Xp = Xs.copy(); Xp[v] = np.random.RandomState(SEED).permutation(Xp[v].values)
    perm[v] = base - rsf.score(Xp, ys_)
m3 = pd.Series(perm).rename("rsf_perm")

# Method 4: SHAP on XGBoost with a Cox objective (exact TreeSHAP, very fast)
#   XGBoost survival:cox encodes censoring as a NEGATIVE time label.
lab_tr = np.where(y_event[idx_tr], y_time[idx_tr], -y_time[idx_tr])
xgb_cox = xgb.XGBRegressor(objective="survival:cox", n_estimators=300,
                           learning_rate=0.05, max_depth=3, subsample=0.8,
                           colsample_bytree=0.8, reg_lambda=1.0,
                           random_state=SEED).fit(Xtr, lab_tr)
sv = shap.TreeExplainer(xgb_cox).shap_values(Xtr)
m4 = pd.Series(np.abs(sv).mean(0), index=FEATURES).rename("shap")

imp_all = pd.concat([m1, m2, m3, m4], axis=1)
rank = imp_all.rank(ascending=False)           # rank-average = consensus
imp_all["mean_rank"] = rank.mean(1)
imp_all = imp_all.sort_values("mean_rank")
imp_all.round(4).to_csv(f"{OUT}/feature_importance.csv")
print("\n=== CONSENSUS FEATURE IMPORTANCE (lower mean_rank = more important) ===")
print(imp_all.round(3).to_string())

TOP = imp_all.head(12).index.tolist()
print("\nSelected TOP-12 features:", TOP)


# %% CELL 9 --- MODEL TRAINING (+ overfitting control) ------------------------
# Overfitting control used here:
#   (a) 5-fold CV on the TRAIN set only, test set untouched until the end
#   (b) regularisation: Coxnet elastic-net, leaf-size limits, subsample<1
#   (c) shallow trees (max_depth=3) + low learning rate
#   (d) explicit train-vs-test C-index gap reported for every model
def cidx(y_ev, y_t, pred):
    return concordance_index_censored(y_ev, y_t, pred)[0]

class XGBCoxWrapper:
    """Thin sksurv-compatible wrapper so XGBoost joins the same CV loop."""
    def __init__(self, **kw): self.kw = kw
    def get_params(self, deep=True): return dict(self.kw)
    def fit(self, X, y):
        ev = np.array([r[0] for r in y]); tt = np.array([r[1] for r in y])
        self.m = xgb.XGBRegressor(objective="survival:cox", random_state=SEED,
                                  **self.kw).fit(X, np.where(ev, tt, -tt))
        return self
    def predict(self, X): return self.m.predict(X)

MODELS = {
 "Coxnet (elastic-net)": CoxnetSurvivalAnalysis(l1_ratio=0.5, alpha_min_ratio=0.01,
                                                fit_baseline_model=True),
 "Random Survival Forest": RandomSurvivalForest(n_estimators=150, min_samples_leaf=25,
                                                max_features="sqrt", n_jobs=-1,
                                                max_samples=0.6, random_state=SEED),
 "XGBoost (Cox)": XGBCoxWrapper(n_estimators=300, learning_rate=0.05, max_depth=3,
                                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0),
}
XtrT, XteT = Xtr[TOP], Xte[TOP]
rows = []
for name, mdl in MODELS.items():
    # ---- cross-validated estimate on TRAIN (honest, no test leakage) ----
    cvs = []
    for tr_i, va_i in KFold(5, shuffle=True, random_state=SEED).split(XtrT):
        m = mdl.__class__(**mdl.get_params()).fit(XtrT.iloc[tr_i],
                                                  Surv.from_arrays(y_event[idx_tr][tr_i],
                                                                   y_time[idx_tr][tr_i]))
        p = m.predict(XtrT.iloc[va_i])
        cvs.append(cidx(y_event[idx_tr][va_i], y_time[idx_tr][va_i], p))
    mdl.fit(XtrT, ytr)
    # Train C-index on a 5k subsample -- it is only an overfitting diagnostic and
    # the concordance index is O(n^2).
    sb = np.random.RandomState(SEED).choice(len(XtrT), min(5000, len(XtrT)), replace=False)
    c_tr = cidx(y_event[idx_tr][sb], y_time[idx_tr][sb], mdl.predict(XtrT.iloc[sb]))
    c_te = cidx(y_event[idx_te], y_time[idx_te], mdl.predict(XteT))
    rows.append({"Model": name, "CV C-index": np.mean(cvs), "CV SD": np.std(cvs),
                 "Train C": c_tr, "Test C": c_te, "Overfit gap": c_tr - c_te})
res = pd.DataFrame(rows).round(3)
res.to_csv(f"{OUT}/model_results.csv", index=False)
print("\n=== MODEL COMPARISON ===")
print(res.to_string(index=False))
best_name = res.sort_values("Test C", ascending=False).iloc[0]["Model"]
best = MODELS[best_name]
print(f"\nBest model: {best_name}")


# %% CELL 10 --- Integrated Brier Score (calibration on the TEST set) --------
# XGBoost-Cox returns risk scores only, so IBS is computed for every model that
# can produce a survival curve. Report these alongside the C-index.
lo, hi = np.percentile(y_time[y_event], [10, 90])
grid = np.arange(max(lo, 2), min(hi, 180), 5)
ibs_rows = []
for name, mdl in MODELS.items():
    if not hasattr(mdl, "predict_survival_function"):
        continue
    try:
        fns = mdl.predict_survival_function(XteT)
        P = np.row_stack([[fn(t) for t in grid] for fn in fns])
        ibs_rows.append({"Model": name,
                         "IBS": integrated_brier_score(ytr, yte, P, grid)})
    except Exception as e:
        print(f"  IBS failed for {name}: {e}")
ibs_df = pd.DataFrame(ibs_rows).round(4)
if len(ibs_df):
    ibs_df.to_csv(f"{OUT}/brier_scores.csv", index=False)
    print("\n=== Integrated Brier Score (0 = perfect, 0.25 = uninformative) ===")
    print(ibs_df.to_string(index=False))


# %% CELL 11 --- Naive-vs-survival bias (the paper's headline) ----------------
km = KaplanMeierFitter().fit(y_time, y_event)
naive_med, km_med = np.median(y_time[y_event]), km.median_survival_time_
print("\n=== BIAS FROM DROPPING CENSORED PATIENTS ===")
print(f"  Naive median PTA (observed only): {naive_med:.0f} days")
print(f"  Kaplan-Meier median (all)       : {km_med:.0f} days")
print(f"  Underestimate                   : {km_med-naive_med:.0f} days "
      f"({100*(km_med-naive_med)/km_med:.0f}%)")


# %% CELL 12 --- SENSITIVITY: drop timeline predictors (same pipeline) --------
# For censored patients duration == rehab discharge day, which is mechanically
# related to days_to_rehab. We refit the SAME models on the SAME selected
# feature set with every timeline predictor removed.
TIMELINE = ["days_to_acute", "days_to_rehab", "tfc_days", "tfc_never"]
TOP_NT = [f for f in TOP if f not in TIMELINE]
print(f"\nSensitivity feature set ({len(TOP_NT)} of {len(TOP)}):", TOP_NT)
sens_rows = []
for name, mdl in MODELS.items():
    m = mdl.__class__(**mdl.get_params()).fit(Xtr[TOP_NT], ytr)
    sens_rows.append({"Model": name,
                      "Test C (full)": res.set_index("Model").loc[name, "Test C"],
                      "Test C (no timeline)": cidx(y_event[idx_te], y_time[idx_te],
                                                   m.predict(Xte[TOP_NT]))})
sens = pd.DataFrame(sens_rows).round(3)
sens.to_csv(f"{OUT}/sensitivity.csv", index=False)
print("\n=== SENSITIVITY ANALYSIS ===")
print(sens.to_string(index=False))

cens_nt = HistGradientBoostingClassifier(random_state=SEED).fit(Xtr[TOP_NT], 1 - y_event[idx_tr])
auc_nt = roc_auc_score(1 - y_event[idx_te], cens_nt.predict_proba(Xte[TOP_NT])[:, 1])
print(f"Censoring AUC without timeline features: {auc_nt:.3f}")


# %% CELL 13 --- PUBLICATION FIGURES (vector PDF, IEEE column widths) ---------
plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})
LBL = {"days_to_rehab": "Days to rehab", "drs_adm": "DRS (adm)",
       "fim_cog_adm": "FIM-Cog (adm)", "tfc_days": "Time to commands",
       "fim_tot_adm": "FIM-Total (adm)", "gcs": "GCS", "fim_mot_adm": "FIM-Motor (adm)",
       "inj_year": "Injury year", "age": "Age", "cause": "Cause",
       "gcs_sedated": "GCS sedated", "edu_years": "Education"}

km = KaplanMeierFitter().fit(y_time, y_event)
risk_te = MODELS[best_name].predict(XteT)
p_cens_te = cens_clf.predict_proba(Xte)[:, 1]

fig, ax = plt.subplots(2, 3, figsize=(7.16, 4.0))
km.plot_survival_function(ax=ax[0,0], color="#1f3b57", legend=False, ci_alpha=.18)
ax[0,0].axhline(.5, ls=":", c="grey", lw=.7)
ax[0,0].axvline(km.median_survival_time_, ls=":", c="#c0392b", lw=.7)
ax[0,0].text(km.median_survival_time_+4, .55,
             f"KM median\n{km.median_survival_time_:.0f} d", fontsize=6, color="#c0392b")
ax[0,0].set(xlabel="Days since injury", ylabel="Proportion still in PTA",
            title="(a) Emergence from PTA", xlim=(0,180), ylim=(0,1))

t10 = imp_all.head(10).iloc[::-1]
ax[0,1].barh([LBL.get(i,i) for i in t10.index], 1/t10["mean_rank"], color="#2a9d8f", height=.7)
ax[0,1].set(xlabel="Consensus importance (1/mean rank)", title="(b) Feature importance")

xp = np.arange(len(res))
ax[0,2].bar(xp-.19, res["Train C"], .38, label="Train", color="#e0a458")
ax[0,2].bar(xp+.19, res["Test C"], .38, label="Test", color="#2a9d8f")
ax[0,2].axhline(.5, ls="--", c="grey", lw=.7); ax[0,2].set_xticks(xp)
ax[0,2].set_xticklabels(["Reg. Cox", "RSF", "XGBoost"])
ax[0,2].set(ylabel="Harrell's C", title="(c) Overfitting check", ylim=(.45,.95))
ax[0,2].legend(frameon=False, loc="upper left")

ev = y_event.astype(bool)
ax[1,0].scatter(df.loc[ev,"fim_cog_adm"], df.loc[ev,"drs_adm"], s=2.5, alpha=.10,
                c="#2a9d8f", label="Emerged", rasterized=True)
ax[1,0].scatter(df.loc[~ev,"fim_cog_adm"], df.loc[~ev,"drs_adm"], s=2.5, alpha=.25,
                c="#c0392b", label="Censored", rasterized=True)
ax[1,0].set(xlabel="FIM-Cognitive (admission)", ylabel="DRS (admission)",
            title="(d) Censoring is not random")
ax[1,0].legend(markerscale=4, frameon=False, loc="upper right")

q = pd.qcut(risk_te, 4, labels=False)
for k,(lab,col) in enumerate(zip(["Q1 fastest","Q2","Q3","Q4 slowest"],
                                 ["#2a9d8f","#8ab17d","#e0a458","#c0392b"])):
    m = q == (3-k)
    KaplanMeierFitter().fit(y_time[idx_te][m], y_event[idx_te][m], label=lab)\
        .plot_survival_function(ax=ax[1,1], color=col, ci_show=False, lw=1.1)
ax[1,1].set(xlabel="Days since injury", ylabel="Proportion still in PTA",
            title="(e) Risk stratification (test)", xlim=(0,180), ylim=(0,1))
ax[1,1].legend(frameon=False)

fpr, tpr, _ = roc_curve(1 - y_event[idx_te], p_cens_te)
ax[1,2].plot(fpr, tpr, c="#c0392b", lw=1.3, label=f"AUC = {auc_c:.3f}")
ax[1,2].plot([0,1],[0,1], "--", c="grey", lw=.7)
ax[1,2].set(xlabel="False positive rate", ylabel="True positive rate",
            title="(f) Predicting who is censored")
ax[1,2].legend(frameon=False, loc="lower right")
plt.tight_layout(pad=.5); plt.savefig(f"{OUT}/fig1.pdf", bbox_inches="tight"); plt.show()

plt.figure(figsize=(3.4, 3.1))
sv_top = shap.TreeExplainer(xgb.XGBRegressor(objective="survival:cox", n_estimators=300,
    learning_rate=.05, max_depth=3, subsample=.8, colsample_bytree=.8, reg_lambda=1.,
    random_state=SEED).fit(XtrT, np.where(y_event[idx_tr], y_time[idx_tr],
                                          -y_time[idx_tr]))).shap_values(XtrT)
shap.summary_plot(sv_top, XtrT, feature_names=[LBL.get(f,f) for f in TOP],
                  max_display=10, show=False, plot_size=None)
plt.xlabel("SHAP value (impact on log hazard)", fontsize=7)
plt.tight_layout(pad=.3); plt.savefig(f"{OUT}/fig2.pdf", bbox_inches="tight"); plt.show()


# %% CELL 14 --- EVERY NUMBER CLAIMED IN THE PAPER ---------------------------
print("\n" + "="*64)
print("PAPER VERIFICATION SHEET")
print("="*64)
print(f"Form 1 cohort                : {f1.shape[0]:,} x {f1.shape[1]}")
print(f"Unknown PTA dropped (9999)   : {int((pd.to_numeric(f1.PTADays,errors='coerce')==9999).sum())}")
print(f"Analytic sample n            : {len(df):,}")
print(f"  emerged                    : {int(df.event.sum()):,} ({100*df.event.mean():.1f}%)")
print(f"  right-censored             : {int((df.event==0).sum()):,} ({100*(1-df.event.mean()):.1f}%)")
print(f"Train / Test                 : {len(idx_tr):,} / {len(idx_te):,}")
print(f"Censoring AUC (held-out)     : {auc_c:.3f}")
print(f"Naive median PTA             : {np.median(y_time[y_event==1]):.0f} d")
print(f"Kaplan-Meier median PTA      : {km.median_survival_time_:.0f} d")
print(f"Underestimate                : {km.median_survival_time_-np.median(y_time[y_event==1]):.0f} d "
      f"({100*(km.median_survival_time_-np.median(y_time[y_event==1]))/km.median_survival_time_:.0f}%)")
print(f"Max |SMD| in Table I         : {tab1.SMD.abs().max():.2f}")
print("\nTable II top-3 features      :", list(imp_all.head(3).index))
print(f"  GCS rank                   : {list(imp_all.index).index('gcs')+1}")
print(f"  Age univariate Cox |z|     : {imp_all.loc['age','cox_z']:.2f}")
print("\nTable III:"); print(res.to_string(index=False))
if len(ibs_df): print("\nIBS:"); print(ibs_df.to_string(index=False))
print("\nSensitivity:"); print(sens.to_string(index=False))
print(f"Censoring AUC (no timeline)  : {auc_nt:.3f}")
print("\nMissing rates (%):")
print((100*X_raw.isna().mean()).round(1).sort_values(ascending=False).head(5).to_string())
print("="*64)


# %% CELL 15 --- ITEM 1: censoring inference (naive / KM / IPCW / g-formula) ---
# Censoring predictability from X shows covariate-dependent censoring, NOT MNAR.
# Marginal Kaplan-Meier is therefore not unbiased either; we bracket the median.
from lifelines import CoxPHFitter
from scipy.stats import spearmanr

Xall = pd.DataFrame(sc.transform(imp.transform(X_raw)), columns=FEATURES)[TOP].copy()
cd_ = Xall.copy(); cd_["T"] = y_time; cd_["Ecens"] = (1 - y_event).astype(int)
cph_c = CoxPHFitter(penalizer=0.1).fit(cd_, "T", "Ecens")
Gt = cph_c.predict_survival_function(Xall)
G_at_T = np.array([np.interp(y_time[i], Gt.index.values, Gt.iloc[:, i].values)
                   for i in range(len(y_time))])
w = np.where(y_event.astype(bool), 1.0 / np.clip(G_at_T, 0.02, 1.0), 0.0)
w = w / w[y_event.astype(bool)].mean()
ev = y_event.astype(bool)
ipcw_med = KaplanMeierFitter().fit(y_time[ev], np.ones(ev.sum()),
                                   weights=w[ev]).median_survival_time_
td_ = Xall.copy(); td_["T"] = y_time; td_["E"] = y_event.astype(int)
S_avg = CoxPHFitter(penalizer=0.1).fit(td_, "T", "E").predict_survival_function(Xall).mean(axis=1)
gf_med = S_avg.index[np.searchsorted(-S_avg.values, -0.5)]
km_all = KaplanMeierFitter().fit(y_time, y_event)
print("Median PTA duration")
print(f"  complete case (naive)   : {np.median(y_time[ev]):.0f} d")
print(f"  marginal Kaplan-Meier   : {km_all.median_survival_time_:.0f} d")
print(f"  IPCW-weighted           : {ipcw_med:.0f} d")
print(f"  covariate-adjusted      : {gf_med:.0f} d")
print(f"  censoring Cox model p   : {cph_c.log_likelihood_ratio_test().p_value:.1e}"
      "  -> covariate-dependent, NOT demonstrably MNAR")


# %% CELL 16 --- ITEM 2: tuned classical Cox vs XGBoost (paired bootstrap) -----
best = (None, -1, None)
for pen in [0.001, 0.01, 0.05, 0.1, 0.5, 1.0]:
    tr = XtrT.copy(); tr["T"] = y_time[idx_tr]; tr["E"] = y_event[idx_tr].astype(int)
    m = CoxPHFitter(penalizer=pen).fit(tr, "T", "E")
    c = cidx(y_event[idx_tr].astype(bool), y_time[idx_tr],
             m.predict_partial_hazard(XtrT).values)
    if c > best[1]: best = (pen, c, m)
pen, _, cox_m = best
r_cox = cox_m.predict_partial_hazard(XteT).values
r_xgb = MODELS["XGBoost (Cox)"].predict(XteT)
Ete, Tte = y_event[idx_te].astype(bool), y_time[idx_te]
c_cox = cidx(Ete, Tte, r_cox); c_xgb = cidx(Ete, Tte, r_xgb)
rng = np.random.RandomState(SEED); diffs = []
for _ in range(800):
    b = rng.choice(len(idx_te), len(idx_te), replace=True)
    if Ete[b].sum() < 30: continue
    diffs.append(cidx(Ete[b], Tte[b], r_xgb[b]) - cidx(Ete[b], Tte[b], r_cox[b]))
lo, hi = np.percentile(diffs, [2.5, 97.5])
print(f"tuned Cox (penalizer={pen}) test C = {c_cox:.3f}")
print(f"XGBoost-Cox               test C = {c_xgb:.3f}")
print(f"paired difference = {np.mean(diffs):+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]"
      f"  -> {'significant' if (lo>0 or hi<0) else 'not significant'}")


# %% CELL 17 --- ITEM 3: disentangling recovery speed from length of stay -----
unc = Ete
print(f"C-index, uncensored test subset only : "
      f"{cidx(np.ones(unc.sum(), bool), Tte[unc], r_xgb[unc]):.3f}  (n={unc.sum():,})")
los_te = df.rehab_los.values[idx_te]; ok = ~np.isnan(los_te)
print(f"Spearman(predicted risk, rehab LOS)  : "
      f"{spearmanr(r_xgb[ok], los_te[ok]).statistic:+.3f}")
NT = [f for f in TOP if f not in ("days_to_acute", "days_to_rehab", "tfc_days", "tfc_never")]
xnt = xgb.XGBRegressor(objective="survival:cox", n_estimators=300, learning_rate=.05,
        max_depth=3, subsample=.8, colsample_bytree=.8, reg_lambda=1.,
        random_state=SEED).fit(Xtr[NT], np.where(y_event[idx_tr], y_time[idx_tr], -y_time[idx_tr]))
r_nt = xnt.predict(Xte[NT])
print(f"no-timeline model: C {cidx(Ete, Tte, r_nt):.3f} | "
      f"uncensored-only {cidx(np.ones(unc.sum(), bool), Tte[unc], r_nt[unc]):.3f} | "
      f"rho(LOS) {spearmanr(r_nt[ok], los_te[ok]).statistic:+.3f}")
print("-> the outcome is bounded by the inpatient episode; disclose this.")


# %% CELL 18 --- ITEM 4: GCS complete-case analysis ---------------------------
from sklearn.model_selection import train_test_split as _tts
cc_mask = df.gcs.notna().values
sub = np.where(cc_mask)[0]
i2, j2 = _tts(sub, test_size=.20, random_state=SEED, stratify=y_event[sub])
im2 = SimpleImputer(strategy="median").fit(X_raw.iloc[i2])
sc2 = StandardScaler().fit(im2.transform(X_raw.iloc[i2]))
A = pd.DataFrame(sc2.transform(im2.transform(X_raw.iloc[i2])), columns=FEATURES)[TOP]
Bm = pd.DataFrame(sc2.transform(im2.transform(X_raw.iloc[j2])), columns=FEATURES)[TOP]
xcc = xgb.XGBRegressor(objective="survival:cox", n_estimators=300, learning_rate=.05,
        max_depth=3, subsample=.8, colsample_bytree=.8, reg_lambda=1.,
        random_state=SEED).fit(A, np.where(y_event[i2], y_time[i2], -y_time[i2]))
print(f"recorded GCS: {cc_mask.sum():,} ({100*cc_mask.mean():.1f}%)")
print(f"complete-case test C = "
      f"{cidx(y_event[j2].astype(bool), y_time[j2], xcc.predict(Bm)):.3f}")
uni = {}
for v in TOP:
    dd = A[[v]].copy(); dd["T"] = y_time[i2]; dd["E"] = y_event[i2].astype(int)
    try: uni[v] = abs(CoxPHFitter().fit(dd, "T", "E").summary.loc[v, "z"])
    except Exception: uni[v] = 0.0
rk = pd.Series(uni).sort_values(ascending=False)
print(f"GCS rank with recorded values only: {list(rk.index).index('gcs')+1} (was 6 imputed)")


# %% CELL 19 --- Competing-risks view -----------------------------------------
# Discharge while still in PTA is a competing event, not censoring.
n_ = len(y_time); o = np.argsort(y_time)
Ts, Es = y_time[o], y_event[o].astype(bool); ar = n_ - np.arange(n_)
Sp = np.concatenate([[1.], np.cumprod(1 - 1./np.maximum(ar, 1))[:-1]])
cif = np.cumsum(Sp * np.where(Es, 1./np.maximum(ar, 1), 0.))
km_180 = 1 - np.interp(180, km_all.survival_function_.index.values,
                       km_all.survival_function_.values.ravel())
print(f"1 - Kaplan-Meier at 180 d          : {km_180:.3f}")
print(f"Aalen-Johansen CIF at 180 d        : {np.interp(180, Ts, cif):.3f}")
print("-> KM overstates the chance of emerging before discharge.")


# %% CELL 20 --- ROUND-2: extended penalty grid, repeated splits, Fine-Gray ----
# Standalone: re-derives everything from Form1.csv. ~15 min.
exec(open('round2_fixes.py').read())


# %% CELL 21 --- ROUND-3: LOS decomposition with CIs, deciles, nested LRT -----
exec(open('round3_fixes.py').read())


# %% CELL 22 --- FIGURES for the paper ----------------------------------------
# fig_3panel.pdf (results), fig_diag.pdf (decomposition/overfit/ROC),
# fig2.pdf (SHAP). The methodology flowchart is TikZ inside paper.tex.
exec(open('fig3panel.py').read())
exec(open('fig_diag.py').read())
