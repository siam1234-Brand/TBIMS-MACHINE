"""
simulation_los_decomposition.py -- does the length-of-stay (LOS) contrast recover a
known emergence-specific signal?  (Reviewer 2 request.)

Generative model (semi-competing risks, times in days):
  X1..X6 ~ N(0,1).  X1 is shared, X2 is emergence-specific, X3 is discharge-specific,
  X4..X6 are noise.
  Emergence   T  ~ Exp(rate = 0.05 exp(0.8 X1 + b X2))
  Discharge-before-emergence  D0 ~ Exp(rate = 0.02 exp(0.8 X1 + g3 X3))
  Observed time = min(T, D0), event = 1{T < D0}  (discharge in PTA censors emergence)
  LOS = D0                 (uncoupled: discharge does not depend on emergence), or
  LOS = min(D0, T + R),    R ~ Exp(mean 10)   (coupled: discharge follows emergence)
Ground truth: the emergence-specific gain  G* = C(0.8 X1 + b X2) - C(0.8 X1), computed on
the same test outcomes with the true coefficients. G* = 0 whenever b = 0.
Estimate: Delta-C = C(XGBoost-Cox PTA score) - C(-XGBoost LOS prediction), exactly as in
the paper, on an 80/20 split.  Within-decile concordance of both scores is also reported.
"""
import os, json, time, numpy as np, pandas as pd
import xgboost as xgb
from sksurv.metrics import concordance_index_censored
SEED = 42
def harrell(e, t, r): return concordance_index_censored(np.asarray(e, bool), np.asarray(t, float), np.asarray(r, float))[0]
def fit_xgb(X, t, e):
    return xgb.XGBRegressor(objective="survival:cox", n_estimators=300, learning_rate=.05, max_depth=3, subsample=.8,
                            colsample_bytree=.8, reg_lambda=1., random_state=SEED, n_jobs=1).fit(X, np.where(e, t, -t))

N, REPS = 6000, 20
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "simulation.json")
MECH = {"clean": (0.0, False), "discharge_specific": (0.8, False),
        "coupled": (0.0, True), "both": (0.8, True)}


def one(b, g3, coupled, rng):
    X = rng.normal(size=(N, 6)); x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
    T = rng.exponential(1 / (0.05 * np.exp(0.8 * x1 + b * x2)))
    D0 = rng.exponential(1 / (0.02 * np.exp(0.8 * x1 + g3 * x3)))
    time_, ev = np.minimum(T, D0), T < D0
    los = np.minimum(D0, T + rng.exponential(10, N)) if coupled else D0
    idx = rng.permutation(N); a, t = idx[:int(.8 * N)], idx[int(.8 * N):]
    Xd = pd.DataFrame(X, columns=[f"x{k}" for k in range(1, 7)])
    r_pta = fit_xgb(Xd.iloc[a], time_[a], ev[a]).predict(Xd.iloc[t])
    lm = xgb.XGBRegressor(n_estimators=300, learning_rate=.05, max_depth=3, subsample=.8,
                          colsample_bytree=.8, reg_lambda=1., random_state=SEED, n_jobs=1)
    pl = lm.fit(Xd.iloc[a], np.log(np.clip(los[a], 0.5, None))).predict(Xd.iloc[t])
    E, Tt = ev[t], time_[t]
    c_true_full = harrell(E, Tt, 0.8 * x1[t] + b * x2[t]); c_true_shared = harrell(E, Tt, x1[t])
    c_pta, c_los = harrell(E, Tt, r_pta), harrell(E, Tt, -pl)
    q = pd.qcut(pl, 10, labels=False); wp, wl = [], []
    for k in range(10):
        m = q == k; wp.append(harrell(E[m], Tt[m], r_pta[m])); wl.append(harrell(E[m], Tt[m], -pl[m]))
    return {"cens": float(1 - ev.mean()), "G_true": c_true_full - c_true_shared,
            "C_true": c_true_full, "c_pta": c_pta, "c_los": c_los, "dC": c_pta - c_los,
            "share": (c_los - .5) / (c_pta - .5), "q10_pta": float(np.mean(wp)), "q10_los": float(np.mean(wl))}


res = {}
for mech, (g3, coupled) in MECH.items():
    for b in (0.0, 0.4, 0.8):
        t0 = time.time(); rng = np.random.default_rng(12345)
        rows = [one(b, g3, coupled, rng) for _ in range(REPS)]
        s = {k: {"mean": float(np.mean([r[k] for r in rows])), "sd": float(np.std([r[k] for r in rows], ddof=1))}
             for k in rows[0]}
        res[f"{mech}|{b}"] = s
        print(mech, b, {k: round(v["mean"], 3) for k, v in s.items()}, round(time.time() - t0, 1), "s", flush=True)
json.dump({"N": N, "REPS": REPS, "results": res}, open(OUT, "w"), indent=1)
print("saved", OUT)
