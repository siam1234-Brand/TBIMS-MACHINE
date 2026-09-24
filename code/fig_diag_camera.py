"""fig_diag_camera.py -- Fig. 2 for the camera-ready version, same style as fig_diag.py.
Panel (a) adds the stay score's own within-decile concordance as a control; all values are read
from camera_ready_checks.json / results/model_results.csv (no hard-coded numbers)."""
import json, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_curve, roc_auc_score
src = open("camera_ready_checks.py").read()
exec(src.split("# ------------------------------------------------ 5. descriptive facts")[0])   # same data/cleaning
itr,ite=train_test_split(np.arange(len(df)),test_size=.2,random_state=SEED,stratify=E); Xtr,Xte=prep(X,itr,ite)
R = json.load(open("camera_ready_checks.json"))["repeated"]
M = pd.read_csv("../results/model_results.csv").set_index("Model")
plt.rcParams.update({"font.size":6.5,"axes.labelsize":6.5,"axes.titlesize":7,"xtick.labelsize":5.6,"ytick.labelsize":6,
                     "legend.fontsize":5.8,"font.family":"serif","axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42})
fig, ax = plt.subplots(1, 3, figsize=(7.16, 1.85))
keys = [("los","LOS model\nalone"),("xgb","XGBoost\n(PTA)"),("q10_pta","PTA model\nwithin deciles"),("q10_los","LOS model\nwithin deciles")]
vals = [R[k]["mean"] for k,_ in keys]; err = [[R[k]["mean"]-R[k]["lo"] for k,_ in keys],[R[k]["hi"]-R[k]["mean"] for k,_ in keys]]
ax[0].bar([l for _,l in keys], vals, yerr=err, capsize=2.2, color=["#c0392b","#2a9d8f","#e0a458","#c0392b"], width=.6,
          edgecolor=["none","none","none","#7b241c"], hatch=["","","","///"])
ax[0].axhline(.5, ls="--", c="grey", lw=.7)
ax[0].annotate("", xy=(1, R["xgb"]["mean"]), xytext=(1, R["los"]["mean"]), arrowprops=dict(arrowstyle="<->", lw=.8, color="black"))
ax[0].text(1.12, (R["xgb"]["mean"]+R["los"]["mean"])/2, "$\\Delta$C\n%.3f" % R["inc"]["mean"], fontsize=5.6, va="center")
ax[0].set(ylabel="Harrell's C", title="(a) Where the discrimination comes from", ylim=(.45,.95))
xp = np.arange(3); rows = ["Coxnet (elastic-net)","Random Survival Forest","XGBoost (Cox)"]
ax[1].bar(xp-.19, [M.loc[r,"Train C"] for r in rows], .38, label="Train", color="#e0a458")
ax[1].bar(xp+.19, [M.loc[r,"Test C"] for r in rows], .38, label="Test", color="#2a9d8f")
ax[1].axhline(.5, ls="--", c="grey", lw=.7); ax[1].set_xticks(xp); ax[1].set_xticklabels(["Reg. Cox","RSF","XGBoost"])
ax[1].set(ylabel="Harrell's C", title="(b) Overfitting check", ylim=(.45,.95)); ax[1].legend(frameon=False, loc="upper left", handlelength=1.2)
cc = HistGradientBoostingClassifier(random_state=SEED).fit(Xtr, 1-E[itr]); pc = cc.predict_proba(Xte)[:,1]
auc = roc_auc_score(1-E[ite], pc); fpr, tpr, _ = roc_curve(1-E[ite], pc)
ax[2].plot(fpr, tpr, c="#c0392b", lw=1.2, label=f"AUC = {auc:.3f}"); ax[2].plot([0,1],[0,1],"--",c="grey",lw=.7)
ax[2].set(xlabel="False positive rate", ylabel="True positive rate", title="(c) Predicting who is censored")
ax[2].legend(frameon=False, loc="lower right", handlelength=1.2)
plt.tight_layout(pad=.35); plt.savefig("fig_diag.pdf", bbox_inches="tight"); print("saved fig_diag.pdf | AUC %.3f" % auc)
