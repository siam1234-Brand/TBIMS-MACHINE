import numpy as np,pandas as pd,warnings;warnings.filterwarnings("ignore")
import matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_curve,roc_auc_score
from scipy.stats import spearmanr
import xgboost as xgb
exec(open('reviewer_fixes.py').read().split('print("="*72)')[0])
plt.rcParams.update({"font.size":6.5,"axes.labelsize":6.5,"axes.titlesize":7,
 "xtick.labelsize":6,"ytick.labelsize":6,"legend.fontsize":5.8,
 "font.family":"serif","axes.spines.top":False,"axes.spines.right":False})
TOPf=["days_to_rehab","drs_adm","fim_cog_adm","tfc_days","fim_tot_adm","gcs",
      "fim_mot_adm","inj_year","age","cause","gcs_sedated","edu_years"]
r_pta=xgb.XGBRegressor(objective="survival:cox",n_estimators=300,learning_rate=.05,max_depth=3,
    subsample=.8,colsample_bytree=.8,reg_lambda=1.,random_state=SEED).fit(
    Xtr[TOPf],np.where(E[itr],T[itr],-T[itr])).predict(Xte[TOPf])
los=df.rehab_los.values; ok=~np.isnan(los[ite])
los_m=xgb.XGBRegressor(n_estimators=300,learning_rate=.05,max_depth=3,subsample=.8,
    colsample_bytree=.8,reg_lambda=1.,random_state=SEED)
oa=~np.isnan(los[itr]); los_m.fit(Xtr[TOPf][oa],np.log(np.clip(los[itr][oa],1,None)))
p_los=los_m.predict(Xte[TOPf])
cc=HistGradientBoostingClassifier(random_state=SEED).fit(Xtr,1-E[itr]); pc=cc.predict_proba(Xte)[:,1]
auc=roc_auc_score(1-E[ite],pc)
fig,ax=plt.subplots(1,3,figsize=(7.16,1.85))
# (a) discrimination decomposition
labs=["LOS model\nalone","XGBoost\n(PTA)","Conditioned\non LOS"]
vals=[0.793,0.857,0.751]; err=[[0.004,0.003,0.005],[0.004,0.003,0.008]]
b=ax[0].bar(labs,vals,yerr=err,capsize=2.5,color=["#c0392b","#2a9d8f","#e0a458"],width=.6)
ax[0].axhline(.5,ls="--",c="grey",lw=.7)
ax[0].annotate("",xy=(1,0.857),xytext=(1,0.793),arrowprops=dict(arrowstyle="<->",lw=.8,color="black"))
ax[0].text(1.12,0.822,"$\\Delta$C\n0.064",fontsize=5.6)
ax[0].set(ylabel="Harrell's C",title="(a) Where the discrimination comes from",ylim=(.45,.95))
# (b) train vs test
xp=np.arange(3)
ax[1].bar(xp-.19,[0.843,0.827,0.862],.38,label="Train",color="#e0a458")
ax[1].bar(xp+.19,[0.845,0.818,0.857],.38,label="Test",color="#2a9d8f")
ax[1].axhline(.5,ls="--",c="grey",lw=.7); ax[1].set_xticks(xp)
ax[1].set_xticklabels(["Reg. Cox","RSF","XGBoost"])
ax[1].set(ylabel="Harrell's C",title="(b) Overfitting check",ylim=(.45,.95))
ax[1].legend(frameon=False,loc="upper left",handlelength=1.2)
# (c) ROC for censoring
fpr,tpr,_=roc_curve(1-E[ite],pc)
ax[2].plot(fpr,tpr,c="#c0392b",lw=1.2,label=f"AUC = {auc:.3f}")
ax[2].plot([0,1],[0,1],"--",c="grey",lw=.7)
ax[2].set(xlabel="False positive rate",ylabel="True positive rate",
          title="(c) Predicting who is censored")
ax[2].legend(frameon=False,loc="lower right",handlelength=1.2)
plt.tight_layout(pad=.35);plt.savefig("fig_diag.pdf",bbox_inches="tight")
print("saved | censoring AUC %.3f | rho(PTA,LOS) %.3f"%(auc,spearmanr(r_pta[ok],p_los[ok]).statistic))
