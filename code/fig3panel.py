import numpy as np,pandas as pd,warnings;warnings.filterwarnings("ignore")
import matplotlib;matplotlib.use("Agg");import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer; from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_curve,roc_auc_score
from lifelines import KaplanMeierFitter
import xgboost as xgb
SEED=42;np.random.seed(SEED)
plt.rcParams.update({"font.size":7,"axes.labelsize":7,"axes.titlesize":7.5,
 "xtick.labelsize":6.5,"ytick.labelsize":6.5,"legend.fontsize":6,
 "font.family":"serif","axes.spines.top":False,"axes.spines.right":False})
d=pd.read_csv('results/analysis_sample.csv') if False else None
import subprocess
exec(open('reviewer_fixes.py').read().split("print(\"=\"*70)")[0])
TOPf=["days_to_rehab","drs_adm","fim_cog_adm","tfc_days","fim_tot_adm","gcs",
      "fim_mot_adm","inj_year","age","cause","gcs_sedated","edu_years"]
xm=xgb.XGBRegressor(objective="survival:cox",n_estimators=300,learning_rate=.05,max_depth=3,
    subsample=.8,colsample_bytree=.8,reg_lambda=1.,random_state=SEED).fit(Xtr[TOPf],np.where(E[itr],T[itr],-T[itr]))
risk=xm.predict(Xte[TOPf])
cc=HistGradientBoostingClassifier(random_state=SEED).fit(Xtr,1-E[itr]); pc=cc.predict_proba(Xte)[:,1]
auc=roc_auc_score(1-E[ite],pc)
fig,ax=plt.subplots(1,3,figsize=(7.16,2.05))
km=KaplanMeierFitter().fit(T,E)
km.plot_survival_function(ax=ax[0],color="#1f3b57",legend=False,ci_alpha=.18)
n=len(T);o=np.argsort(T);Ts,Es=T[o],E[o];ar=n-np.arange(n)
S=np.cumprod(1-1./np.maximum(ar,1));Sp=np.concatenate([[1.],S[:-1]])
cif=np.cumsum(Sp*np.where(Es==1,1./np.maximum(ar,1),0.))
ax[0].plot(Ts,1-cif,color="#c0392b",lw=1.1,ls="--")
ax[0].legend(["Kaplan--Meier","Aalen--Johansen"],frameon=False,loc="upper right")
ax[0].set(xlabel="Days since injury",ylabel="Not yet emerged",title="(a) Emergence from PTA",xlim=(0,180),ylim=(0,1))
q=pd.qcut(risk,4,labels=False)
for k,(lab,col) in enumerate(zip(["Q1 fastest","Q2","Q3","Q4 slowest"],["#2a9d8f","#8ab17d","#e0a458","#c0392b"])):
    m=q==(3-k)
    KaplanMeierFitter().fit(T[ite][m],E[ite][m],label=lab).plot_survival_function(ax=ax[1],color=col,ci_show=False,lw=1.)
ax[1].set(xlabel="Days since injury",ylabel="Still in PTA",title="(b) Risk stratification (test)",xlim=(0,180),ylim=(0,1))
ax[1].legend(frameon=False)
ev=E.astype(bool)
ax[2].scatter(df.loc[ev,"fim_cog_adm"],df.loc[ev,"drs_adm"],s=2,alpha=.09,c="#2a9d8f",label="Emerged",rasterized=True)
ax[2].scatter(df.loc[~ev,"fim_cog_adm"],df.loc[~ev,"drs_adm"],s=2,alpha=.22,c="#c0392b",label="Censored",rasterized=True)
ax[2].set(xlabel="FIM-Cognitive (adm)",ylabel="DRS (adm)",title=f"(c) Censoring not random (AUC {auc:.3f})")
ax[2].legend(markerscale=4,frameon=False,loc="upper right")
plt.tight_layout(pad=.4);plt.savefig("fig_3panel.pdf",bbox_inches="tight")
print("saved fig_3panel.pdf | censoring AUC %.3f"%auc)
