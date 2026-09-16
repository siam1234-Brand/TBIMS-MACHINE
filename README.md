# Predicting Emergence from Post-Traumatic Amnesia Within Inpatient Rehabilitation

Code and aggregate results for the paper *"Predicting Emergence from
Post-Traumatic Amnesia Within Inpatient Rehabilitation: How Much Is Recovery and
How Much Is Discharge Timing?"*

**Authors:** Mohammad Ali Masud, Md. Mahabubur Rahaman Siam, Sanjida Afrin
Bristi, Abdullah Al Shaihid Mahem, Sharfuddin Mahmood
Department of Computer Science and Engineering, American International
University-Bangladesh (AIUB), Dhaka, Bangladesh

---

## ⚠️ Data is not included, and must not be added

This repository contains **no participant-level data**. The TBI Model Systems
(TBIMS) National Database is governed by a Data Use Agreement that prohibits
redistribution. `Form1.csv`, any derived participant-level file, and
`analysis_sample.csv` must never be committed here. A `.gitignore` is included
as a safety net — please do not remove those rules.

**Repository:** <https://github.com/siam1234-Brand/TBIMS-MACHINE>

To reproduce the results, obtain the public use dataset yourself:

1. Go to <https://www.tbindsc.org/Researchers.aspx>
2. Complete the Data Request and Use Agreement Form
3. Place the resulting `Form1.csv` in the working directory
4. Run the notebook

---

## What the paper asks

The outcome — time to emergence from post-traumatic amnesia — is observed only
while the patient remains an inpatient. So how much of a model's predictive
performance is about *recovery*, and how much is about *when the episode ends*?

**Headline result.** A model predicting rehabilitation length of stay from the
identical feature set already reaches C = 0.793 for PTA emergence; the survival
model reaches 0.857. The increment attributable to modelling emergence rather
than discharge timing is **0.064** (95% range 0.061–0.068 over 20 splits).
Roughly three quarters of concordance over chance is shared with discharge
timing.

---

## Repository layout

```
code/
  TBIMS_complete.ipynb       Full pipeline, 23 cells: preprocessing → figures
  tbims_paper_complete.py    Same pipeline as a plain script
  reviewer_fixes.py          IPCW & g-formula medians, Cox benchmark,
                             LOS check, GCS complete case, competing risks
  round2_fixes.py            Penalty grid, 20 repeated splits, Fine–Gray
  round3_fixes.py            LOS decomposition with CIs, deciles, nested LRT
  fig3panel.py, fig_diag.py  Publication figures
results/                     Aggregate results only (safe to share)
figures/                     Figures used in the paper
```

## Running it

```bash
pip install lifelines scikit-survival shap xgboost
# place Form1.csv in the working directory, then:
jupyter notebook code/TBIMS_complete.ipynb     # Runtime ≈ 60–75 min
```

Or in Google Colab: upload the notebook, upload `Form1.csv`, and also upload
`round2_fixes.py`, `round3_fixes.py`, `fig3panel.py`, `fig_diag.py` (needed by
cells 20–22). Then Runtime → Run all.

The slowest steps are cell 9 (random survival forest) and cells 16 and 20
(bootstrap and 20 repeated splits).

## Key numbers reproduced by this code

| Quantity | Value |
|---|---|
| Analytic sample | 17,895 (14,231 emerged / 3,664 censored = 20.5%) |
| LOS model alone, as a risk score for PTA emergence | C = 0.793 [0.789, 0.797] |
| PTA survival model | C = 0.857 [0.854, 0.860] |
| **Increment (emergence-specific)** | **0.064 [0.061, 0.068]**, p < 10⁻²⁹ |
| Conditioned on predicted LOS (quintiles / deciles) | 0.762 / 0.751 |
| Censoring predictable from admission data | AUC = 0.866 |
| Median PTA: naive / KM / IPCW / covariate-adjusted | 19 / 26 / **23** / **24** days |
| XGBoost-Cox vs tuned classical Cox (20 splits) | 0.857 vs 0.845, +0.0121, p < 10⁻²⁰ |
| GCS complete case (n = 10,109) | C = 0.846, GCS still 6th |
| Fine–Gray vs cause-specific | 0.818 vs 0.824 (no improvement) |

---

## Data citation

Traumatic Brain Injury Model Systems Program. *Traumatic Brain Injury Model
Systems National Database.* Distributed by the Traumatic Brain Injury Model
Systems National Data and Statistical Center. Public use dataset, release
2026-02-17. DOI [10.17605/OSF.IO/A4XZB](https://doi.org/10.17605/OSF.IO/A4XZB).
Available: <http://www.tbindsc.org>

## Acknowledgment

The TBI Model Systems National Database is a multicenter study of the TBI Model
Systems Centers Program, and is supported by the National Institute on
Disability, Independent Living, and Rehabilitation Research (NIDILRR), a center
within the Administration for Community Living (ACL), U.S. Department of Health
and Human Services (HHS). However, these contents do not necessarily reflect the
opinions or views of the TBI Model Systems Centers, NIDILRR, ACL or HHS.

## Licence

Code is released under the MIT Licence (see `LICENSE`). The licence covers the
code in this repository only; it does not extend to the TBIMS data, which
remains governed by its own Data Use Agreement.
