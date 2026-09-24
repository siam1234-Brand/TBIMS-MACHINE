# Camera-ready changes (the submitted version is commit [`b9bdaa6`](https://github.com/siam1234-Brand/TBIMS-MACHINE/tree/b9bdaa6b06e2ff74644c5a4ce701c14ecefba588), 16 September 2026)

The submitted design, cohort and headline results are unchanged (C = 0.857, 0.793, increment 0.064).

## Added (reviewer requests)
- Ground-truth simulation of the length-of-stay increment (`simulation_los_decomposition.py`).
- Missing-data assumptions and GCS missingness structure; risk-quartile results completing Sec. IV-D.
- Landmark sensitivity analysis (participants still in PTA after rehabilitation admission: C = 0.793).

## Corrected statements (final audit)
- Shared fraction: (0.793-0.5)/(0.857-0.5) = 82%, not "three quarters".
- IPCW median: the submitted 23 days is the self-normalised form (forces the distribution to reach one, although
  4% remain in PTA at the end of follow-up); the unnormalised form gives 26 days (weights sum to 0.91). Both are now
  reported: adjusted medians 23-26 days, so Kaplan-Meier (26) overstates by at most 3 days.
- Likelihood-ratio test: one split failed numerically (non-positive statistic); significant in all 19 converged splits.
- p-values: the naive t-tests were computed correctly but ignore overlapping splits; the Nadeau-Bengio corrected
  resampled t-test (1.1e-15; 2.2e-22) is reported instead.
- Decile analysis now reported with the stay score's own within-stratum concordance (0.553; quintiles 0.595).
- Quintile concordance 0.7615 is 0.761 (previously rounded to 0.762); cause-specific competing-risks C 0.8405 is 0.840.
- Feature ranking: the top three are top three in the consensus (and RSF, SHAP), not under every method; GCS
  complete-case rank (6th) is now compared with its rank by the same measure (7th), not with the consensus rank.
- PTADays = 0 codes "never in PTA"; 4,914 emergences (34.5%) preceded rehabilitation admission.
- Coding: FIM sentinel list removed 623/158 valid scores (C unchanged after correction); time to follow commands is
  recorded after admission for 899 participants (truncation changes C by < 0.001).
- References re-verified; an unlocatable reference replaced (Kowalski et al., 2021).
