# Senior ML / Code Review — Student Wellbeing Project

> Review date: 2026-06-04 · Reviewer scope: `Scripts/ML/` and `results/`
> Dataset: 4 participants (Mo, Beatrice, Alvaro, Alessia), participant column = `video_source`.

## Executive summary

The **evaluation machinery is genuinely good**: Leave-One-Subject-Out is implemented correctly by `video_source`, preprocessing is fit per-fold, identifiers are excluded from features, the hierarchical classifier is evaluated end-to-end, and unsupervised labels are only attached after clustering. The requirements are almost fully met.

**But the headline numbers are not trustworthy**, and the reason is upstream of the ML code, in `Scripts/build_training_dataset.py`. The perfect sleepy-vs-stress result is a **data-construction confound**, not a modelling success. This is ranked issue #1 because it changes how every result should be interpreted.

---

## 1. Requirement checklist

### Supervised

| Requirement | Status | Notes |
|---|---|---|
| Binary normal/abnormal, LOSO by `video_source` | ✅ Satisfied | `LeaveOneGroupOut` on `video_source` |
| Compare LR / SVM / RF / GB(or HistGB) | ✅ Satisfied | LR, SVM, RF, GradientBoosting |
| Binary metrics (acc, bal-acc, macro-F1, P, R, CM, report) | ✅ Satisfied | All present, per-fold + overall |
| Subtype sleepy/stress, LOSO, same models/metrics | ✅ Satisfied | — but result is confounded (see Major #1) |
| Hierarchical: normal→stop, abnormal→subtype | ✅ Satisfied | `evaluate_hierarchical_task` correct end-to-end |
| Hierarchical full metric set + per-fold | ✅ Satisfied | weighted-F1, per-class P/R, CM, report all there |
| Direct 3-class, excludes anomalous.csv | ✅ Satisfied | `build_direct_3class_dataset` uses normal+sleepy+stress only |
| Direct 3-class vs hierarchical comparison | ✅ Satisfied | `hierarchical_vs_direct_3class_comparison.csv` |

### Unsupervised

| Requirement | Status | Notes |
|---|---|---|
| Binary: KMeans / GMM / Agglomerative, 2 groups, labels post-hoc | ✅ Satisfied | — |
| Anomaly detection trained on normal only | ✅ Satisfied | — |
| LOSO anomaly version (train 3 normals, test held-out) | ✅ Satisfied | `run_anomaly_detection_loso` — clean |
| Binary eval (silhouette, ARI, NMI, mapped acc, bal-acc, macro-F1, CM, report) | 🟡 Partial | Metrics yes; **clustering** methods get a cluster-vs-label table but **no confusion-matrix PNG and no classification_report** (only anomaly detectors do). The 3-class script *does* save these — inconsistent. |
| 3-class: KMeans/GMM/Agglo, 3 groups | ✅ Satisfied | — |
| 3-class metrics (sil, DB, CH, ARI, NMI, homog, complet, V) | ✅ Satisfied | All present |
| Majority-vote only for interpretation | ✅ Satisfied | — |
| Plots + result CSVs | ✅ Satisfied | PCA/t-SNE saved; UMAP optional |

**Net:** essentially all requirements satisfied; only one minor gap (binary clustering confusion-matrix/report parity).

---

## 2. Major issues (ranked by severity)

### 🔴 #1 — Sleepy-vs-stress is a protocol confound, not a learned signal *(most important)*

The perfect `1.000` subtype result is the single most suspicious number in the project, and the data confirms why:

- Sleepy HR: mean **51 bpm**, range 27–79. Stress HR: mean **95 bpm**, range 66–126.
- Only **1.9%** of sleepy rows have HR > 65; only **7.8%** of stress rows have HR < 79. The classes are **near-disjoint on a single feature (HR)**, and likewise on RMMSD (sleepy 0.03–0.25 vs stress 0.0009–0.05).

The "stress" recordings were captured under a stress-induction protocol that raised heart rate, and "sleepy" at rest. So the model is detecting **which acquisition condition the heart trace came from**, not a subtle wellbeing distinction. A perfect LOSO score here is a *red flag*, not a success — it means the task, as constructed, is trivial.

**What to do:** report this explicitly; run sleepy-vs-stress **with HR/RMSSD removed** (posture+blink only) to show how much signal survives without the confound. The unsupervised ablation already hints at this — extend it to the supervised subtype task.

### 🔴 #2 — Heart features are joined by row position, from a different session

In `build_training_dataset.py` (`attach_heart`), HR/RMMSD come from separate `Heart_*.csv` files that have **no `window_start` and no `video_source`**, and are attached **by index position** (`iloc[:n]`, trimmed to the shorter length). Consequences:

- HR for window *i* is not time-synchronized with the posture/blink window *i*.
- Because the heart file has no participant key, HR rows are spread across all 4 participants' windows **by position** — the HR assigned to "Mo" windows may not be Mo's. This silently breaks the per-subject integrity that LOSO depends on for the HR channel.
- HR is also the top driver of the abnormal/subtype splits, so this fragility propagates into the most important results.

**What to do:** re-extract heart features keyed by participant + time window, or at minimum document that HR/RMSSD are *unaligned* and treat them as session-level, not window-level, features. Until then, HR-derived conclusions should be heavily caveated.

### 🟠 #3 — 4 participants ⇒ LOSO has only 4 folds; high variance

LOSO is the right choice, but n=4 means each fold is one person and estimates swing a lot. The binary RF Mo fold drops to **0.88 acc / 0.81 abnormal-recall**; direct-3-class Mo fold drops to **0.76** for LR/GB. The pooled "overall" numbers (0.957, 0.951) mask this. The interpretation text does report fold ranges (good), but the headline should lead with **mean ± std across the 4 folds**, not the pooled figure, and explicitly state that 4 subjects cannot support generalization claims.

### 🟠 #4 — Pooled "overall" metrics computed on concatenated fold predictions

`evaluate_flat_task` concatenates all fold predictions and computes one metric set. That's a legitimate "micro-averaged over folds" estimate, but it weights larger participants more and hides per-subject variance. Report it alongside the **macro-over-folds mean ± std**, and use the latter as the primary number. (Per-fold rows are already stored, so this is a reporting change only.)

---

## 3. Data-leakage audit

All **clean**:

- ✅ Features = the 18 numeric sensor columns only; `window_start`, `video_source`, `label`, `true_label` never enter `X` (`df.loc[:, features]`).
- ✅ Supervised preprocessing fit only on training folds — every model is a `Pipeline` that is `clone()`d and `.fit()` per fold. No global scaler.
- ✅ LOSO anomaly detection fits imputer+scaler+detector **on training-normal only**, then `transform`s the held-out fold. Correct.
- ✅ `anomalous.csv` is excluded from direct 3-class (verified: anomalous = sleepy+stress exactly, 431 rows; direct task rebuilds from the three base files).

One nuance, not strictly leakage: in the **non-LOSO** clustering (`preprocess_full`) the scaler is fit on all rows including the points being evaluated. That's standard and acceptable for transductive clustering, but the silhouette/ARI numbers still pool correlated subjects — worth a sentence acknowledging it.

---

## 4. Result interpretation

- **Binary 0.957 / subtype 1.000 / hierarchical 0.957** — the binary and hierarchical numbers are *plausible* (posture features like `p_head_tilt_std`/`p_head_tilt_mean` dominate importance, which is physiologically sensible for slumping/drowsiness). The **subtype 1.000 is not credible as a wellbeing result** — see Major #1.
- **Fold stability**: not stable. Alessia/Beatrice ≈ perfect; Mo is consistently the hard fold (0.88 binary, 0.76–0.88 3-class). This is exactly what 4-subject LOSO should reveal and it's good that it does — but it means the pooled headline overstates reliability.
- **Hierarchical ≈ direct (0.957 vs 0.951)**: sensible conclusion that the extra hierarchy buys little; worth stating that the difference is within fold noise, so "hierarchical is better" is not supported.
- **Unsupervised** (GMM binary macro-F1 0.81, 3-class 0.82; LOF LOSO bal-acc 0.78 with **0 anomalous missed but 97 normals flagged**) — the LOF confusion matrix `[[121,97],[0,431]]` shows it basically calls everything anomalous; balanced accuracy is the honest metric there and should be foregrounded over the 0.851 accuracy.
- **Conclusions appropriately limited?** Yes — both summaries repeatedly state "proof-of-concept, 4 participants, more needed." Good scientific hygiene. The one thing missing is naming the **HR confound** as a specific limitation.

---

## 5. Code quality

**Strengths:** clean, typed, well-documented, deterministic (`random_state=42`), pipelines used correctly, results organized into clear subfolders, notebooks are thin wrappers that `import` the scripts (so they **cannot drift** — nice choice).

**Issues:**

- **Heavy duplication across the three scripts**: `validate_features`, `coerce_feature_columns`, `candidate_group_columns`, `infer_group_column`, `_column_is_group_like`, `classification_metrics`, `majority_vote_mapping`, `save_confusion_matrix_png`, `FEATURE_COLUMNS`, `IDENTIFIER_HINTS`, `GROUP_PRIORITY` are copy-pasted. Extract a shared `wellbeing_common.py`. This is ~300 duplicated lines.
- **Inconsistent per-class metric keys**: the supervised script uses `safe_label()` (→ `precision_sleepy`), the unsupervised ones use the raw label in the f-string. Unify.
- **Binary clustering lacks the confusion-matrix PNG + classification_report** that the 3-class script produces — make `evaluate_clustering` symmetric across both scripts (fold this into the shared module).
- **Committed artifacts**: `results/**/.matplotlib/fontlist-v390.json` is a matplotlib cache and shouldn't be in git. Add `results/**/.matplotlib/` to `.gitignore`.
- A couple of unused imports (`Iterable`, `Optional`) in the hierarchical script.

---

## 6. Concrete recommendations

### ML methodology

1. **Re-report every supervised task as mean ± std over the 4 LOSO folds**, with pooled micro-average as secondary. Add 95% CIs or at least the min/max already computed.
2. **Run the subtype and binary tasks with HR/RMSSD ablated.** This is the single most valuable experiment: it tells you whether posture+blink alone carry the signal, and quantifies how much of the 0.957/1.000 is the HR confound.
3. **Fix or caveat heart-feature alignment** (Major #2). If you can't re-key it, run a sensitivity analysis dropping HR/RMSSD entirely and report both.
4. **Add a permutation / label-shuffle baseline** and a majority-class baseline so reviewers can see the headroom above chance under LOSO.
5. **Nested CV or fixed, justified hyperparameters.** Current models use defaults (`C=1.0`, RF `n_estimators=500`). With 4 folds you can't really tune, so either keep defaults and say so explicitly, or do leave-one-subject-out *inner* CV for tuning. Don't tune on the test fold.
6. **Calibration & uncertainty**: for the binary stage, add `CalibratedClassifierCV` and report Brier score / reliability, and an abstention threshold for the hierarchical router (low-confidence → "uncertain" rather than forcing normal/abnormal).
7. **Subject-aware unsupervised eval**: report clustering ARI/NMI *per held-out subject* too, not just pooled, to match the supervised rigor.

### Feature engineering / reporting

8. Rename `RMMSD`→`RMSSD` at the data-build step (the scripts already handle the alias gracefully, but fixing the source avoids the perpetual footnote).
9. Report **per-feature distributions split by class and by participant** — this would have surfaced the HR confound immediately and belongs in the write-up.
10. Add a short **data provenance** section documenting that posture/blink are window-aligned but HR is positionally attached from a separate session.

### Additional experiments that would strengthen it

- Posture-only vs blink-only vs HR-only vs all (already done for unsupervised — add it for supervised).
- Group-stratified learning curve: 2 vs 3 training subjects, to show the trend toward (in)stability.
- More participants — the only real fix for generalization.

---

## 7. Final judgment

**Acceptable as a proof-of-concept — yes, with one mandatory caveat.** The pipeline design, LOSO discipline, leakage control, and reporting structure are above the level typically seen at this stage; the code is clean and the notebooks stay in sync with the scripts. The requirements are met.

**Before any claim of strong generalization, you need:**

1. To **explicitly flag and quantify the HR/stress-protocol confound** (and re-run subtype without HR) — otherwise the 1.000 will (rightly) be read as too good to be true.
2. To **fix or formally caveat the positional heart-feature alignment**.
3. **More subjects.** Four-subject LOSO is a sanity check, not evidence of generalization; report mean ± std over folds and stop leading with pooled numbers.
4. Minor: refactor the duplicated code into a shared module and add the missing binary-clustering confusion-matrix/report for parity.

Address #1 and #2 and this moves from "promising prototype with one misleading result" to "honest, defensible proof-of-concept."
