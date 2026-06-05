# Calibration & Augmentation Pipelines

Design proposal for the group. Two pipelines:

- **Pipeline A** — re-collecting data so that *calibration* becomes valid.
- **Pipeline B** — *data augmentation* for the current (and future) dataset.

Both are grounded in the existing setup: the `{Person}_{class}.mp4` videos under
`Video/`, the extraction scripts `Scripts/Posture/main.py` and
`Scripts/Blinking/blink_main.py`, the `video_source` column in
`data/supervised`, and **Leave-One-Subject-Out (LOSO)** validation.

## Why these changes

- **Only 4 subjects.** The ~218 windows per class are sliding windows (60 s
  window, 10 s step) and heavily overlap, so the effective independent sample
  size is ≈ 4 people. LOSO fold stability is poor (macro-F1 ranges 0.39–0.98).
- **Calibration was blocked** by how data was collected: the three condition
  videos per person were recorded in *different sessions* and the sleepy/stress
  clips contain *no neutral period*, so no valid per-session baseline existed.
- The new protocol in Pipeline A fixes this by recording **one session per
  participant** with a dedicated **10 s neutral calibration clip**.

---

# Pipeline A — Data collection for calibration

**Core idea:** record each participant in a **single uninterrupted session**
(same setup throughout). At the start, capture **10 s of neutral, upright, alert
posture** and save it as a **separate calibration video**. The conditions are
then recorded in that same session.

## Stage 1 — Recording protocol

- **One session per participant.** Camera position, distance, angle, seat and
  lighting must **not change** for the whole session.
- **Calibration clip (first 10 s):** participant sits upright, neutral, alert,
  eyes naturally open, looking at the screen, **blinking normally** (do *not*
  force the eyes open — natural blinking is fine and handled in Stage 2). Save as
  a **dedicated file**, e.g. `Video/Calibration/{Person}_calib.mp4`.
- **Condition recordings:** still in the same session and same setup, induce and
  record normal → sleepy → stress, saved as `{Person}_normal.mp4`,
  `{Person}_sleepy.mp4`, `{Person}_stress.mp4`.
- **Interruptions are fine:** if the recording software splits the session into
  several files, that is OK — as long as it is the *same physical session and
  setup*, all parts share the same calibration clip.
- Keep the calibration folder **separate** from `Video/Normal|Sleepy|Stress` so
  the extraction scripts do not treat the calib clip as training data.
- Log metadata per participant: ID, date, camera setup, recording order.

**Why this fixes the previous problem:** the neutral reference and all condition
data now live in the *same session* → identical camera/pose/lighting → the
baseline transfers correctly. The condition (sleepy/stress) still appears as a
deviation from the alert baseline → the signal is preserved, not erased. The
earlier "different sessions" confound is removed.

## Stage 2 — Compute the per-person baseline (one per participant)

From `{Person}_calib.mp4`:

- **Posture neutral:** median of shoulder angle, head tilt, and FHP over the
  10 s → the participant's neutral pose.
- **EAR open-eye reference:** a high percentile (e.g. 90th) of EAR over the 10 s
  → the participant's fully-open eye. The high percentile **automatically picks
  the open-eye level and ignores the brief blink dips**, so the participant does
  *not* need to suppress blinking. The only requirement is that they are
  **alert** during the clip (a drowsy calibration would set the baseline too
  low). At ~30 fps, 10 s ≈ 300 frames, leaving plenty of open-eye frames even
  with normal blinking.

One baseline per participant, applied to **all three** of their condition videos.

## Stage 3 — Extract features *with* calibration

Re-run `Scripts/Posture/main.py` + `Scripts/Blinking/blink_main.py`, but
transform the per-frame raw values **before windowing**:

- **Posture:** subtract the neutral pose (deviation from neutral); FHP expressed
  relative to the neutral ratio.
- **Blink:** set the "eye closed" threshold to a fraction (e.g. 0.6) of the
  participant's open-eye EAR baseline, instead of the fixed
  `EAR_CLOSED_THRESHOLD = 0.21`.

Then window as usual → calibrated `data/raw` → `build_training_dataset.py` →
calibrated `data/supervised`.

> Note: this requires re-running extraction from video. `data/raw` only stores
> windowed features, so calibration that acts per-frame cannot be applied to the
> existing CSVs.

## Stage 4 — Validate

- Keep **LOSO grouped by `video_source`**.
- Compare calibrated vs uncalibrated **fold stability** (today LOSO macro-F1
  swings from 0.39 to 0.98). A smaller swing = calibration helped. This is the
  number that justifies the re-recording.

## Stage 5 — Deployment consistency (non-negotiable)

- The live GUI adds the **same** step: record 10 s neutral/alert at session
  start, compute the same baseline, apply the same per-frame transform during
  inference.
- Train and inference must use **identical** calibration logic → no
  train/inference mismatch.

## Caveats to flag

- Works only if **every** participant (and every live user) has the 10 s neutral
  clip. Old videos cannot be retrofitted.
- The 10 s must be genuinely neutral and alert; if the participant fidgets at the
  very start, trim to a stable sub-window (with only 10 s there is less margin
  than with a longer clip, so make sure the segment is clean).
- The single-session design also removes the condition/session confound we had
  before — worth stating as a methodological improvement.

---

# Pipeline B — Data augmentation

**Goal:** improve robustness without inventing subjects, without breaking LOSO,
and without erasing the class signal.

## The cardinal rule — group by `video_source`

- Every augmented sample **inherits the `video_source` (participant)** of its
  origin.
- LOSO splits by `video_source`, so augmented copies of the held-out participant
  are held out too → **no leakage**.
- This makes it safe to **pre-bake** augmented rows into the CSVs, *as long as*
  `video_source` is correct. Never invent a new participant name.

## Level 1 — Raw-level augmentation (preferred: realistic correlations)

Regenerate features by re-running extraction on transformed video:

1. **Horizontal mirror** — flip the frames. Shoulder/head-tilt use `np.abs` and
   FHP is symmetric, so abs-based features are preserved; removes left/right bias
   and **doubles the data**. Start here.
2. **Photometric variants** — brightness / contrast / gamma / mild Gaussian
   noise → slightly perturb landmark detection → realistic EAR/posture jitter.
   Generate 2–3 per clip.
3. **Temporal variants** — phase-shifted window offsets, simulated lower frame
   rate / frame dropout (robustness to detection gaps). Mild time-warp (±5–10 %)
   **with caution** (it shifts `b_blink_freq` and HR alignment).

Each augmented clip → extraction → new rows in `data/raw` with the **same
`video_source` and class**.

## Level 2 — Feature-level augmentation (cheap supplement)

On the 18-dim feature vectors, after extraction:

- **Gaussian jitter** scaled to each feature's std (small, ~5–10 %).
- **Within-subject, within-class interpolation** (mixup/SMOTE restricted to same
  `video_source` + same `label`) → never blend different people into a fake
  "average person."

## What NOT to do

- No **cross-person** SMOTE/mixup → creates fake subjects, corrupts LOSO.
- Never augment the **held-out test fold**.
- No transform that **changes the label meaning** (e.g. heavy time-warp turning a
  normal blink rate into a sleepy-looking one).
- No vertical flip — meaningless for posture.

## Validate

- Compare LOSO **with vs without** augmentation. Success = tighter fold variance
  and better minority-class recall, **not** just a higher average.
- Make sure augmentation isn't only inflating the (already optimistic) non-LOSO
  numbers.

## Suggested order

1. Mirror (doubles data).
2. Photometric (×2–3).
3. Optional feature-level jitter as a light final layer.
4. Re-run LOSO and report the deltas.

## Interaction with calibration (Pipeline A)

- Compute the baseline from the **calibration clip**, then apply it to the
  augmented condition clips of the same participant. For the mirror, the
  `np.abs` symmetry already keeps posture consistent; for photometric/temporal
  variants, apply the participant's baseline unchanged (same session, same
  setup).
