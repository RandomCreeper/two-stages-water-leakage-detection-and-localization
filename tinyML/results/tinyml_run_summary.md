# BattLeDIM tiny MCU phase summary

## Current architecture from the attachments

- Stage 1 is the recall-first BattLeDIM detector described in the notebook/README: causal smoothing, seasonality suppression, pressure and pairwise residual evidence, flow context, a leak-state machine, and periodic Stage 2 rechecks.
- The patched Stage 2 in the attached run uses 12-hour packets (144 steps), four summary horizons `[12, 36, 72, 144]`, pseudo-hard negatives, a calibrated HistGradientBoosting binary head, and one-vs-rest pipe classifiers.
- The attached reference run reports 721 actual late-2018 triggers (714 positive, 7 negative), threshold 0.9739572638050862, 2019 binary recall 0.964384, and 2019 localisation top-3 hit rate 0.716438.

## Why the current implementation is too large for the MCU target

- The current feature builder expands each trigger to about **3146 binary features** and **3207 localisation features**.
- The current Stage 2 packet lookback alone is roughly **92.8 KB RAM** and the float32 feature workspace adds about **12.5 KB RAM**.
- Stage 1's original 2016-bin weekly baseline table is about **291.4 KB flash** on its own.
- The reference boosted Stage 2 models are conservatively estimated at about **7507.4 KB flash**, well beyond a 1 MB microcontroller budget.

## Candidate tiny pipelines considered

- **ultra_tiny_none_proto**: Stage 1 `none / 6 pressures`, localizer `prototype`, RAM 10.0 KB, Flash 47.9 KB, Stage 1 recall 0.9755, val-actual binary recall 0.9500, 2019 top-3 0.8822.
- **tiny_daily24_proto**: Stage 1 `daily_hour / 6 pressures`, localizer `prototype`, RAM 10.1 KB, Flash 48.9 KB, Stage 1 recall 0.9960, val-actual binary recall 0.9507, 2019 top-3 0.9370.
- **tiny_weekly168_ovr**: Stage 1 `weekly_hour / 8 pressures`, localizer `ovr_logreg`, RAM 11.4 KB, Flash 58.3 KB, Stage 1 recall 1.0000, val-actual binary recall 1.0000, 2019 top-3 0.8890.

## Chosen tiny architecture

- **Chosen candidate:** `tiny_weekly168_ovr` - 168-bin weekly-hour Stage 1 baseline with linear OVR localizer.
- Selection priority followed the project brief: MCU fit first, then Stage 1 fidelity and practical call rate, then Stage 2 recall on true validation triggers, then localisation.
- **Stage 1:** `weekly_hour` baseline table, `8` selected pressure sensors, `2` pairwise anchors, same `battledim_stage1` import path/API, causal leak-state logic, and 12-hour periodic rechecks.
- **Stage 2 binary head:** folded linear logistic model over `30` compact causal packet features.
- **Stage 2 localizer:** `ovr_logreg` over `40` compact features, returning per-pipe scores and top-3 rankings.

## MCU fit check

- Estimated runtime RAM: **11.4 KB**.
- Estimated flash footprint: **58.3 KB**.
- Both are inside the requested ~256 KB RAM and ~1 MB Flash budget with an explicit stack/workspace allowance included.

## Performance vs the provided patched baseline

- Stage 1 late-2018 timestep recall delta: **+0.0000**.
- Stage 1 late-2018 calls/day: **1.9972** vs reference **2.0018**.
- Stage 2 binary validation-actual recall: **1.0000** vs reference **1.0000**.
- Stage 2 binary 2019 recall delta: **+0.0164**.
- Stage 2 localisation validation top-3: **1.0000** vs reference **1.0000**.
- Stage 2 localisation 2019 top-3 delta: **+0.1726**.

## Risks / caveats

- Late-2018 actual trigger validation still has only **7 real hard-negative Stage 1 triggers** for the chosen tiny pipeline, so binary false-alarm rejection uncertainty remains high.
- As in the attached patched run, 2019 is effectively all-positive in this merged table, so it cannot measure binary false-alarm rejection. 2019 remains useful mainly for recall and localisation continuity.
- The exported C headers are parameter tables and folded coefficients. Firmware still needs a small runtime wrapper for streaming median, EWMA/CUSUM, rolling sums, and sigmoid/logit evaluation.

## Files written

- See `tinyml_artifact_manifest.csv` for the complete file list.