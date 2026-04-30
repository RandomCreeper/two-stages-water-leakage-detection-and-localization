# Two-Stage Water Leakage Detection and Localization

A BattLeDIM-style water-leak detection and localization project with **two versions of the pipeline**:

- a **full-size research version** for analysis, notebook experiments, and reference performance
- a **TinyML / microcontroller-oriented version** that keeps the same two-stage idea while fitting a small embedded budget

This repository was built for an ML709 IoT Smart Systems, Services and Applications project focused on **detecting and localizing water leaks from 5-minute SCADA data** using a **two-stage detector deployable on a microcontroller**.

---
## Repository structure

```text
.
├── full_version/
│   ├── battledim_stage1.py
│   ├── battledim_stage1_tuned_config.json
│   └── battledim_stage2_a_v2.ipynb
├── preprocess/
│   ├── data_preprocessing.ipynb
│   └── preprocessed_data_check.ipynb
├── battledim_stage1_tiny.py
├── battledim_tinyml_final.py
├── battledim_tinyml_driver.py
└── battledim_tinyml_export.py
```

### What each part does

#### `preprocess/`
Contains the data-preparation notebooks.

- `data_preprocessing.ipynb`  builds or cleans the merged SCADA table used by the models
- `preprocessed_data_check.ipynb` validates the processed dataset and checks the resulting schema

#### `full_version/`
Contains the **reference / full-size pipeline**.

- `battledim_stage1.py`  recall-first Stage 1 detector
- `battledim_stage1_tuned_config.json`  tuned Stage 1 configuration
- `battledim_stage2_a_v2.ipynb`  patched Stage 2 notebook for packet building, confirmation, localization, evaluation, and reporting

#### TinyML scripts in the repo root
These implement the **compressed MCU-friendly version**.

- `battledim_stage1_tiny.py`  compact Stage 1 detector intended for embedded-style use
- `battledim_tinyml_final.py`  core TinyML pipeline implementation and exports
- `battledim_tinyml_driver.py`  easiest entry point to run all TinyML candidates and select the best one
- `battledim_tinyml_export.py`  exports final run artifacts from cached candidate summaries

---

## Data expected by the project

The project is built around a merged BattLeDIM-style table containing 5-minute samples with columns such as:

- `Timestamp`
- `press_1 ... press_33`
- `flow_1, flow_2, flow_3`
- `T1`
- many demand columns like `n4`, `n343`, etc.
- `pipe_1_bin ... pipe_14_bin`
- `leak_count`
- `leak_mag_total`
- `leak_binary`

### Important dataset notes

- `flow_3` behaves like a tank/control signal, so it is kept as **optional context**, not as a primary leak-evidence channel
- `leak_binary` is not the best main target because it is almost always positive in the merged table
- the more useful labels are typically:
  - `pipe_*_bin`
  - `leak_count`
  - `leak_mag_total`

If your CSV lives in a different path than the default one expected by the scripts, update the path inside the scripts or pass the correct input path through your workflow.

---

## Full-size pipeline

The full-size pipeline keeps the original BattLeDIM spirit.

### Full-size Stage 1
The full-size Stage 1 detector is a **cheap, recall-first streaming detector**.

It performs operations such as:

- causal smoothing
- weekly seasonality suppression
- pressure residual scoring
- pairwise pressure consistency checks
- flow-based context scoring
- EWMA / CUSUM / slope-style evidence accumulation
- leak-state tracking
- trigger generation for Stage 2

### Full-size Stage 2
The full-size Stage 2 is heavier and designed for offline experiments or desktop execution.

Its responsibilities are:

- build trigger packets from a lookback window
- compute high-dimensional packet features
- confirm whether a Stage 1 trigger is a likely leak
- predict the most likely leaking pipes
- export plots, tables, and evaluation outputs

This is the version to use when you want the **strongest desktop baseline** and the richest diagnostic outputs.

---

## TinyML pipeline

The TinyML version keeps the **same two-stage logic**, but compresses both stages so the system becomes realistic for a microcontroller target.

### What changed in TinyML

The TinyML work here is not just “prune a neural network.”
It is a broader **system-level compression** of the whole pipeline:

- sensor pruning
- feature pruning
- temporal compression
- smaller seasonal memory
- simpler model classes
- smaller runtime state and ring buffers
- Stage 2 runs only on triggers
- deployment-friendly parameter export

### TinyML techniques used

#### 1. Conditional computation
The biggest embedded-system trick is that **Stage 2 is not always running**.
Only Stage 1 stays active all the time.

This dramatically reduces average compute and energy cost.

#### 2. Sensor pruning
The TinyML detector monitors only a selected subset of the most useful sensors instead of treating every channel as equally important.

#### 3. Feature pruning
The original Stage 2 packet builder creates thousands of features.
The TinyML version compresses that into a small set of causal summary features.

#### 4. Temporal compression
The original system uses a detailed weekly baseline.
The TinyML system replaces it with much smaller seasonal memories such as:

- no seasonal table
- a 24-bin daily table
- a 168-bin weekly-hour table

#### 5. Model simplification
The TinyML version replaces large stage-2 models with much smaller alternatives such as:

- linear confirmation models
- prototype localizers
- linear one-vs-rest localizers

#### 6. Deployment-oriented quantization / export
Instead of deploying Python model objects directly, the final parameters are exported as compact JSON and C/C++ header files.

---

## TinyML candidates evaluated

The TinyML pipeline evaluates three candidate compressed models.

### 1. `ultra_tiny_none_proto`
**Meaning:**
- `ultra_tiny` = most aggressively compressed candidate
- `none` = no explicit seasonal lookup table in Stage 1
- `proto` = Stage 2 uses a prototype-based localizer

**Interpretation:**
This is the smallest end-to-end version. Stage 1 uses causal streaming statistics without storing a seasonal baseline. Stage 2 confirms leaks with a small linear model and localizes by comparing trigger packets to one average pattern per pipe.

**Best for:**
- minimum memory
- easiest firmware implementation
- simplest logic

### 2. `tiny_daily24_proto`
**Meaning:**
- `tiny` = still MCU-friendly
- `daily24` = 24-bin daily baseline in Stage 1
- `proto` = Stage 2 uses a prototype-based localizer

**Interpretation:**
This version stores one expected normal value per hour of the day, which helps remove regular day/night demand variation before anomaly scoring. Stage 2 stays compact with prototype localization.

**Best for:**
- a good middle ground
- preserving some daily operating pattern awareness
- staying very small

### 3. `tiny_weekly168_ovr`
**Meaning:**
- `tiny` = MCU-friendly
- `weekly168` = 168 weekly-hour baseline in Stage 1
- `ovr` = one-vs-rest linear localizer in Stage 2

**Interpretation:**
This is the best balanced candidate. It compresses the original weekly baseline idea down to one value per hour of the week and uses one lightweight linear classifier per pipe for localization.

**Best for:**
- best size vs performance trade-off
- preserving the original BattLeDIM Stage 1 logic more faithfully
- stronger localization than prototype matching

---

## Recommended TinyML candidate

The best final candidate from the TinyML experiments is:

### `tiny_weekly168_ovr`

Why it was selected:
- it stays far below the MCU budget
- it preserves the original time-of-week normalization idea
- it keeps Stage 1 behavior very close to the full-size reference
- it gives the strongest overall balance of confirmation and localization quality

### Example selected TinyML run

**Chosen candidate:** `tiny_weekly168_ovr`

**Approximate resource use:**
- RAM: **11,712 bytes**
- Flash: **59,690 bytes**

**Compared with the patched full-size reference:**
- Stage 1 recall on 2018 remained **1.0000**
- Stage 1 calls/day stayed close to **2.0**
- Stage 2 2019 binary recall on actual trigger packets was **0.9808**
- Stage 2 2019 top-3 localization hit rate was **0.8890**

This is much smaller than the full patched reference, which is roughly:
- **157,032 bytes RAM**
- **7,687,536 bytes Flash**

So the TinyML version keeps the project idea while making deployment realistic for a small embedded target.

---

## Quick start

## 1) Preprocessing
Open the notebooks in `preprocess/` and run them in order:

1. `preprocess/data_preprocessing.ipynb`
2. `preprocess/preprocessed_data_check.ipynb`

Use this stage to generate or validate the merged dataset before running the models.

---

## 2) Run the full-size Stage 1 detector
From the repo root, adjust paths as needed and run:

```bash
python full_version/battledim_stage1.py \
  --merged-csv data_full_2018_2019_fixed.csv \
  --timestamp-col Timestamp \
  --train-year 2018 \
  --test-year 2019 \
  --output-dir ./stage1_outputs
```

Then open the Stage 2 notebook:

```text
full_version/battledim_stage2_a_v2.ipynb
```

That notebook is the main entry point for the full-size Stage 2 experiments.

---

## 3) Run the TinyML pipeline
The easiest command is:

```bash
python battledim_tinyml_driver.py --output-root ./outputs
```

This will:
- evaluate the TinyML candidates
- compare them
- choose the best one
- write a timestamped output folder with artifacts

### Evaluate a single candidate

```bash
python battledim_tinyml_driver.py \
  --eval-candidate tiny_weekly168_ovr \
  --summary-json ./eval/tiny_weekly168_ovr_summary.json \
  --memory-csv ./eval/tiny_weekly168_ovr_memory.csv
```

### Export from cached candidate summaries

```bash
python battledim_tinyml_export.py \
  --summary-dir ./eval \
  --output-root ./outputs
```

---

## TinyML output artifacts

A successful TinyML run produces files such as:

- `tinyml_metadata.json`
- `tinyml_memory_budget.csv`
- `tinyml_model_comparison.csv`
- `tinyml_stage1_metrics_2018.csv`
- `tinyml_stage1_metrics_2019.csv`
- `tinyml_stage2_binary_metrics_val_actual.csv`
- `tinyml_stage2_binary_metrics_2019.csv`
- `tinyml_stage2_localisation_metrics_val.csv`
- `tinyml_stage2_localisation_metrics_2019.csv`
- `tinyml_predictions_2019.csv`
- `tiny_stage1_params.json`
- `tiny_stage1_params.h`
- `tiny_stage2_params.json`
- `tiny_stage2_params.h`

These artifacts are meant for:
- analysis
- plotting
- report writing
- firmware integration

---

## How to think about “full model” vs “TinyML model”

### Full-size model
Think of the full-size system as a detailed analyst:
- more sensors
- more features
- finer time memory
- larger models
- more diagnostic richness

### TinyML model
Think of the TinyML system as the **pocket version of the same analyst**:
- fewer sensors
- fewer features
- compressed time memory
- smaller models
- lower RAM/Flash usage
- still keeps the same two-stage reasoning

So the TinyML work here is **not a totally different project**.
It is the same project idea compressed into a deployment-friendly form.

---

## Why this method is a good fit for water-leak detection

This approach is strong for this problem because:

1. water-network signals have recurring daily and weekly behavior, so Stage 1 can model normal patterns and detect deviations
2. leaks often create persistent changes, so packet-level confirmation works better than single-timestep classification
3. a two-stage design is operationally realistic, always-on cheap screening, trigger-only deeper analysis
4. the localization step is only run when needed, which is ideal for embedded deployment
5. the TinyML version preserves the useful structure instead of oversimplifying the whole task into one weak classifier

---

## Questions

Questions, issues, and suggestions are welcome.
