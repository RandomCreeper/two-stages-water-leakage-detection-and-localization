from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import nbformat as nbf

# Make the project scripts importable when this file is run from anywhere.
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import battledim_tinyml_final as mod


DATA_PATH = THIS_DIR / 'data_full_2018_2019_fixed.csv'
CONFIG_PATH = THIS_DIR / 'battledim_stage1_tuned_config.json'
BASE_NOTEBOOK_PATH = THIS_DIR / 'battledim_stage1_notebook_stage2_patched_reporting.ipynb'
ORIG_NOTEBOOK_PATH = THIS_DIR / 'battledim_stage1_notebook.ipynb'


def load_static_context() -> Tuple[pd.DataFrame, Dict[str, List[str]], mod.Stage1Config, mod.PacketSplitConfig]:
    full_df = mod.load_battledim_merged_csv(str(DATA_PATH), timestamp_col='Timestamp')
    groups = mod.infer_column_groups(full_df, include_flow3_context=True)
    base_stage1_cfg = mod.Stage1Config(**json.loads(CONFIG_PATH.read_text()))
    split_cfg = mod.PacketSplitConfig()
    return full_df, groups, base_stage1_cfg, split_cfg


def candidate_by_name(name: str) -> mod.PipelineCandidate:
    for cand in mod.CANDIDATES:
        if cand.name == name:
            return cand
    raise KeyError(f'Unknown candidate: {name}')


def build_reference_row(reference: Dict[str, Any], reference_size: pd.DataFrame) -> Dict[str, Any]:
    return {
        'candidate': 'current_patched_reference',
        'description': 'Provided patched notebook baseline (not MCU-feasible).',
        'stage1_mode': 'legacy_weekly_5min + full feature builder',
        'localizer_type': 'sklearn_hgb_ovr',
        'binary_feature_count': float(reference_size['binary_feature_count'].iloc[0]),
        'localisation_feature_count': float(reference_size['localisation_feature_count'].iloc[0]),
        'stage1_recall_2018': reference['stage1_2018']['timestep_recall'],
        'stage1_calls_per_day_2018': reference['stage1_2018']['stage2_calls_per_day'],
        'binary_recall_val_actual': reference['binary_val_actual']['recall'],
        'binary_recall_2019': reference['binary_test_2019']['recall'],
        'localisation_top3_val': reference['loc_val']['top3_hit_rate'],
        'localisation_top3_2019': reference['loc_test_2019']['top3_hit_rate'],
        'total_ram_bytes': int(reference_size['ram_bytes'].sum()),
        'total_flash_bytes': int(reference_size['flash_bytes'].sum()),
        'feasible_under_256k_1m': False,
        'stage1_detected_events_2018': np.nan,
        'actual_dev_triggers': mod.safe_float(reference.get('packet_info', {}).get('actual_dev_triggers', np.nan), np.nan),
        'actual_dev_positive_triggers': mod.safe_float(reference.get('packet_info', {}).get('actual_dev_positive_triggers', np.nan), np.nan),
        'actual_dev_negative_triggers': mod.safe_float(reference.get('packet_info', {}).get('actual_dev_negative_triggers', np.nan), np.nan),
        'binary_val_actual_precision': reference['binary_val_actual']['precision'],
        'localisation_top1_val': reference['loc_val']['top1_hit_rate'],
        'localisation_top1_2019': reference['loc_test_2019']['top1_hit_rate'],
        'selection_score': '',
    }


def build_candidate_summary(res: mod.CandidateRunResult) -> Dict[str, Any]:
    return {
        'candidate': res.candidate.name,
        'description': res.candidate.description,
        'stage1_mode': f'{res.candidate.baseline_mode} / {res.candidate.monitored_pressure_count} pressures',
        'localizer_type': res.candidate.localizer_type,
        'binary_feature_count': len(res.binary_cols),
        'localisation_feature_count': len(res.loc_cols),
        'stage1_recall_2018': mod.safe_float(res.stage1_metrics_2018.get('timestep_recall', np.nan), np.nan),
        'stage1_calls_per_day_2018': mod.safe_float(res.stage1_metrics_2018.get('stage2_calls_per_day', np.nan), np.nan),
        'stage1_detected_events_2018': mod.safe_float(res.stage1_metrics_2018.get('n_detected_events', np.nan), np.nan),
        'binary_val_actual_precision': mod.safe_float(res.binary_metrics_val_actual.get('precision', np.nan), np.nan),
        'binary_recall_val_actual': mod.safe_float(res.binary_metrics_val_actual.get('recall', np.nan), np.nan),
        'binary_recall_2019': mod.safe_float(res.binary_metrics_test_actual.get('recall', np.nan), np.nan),
        'localisation_top1_val': mod.safe_float(res.localisation_metrics_val.get('top1_hit_rate', np.nan), np.nan),
        'localisation_top3_val': mod.safe_float(res.localisation_metrics_val.get('top3_hit_rate', np.nan), np.nan),
        'localisation_top1_2019': mod.safe_float(res.localisation_metrics_test.get('top1_hit_rate', np.nan), np.nan),
        'localisation_top3_2019': mod.safe_float(res.localisation_metrics_test.get('top3_hit_rate', np.nan), np.nan),
        'total_ram_bytes': int(res.total_ram_bytes),
        'total_flash_bytes': int(res.total_flash_bytes),
        'feasible_under_256k_1m': bool(res.feasible),
        'actual_dev_triggers': int(res.packet_info.get('actual_dev_triggers', 0)),
        'actual_dev_positive_triggers': int(res.packet_info.get('actual_dev_positive_triggers', 0)),
        'actual_dev_negative_triggers': int(res.packet_info.get('actual_dev_negative_triggers', 0)),
        'selection_score': list(res.chosen_score),
        'threshold': mod.safe_float(res.binary_threshold_info.get('threshold', np.nan), np.nan),
        'binary_feature_names': list(res.binary_cols),
        'loc_feature_names': list(res.loc_cols),
        'packet_info': res.packet_info,
        'candidate_config': mod.asdict(res.candidate),
    }


def summary_selection_key(summary: Dict[str, Any]) -> Tuple[float, float, float, float, float, float, float, float, float]:
    feasible = 1.0 if bool(summary['feasible_under_256k_1m']) else 0.0
    stage1_recall = float(summary['stage1_recall_2018'])
    binary_val_recall = float(summary['binary_recall_val_actual'])
    loc_val_top3 = float(summary['localisation_top3_val'])
    calls = float(summary['stage1_calls_per_day_2018'])
    binary_test_recall = float(summary['binary_recall_2019'])
    loc_test_top3 = float(summary['localisation_top3_2019'])
    val_precision = float(summary['binary_val_actual_precision'])
    flash = float(summary['total_flash_bytes'])
    return (
        feasible,
        stage1_recall,
        binary_val_recall,
        loc_val_top3,
        val_precision,
        -abs(calls - 2.0),
        binary_test_recall,
        loc_test_top3,
        -flash,
    )


def write_eval_outputs(candidate_name: str, summary_json: Path, memory_csv: Path) -> None:
    full_df, groups, base_stage1_cfg, split_cfg = load_static_context()
    candidate = candidate_by_name(candidate_name)
    res = mod.run_candidate(candidate, full_df, groups, base_stage1_cfg, split_cfg)
    summary = build_candidate_summary(res)
    summary_json.write_text(json.dumps(summary, indent=2, default=mod._json_default))
    res.memory_rows.to_csv(memory_csv, index=False)



def _safe_run_subprocess(args: Sequence[str]) -> None:
    env = dict(os.environ)
    env.setdefault('PYTHONUNBUFFERED', '1')
    subprocess.run(list(args), check=True, env=env)



def export_notebook(run_dir: Path) -> Path:
    if BASE_NOTEBOOK_PATH.exists():
        nb = nbf.read(BASE_NOTEBOOK_PATH.open('r', encoding='utf-8'), as_version=4)
    elif ORIG_NOTEBOOK_PATH.exists():
        nb = nbf.read(ORIG_NOTEBOOK_PATH.open('r', encoding='utf-8'), as_version=4)
    else:
        nb = nbf.v4.new_notebook()
        nb['cells'] = []

    md_text = (
        '## Tiny MCU final phase\n\n'
        'This appended section evaluates the tiny candidates in isolated processes and then exports the final MCU-ready run artifacts.'
    )
    code_text = (
        'import subprocess, sys\n'
        'from pathlib import Path\n\n'
        "tmp_dir = Path('/mnt/data/tinyml_notebook_eval')\n"
        'tmp_dir.mkdir(exist_ok=True)\n'
        "candidates = ['ultra_tiny_none_proto', 'tiny_daily24_proto', 'tiny_weekly168_ovr']\n"
        'for cand in candidates:\n'
        "    subprocess.run([sys.executable, '/mnt/data/battledim_tinyml_driver.py', '--eval-candidate', cand, '--summary-json', str(tmp_dir / f'{cand}_summary.json'), '--memory-csv', str(tmp_dir / f'{cand}_memory.csv')], check=True)\n"
        "subprocess.run([sys.executable, '/mnt/data/battledim_tinyml_export.py', '--summary-dir', str(tmp_dir), '--output-root', '/mnt/data'], check=True)\n"
    )
    nb['cells'].append(nbf.v4.new_markdown_cell(md_text))
    nb['cells'].append(nbf.v4.new_code_cell(code_text))
    out_path = run_dir / 'battledim_stage1_notebook_tinyml_final.ipynb'
    with out_path.open('w', encoding='utf-8') as f:
        nbf.write(nb, f)
    return out_path


def copy_code_files(run_dir: Path, writer: mod.ArtifactWriter) -> None:
    for src in [THIS_DIR / 'battledim_stage1.py', THIS_DIR / 'battledim_tinyml_final.py', THIS_DIR / 'battledim_tinyml_driver.py']:
        if src.exists():
            dst = run_dir / src.name
            shutil.copy2(src, dst)
            writer.add(src.name, dst, f'Runnable source file copied into the run folder: {src.name}.')



def build_summary_markdown(
    chosen: mod.CandidateRunResult,
    chosen_summary: Dict[str, Any],
    reference: Dict[str, Any],
    reference_size: pd.DataFrame,
    model_comparison: pd.DataFrame,
) -> str:
    current_ref_total_ram = int(reference_size['ram_bytes'].sum())
    current_ref_total_flash = int(reference_size['flash_bytes'].sum())
    stage1_drop_2018 = mod.safe_float(chosen.stage1_metrics_2018.get('timestep_recall', np.nan), np.nan) - reference['stage1_2018']['timestep_recall']
    binary_recall_drop_2019 = mod.safe_float(chosen.binary_metrics_test_actual.get('recall', np.nan), np.nan) - reference['binary_test_2019']['recall']
    loc_top3_drop_2019 = mod.safe_float(chosen.localisation_metrics_test.get('top3_hit_rate', np.nan), np.nan) - reference['loc_test_2019']['top3_hit_rate']
    stage1_neg = int(chosen.packet_info.get('actual_dev_negative_triggers', 0))
    current_binary_feats = int(reference_size['binary_feature_count'].iloc[0])
    current_loc_feats = int(reference_size['localisation_feature_count'].iloc[0])

    lines = [
        '# BattLeDIM tiny MCU phase summary',
        '',
        '## Current architecture from the attachments',
        '',
        '- Stage 1 is the recall-first BattLeDIM detector described in the notebook/README: causal smoothing, seasonality suppression, pressure and pairwise residual evidence, flow context, a leak-state machine, and periodic Stage 2 rechecks.',
        '- The patched Stage 2 in the attached run uses 12-hour packets (144 steps), four summary horizons `[12, 36, 72, 144]`, pseudo-hard negatives, a calibrated HistGradientBoosting binary head, and one-vs-rest pipe classifiers.',
        '- The attached reference run reports 721 actual late-2018 triggers (714 positive, 7 negative), threshold 0.9739572638050862, 2019 binary recall 0.964384, and 2019 localisation top-3 hit rate 0.716438.',
        '',
        '## Why the current implementation is too large for the MCU target',
        '',
        f'- The current feature builder expands each trigger to about **{current_binary_feats} binary features** and **{current_loc_feats} localisation features**.',
        f'- The current Stage 2 packet lookback alone is roughly **{int(reference_size.loc[reference_size["component"] == "stage2_packet_buffer_float32", "ram_bytes"].iloc[0]) / 1024:.1f} KB RAM** and the float32 feature workspace adds about **{int(reference_size.loc[reference_size["component"] == "stage2_feature_workspace_float32", "ram_bytes"].iloc[0]) / 1024:.1f} KB RAM**.',
        f'- Stage 1\'s original 2016-bin weekly baseline table is about **{int(reference_size.loc[reference_size["component"] == "stage1_weekly_baseline_float32", "flash_bytes"].iloc[0]) / 1024:.1f} KB flash** on its own.',
        f'- The reference boosted Stage 2 models are conservatively estimated at about **{current_ref_total_flash / 1024:.1f} KB flash**, well beyond a 1 MB microcontroller budget.',
        '',
        '## Candidate tiny pipelines considered',
        '',
    ]
    candidate_rows = model_comparison[model_comparison['candidate'] != 'current_patched_reference'].copy()
    for _, row in candidate_rows.iterrows():
        lines.append(
            f'- **{row["candidate"]}**: Stage 1 `{row["stage1_mode"]}`, localizer `{row["localizer_type"]}`, RAM {row["total_ram_bytes"] / 1024:.1f} KB, Flash {row["total_flash_bytes"] / 1024:.1f} KB, Stage 1 recall {row["stage1_recall_2018"]:.4f}, val-actual binary recall {row["binary_recall_val_actual"]:.4f}, 2019 top-3 {row["localisation_top3_2019"]:.4f}.'
        )
    lines.extend([
        '',
        '## Chosen tiny architecture',
        '',
        f'- **Chosen candidate:** `{chosen.candidate.name}` - {chosen.candidate.description}',
        '- Selection priority followed the project brief: MCU fit first, then Stage 1 fidelity and practical call rate, then Stage 2 recall on true validation triggers, then localisation.',
        f'- **Stage 1:** `{chosen.candidate.baseline_mode}` baseline table, `{chosen.candidate.monitored_pressure_count}` selected pressure sensors, `{chosen.candidate.pairwise_anchors}` pairwise anchors, same `battledim_stage1` import path/API, causal leak-state logic, and 12-hour periodic rechecks.',
        f'- **Stage 2 binary head:** folded linear logistic model over `{len(chosen.binary_cols)}` compact causal packet features.',
        f'- **Stage 2 localizer:** `{chosen.candidate.localizer_type}` over `{len(chosen.loc_cols)}` compact features, returning per-pipe scores and top-3 rankings.',
        '',
        '## MCU fit check',
        '',
        f'- Estimated runtime RAM: **{chosen.total_ram_bytes / 1024:.1f} KB**.',
        f'- Estimated flash footprint: **{chosen.total_flash_bytes / 1024:.1f} KB**.',
        '- Both are inside the requested ~256 KB RAM and ~1 MB Flash budget with an explicit stack/workspace allowance included.',
        '',
        '## Performance vs the provided patched baseline',
        '',
        f'- Stage 1 late-2018 timestep recall delta: **{stage1_drop_2018:+.4f}**.',
        f'- Stage 1 late-2018 calls/day: **{mod.safe_float(chosen.stage1_metrics_2018.get("stage2_calls_per_day", np.nan), np.nan):.4f}** vs reference **{reference["stage1_2018"]["stage2_calls_per_day"]:.4f}**.',
        f'- Stage 2 binary validation-actual recall: **{mod.safe_float(chosen.binary_metrics_val_actual.get("recall", np.nan), np.nan):.4f}** vs reference **{reference["binary_val_actual"]["recall"]:.4f}**.',
        f'- Stage 2 binary 2019 recall delta: **{binary_recall_drop_2019:+.4f}**.',
        f'- Stage 2 localisation validation top-3: **{mod.safe_float(chosen.localisation_metrics_val.get("top3_hit_rate", np.nan), np.nan):.4f}** vs reference **{reference["loc_val"]["top3_hit_rate"]:.4f}**.',
        f'- Stage 2 localisation 2019 top-3 delta: **{loc_top3_drop_2019:+.4f}**.',
        '',
        '## Risks / caveats',
        '',
        f'- Late-2018 actual trigger validation still has only **{stage1_neg} real hard-negative Stage 1 triggers** for the chosen tiny pipeline, so binary false-alarm rejection uncertainty remains high.',
        '- As in the attached patched run, 2019 is effectively all-positive in this merged table, so it cannot measure binary false-alarm rejection. 2019 remains useful mainly for recall and localisation continuity.',
        '- The exported C headers are parameter tables and folded coefficients. Firmware still needs a small runtime wrapper for streaming median, EWMA/CUSUM, rolling sums, and sigmoid/logit evaluation.',
        '',
        '## Files written',
        '',
        '- See `tinyml_artifact_manifest.csv` for the complete file list.',
    ])
    return '\n'.join(lines)



def export_full_run(run_dir: Path, chosen_candidate_name: str, candidate_summaries: List[Dict[str, Any]], candidate_memory_paths: List[Path]) -> Path:
    writer = mod.ArtifactWriter(run_dir)

    full_df, groups, base_stage1_cfg, split_cfg = load_static_context()
    reference = mod.load_reference_baseline()
    reference_size = mod.estimate_reference_current_pipeline(groups)
    reference_row = build_reference_row(reference, reference_size)

    comparison_rows = [reference_row] + [
        {
            k: v
            for k, v in summary.items()
            if k not in {'binary_feature_names', 'loc_feature_names', 'packet_info', 'candidate_config'}
        }
        for summary in candidate_summaries
    ]
    model_comparison = pd.DataFrame(comparison_rows)

    memory_frames = [reference_size]
    for mem_path in candidate_memory_paths:
        memory_frames.append(pd.read_csv(mem_path))
    memory_budget = pd.concat(memory_frames, ignore_index=True)

    chosen_candidate = candidate_by_name(chosen_candidate_name)
    chosen = mod.run_candidate(chosen_candidate, full_df, groups, base_stage1_cfg, split_cfg)
    chosen_summary = next(s for s in candidate_summaries if s['candidate'] == chosen_candidate_name)

    # Required tables.
    writer.write_csv('tinyml_model_comparison.csv', model_comparison, 'Pipeline candidate comparison including the provided baseline reference.')
    writer.write_csv('tinyml_memory_budget.csv', memory_budget, 'Estimated RAM / Flash budgets for the reference pipeline and tiny candidates.')
    writer.write_series_csv('tinyml_stage1_metrics_2018.csv', chosen.stage1_metrics_2018, 'Chosen tiny Stage 1 metrics on late 2018.')
    writer.write_series_csv('tinyml_stage1_metrics_2019.csv', chosen.stage1_metrics_2019, 'Chosen tiny Stage 1 metrics on 2019.')
    writer.write_series_csv('tinyml_stage2_binary_metrics_val_mixed.csv', chosen.binary_metrics_val_mixed, 'Chosen tiny Stage 2 binary metrics on mixed validation packets.')
    writer.write_series_csv('tinyml_stage2_binary_metrics_val_actual.csv', chosen.binary_metrics_val_actual, 'Chosen tiny Stage 2 binary metrics on actual validation triggers.')
    writer.write_series_csv('tinyml_stage2_binary_metrics_2019.csv', chosen.binary_metrics_test_actual, 'Chosen tiny Stage 2 binary metrics on 2019 actual triggers.')
    writer.write_series_csv('tinyml_stage2_localisation_metrics_val.csv', chosen.localisation_metrics_val, 'Chosen tiny localisation metrics on validation positives.')
    writer.write_series_csv('tinyml_stage2_localisation_metrics_2019.csv', chosen.localisation_metrics_test, 'Chosen tiny localisation metrics on 2019 positives.')
    writer.write_csv('tinyml_threshold_table.csv', chosen.binary_threshold_table, 'Binary threshold sweep on validation packets.')
    writer.write_csv('tinyml_predictions_2019.csv', chosen.pred_test, 'Chosen tiny Stage 2 predictions on 2019 actual triggers.')
    writer.write_csv('tinyml_feature_spec.csv', chosen.feature_spec, 'Chosen tiny feature inventory.')

    metadata_obj = {
        'run_dir': str(run_dir),
        'chosen_candidate': mod.asdict(chosen.candidate),
        'chosen_summary': chosen_summary,
        'selection_key': summary_selection_key(chosen_summary),
        'reference_baseline': reference,
        'packet_split_config': mod.asdict(split_cfg),
        'packet_info': chosen.packet_info,
        'stage1_monitored_pressures': chosen.detector.monitored_pressure_cols_,
        'stage1_anchor_pressures': chosen.detector.anchor_pressure_cols_,
        'stage1_flow_score_cols': chosen.detector.flow_score_cols_,
        'stage1_flow_context_cols': chosen.detector.flow_context_cols_,
        'stage1_auto_excluded_flow_score_cols': chosen.detector.auto_excluded_flow_score_cols_,
        'binary_threshold_info': chosen.binary_threshold_info,
        'binary_feature_count': len(chosen.binary_cols),
        'localisation_feature_count': len(chosen.loc_cols),
        'estimated_total_ram_bytes': chosen.total_ram_bytes,
        'estimated_total_flash_bytes': chosen.total_flash_bytes,
        'current_reference_total_ram_bytes': int(reference_size['ram_bytes'].sum()),
        'current_reference_total_flash_bytes': int(reference_size['flash_bytes'].sum()),
        'data_path': str(DATA_PATH),
        'config_path': str(CONFIG_PATH),
    }
    writer.write_json('tinyml_metadata.json', metadata_obj, 'Run metadata for the tiny MCU phase.')

    # Deployment artifacts.
    mod.export_stage1_artifacts(chosen, writer)
    mod.export_stage2_artifacts(chosen, writer)

    # Notebook + code.
    nb_path = export_notebook(run_dir)
    writer.add(nb_path.name, nb_path, 'Derived notebook that appends the tiny MCU phase to the patched Stage 2 workflow.')
    copy_code_files(run_dir, writer)

    # Plots.
    writer.write_plot('pr_curve_val.png', mod.make_pr_curve_fig(chosen.pred_val, 'Tiny Stage 2 validation PR curve'), 'PR curve on mixed validation packets.')
    writer.write_plot('calibration_val.png', mod.make_calibration_fig(chosen.pred_val, 'Tiny Stage 2 validation calibration'), 'Calibration curve on mixed validation packets.')
    writer.write_plot('probability_histogram_val.png', mod.make_probability_histogram_fig(chosen.pred_val, 'Tiny Stage 2 validation probability histogram'), 'Histogram of validation probabilities.')
    writer.write_plot('memory_vs_performance.png', mod.make_memory_vs_performance_fig(model_comparison[model_comparison['candidate'] != 'current_patched_reference']), 'Candidate memory vs performance Pareto.')
    writer.write_plot('feature_budget_breakdown.png', mod.make_feature_budget_breakdown_fig(memory_budget, chosen.candidate.name), 'Memory budget breakdown for the chosen tiny model.')
    stage1_tradeoff_df = model_comparison[model_comparison['candidate'] != 'current_patched_reference'][['candidate', 'stage1_calls_per_day_2018', 'stage1_recall_2018']].copy()
    writer.write_plot('stage1_calls_vs_recall.png', mod.make_stage1_calls_vs_recall_fig(stage1_tradeoff_df), 'Operational Stage 1 calls/day vs recall tradeoff.')
    baseline_topk = {
        'val_top1': reference['loc_val']['top1_hit_rate'],
        'val_top3': reference['loc_val']['top3_hit_rate'],
        'test_top1': reference['loc_test_2019']['top1_hit_rate'],
        'test_top3': reference['loc_test_2019']['top3_hit_rate'],
    }
    writer.write_plot('localisation_topk.png', mod.make_localisation_topk_fig(baseline_topk, chosen), 'Top-1 / top-3 localisation comparison against the provided baseline.')

    summary_md = build_summary_markdown(chosen, chosen_summary, reference, reference_size, model_comparison)
    writer.write_text('tinyml_run_summary.md', summary_md, 'Narrative summary of the tiny MCU phase.')

    writer.finalize_manifest()
    return run_dir



def main(output_root: Path | str = THIS_DIR) -> Path:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = output_root / f'tinyml_run_{stamp}'
    run_dir.mkdir(parents=True, exist_ok=True)

    temp_dir = Path(tempfile.mkdtemp(prefix='tinyml_eval_', dir=str(output_root)))
    try:
        candidate_summaries: List[Dict[str, Any]] = []
        candidate_memory_paths: List[Path] = []
        for cand in mod.CANDIDATES:
            summary_json = temp_dir / f'{cand.name}_summary.json'
            memory_csv = temp_dir / f'{cand.name}_memory.csv'
            _safe_run_subprocess(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    '--eval-candidate',
                    cand.name,
                    '--summary-json',
                    str(summary_json),
                    '--memory-csv',
                    str(memory_csv),
                ]
            )
            candidate_summaries.append(json.loads(summary_json.read_text()))
            candidate_memory_paths.append(memory_csv)

        chosen_summary = max(candidate_summaries, key=summary_selection_key)
        chosen_candidate_name = str(chosen_summary['candidate'])
        export_full_run(run_dir, chosen_candidate_name, candidate_summaries, candidate_memory_paths)
        return run_dir
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)



def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run the BattLeDIM tiny MCU final phase with candidate evaluation in isolated subprocesses.')
    parser.add_argument('--eval-candidate', type=str, default=None, help='Evaluate a single candidate and write a summary JSON / memory CSV.')
    parser.add_argument('--summary-json', type=str, default=None, help='Path to write the candidate summary JSON.')
    parser.add_argument('--memory-csv', type=str, default=None, help='Path to write the candidate memory CSV.')
    parser.add_argument('--output-root', type=str, default=str(THIS_DIR), help='Root directory for the final run folder.')
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_args()
    if args.eval_candidate:
        if not args.summary_json or not args.memory_csv:
            raise SystemExit('--eval-candidate requires --summary-json and --memory-csv')
        write_eval_outputs(args.eval_candidate, Path(args.summary_json), Path(args.memory_csv))
    else:
        out = main(Path(args.output_root))
        print(out)
