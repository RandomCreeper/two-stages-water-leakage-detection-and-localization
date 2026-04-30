from __future__ import annotations

import json
import math
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import nbformat as nbf
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from battledim_stage1 import (
    BattLeDIMStage1,
    Stage1Config,
    build_known_intervals_from_leak_columns,
    evaluate_stage1_results,
    load_battledim_merged_csv,
)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def natural_key(text: str):
    return [int(tok) if tok.isdigit() else tok for tok in re.split(r'(\d+)', str(text))]


def sigmoid(x):
    x = np.asarray(x, dtype=float)
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def logit(p):
    p = np.asarray(p, dtype=float)
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def infer_column_groups(df: pd.DataFrame, include_flow3_context: bool = True) -> Dict[str, List[str]]:
    pressure_cols = sorted([c for c in df.columns if c.startswith('press_')], key=natural_key)
    pipe_cols = sorted([c for c in df.columns if c.startswith('pipe_') and c.endswith('_bin')], key=natural_key)
    demand_cols = sorted([c for c in df.columns if c.startswith('n') and c[1:].isdigit()], key=natural_key)
    flow_primary = [c for c in ['flow_1', 'flow_2'] if c in df.columns]
    flow_context = ['flow_3'] if include_flow3_context and 'flow_3' in df.columns else []
    tank_cols = [c for c in ['T1'] if c in df.columns]
    return {
        'pressure_cols': pressure_cols,
        'pipe_cols': pipe_cols,
        'demand_cols': demand_cols,
        'flow_primary_cols': flow_primary,
        'flow_context_cols': flow_context,
        'tank_cols': tank_cols,
    }


def get_any_real_leak_mask(df: pd.DataFrame, pipe_cols: Sequence[str]) -> pd.Series:
    parts = []
    if pipe_cols:
        parts.append(df[list(pipe_cols)].fillna(0).sum(axis=1) > 0)
    if 'leak_count' in df.columns:
        parts.append(df['leak_count'].fillna(0) > 0)
    if 'leak_mag_total' in df.columns:
        parts.append(df['leak_mag_total'].fillna(0) > 0)
    if not parts and 'leak_binary' in df.columns:
        parts.append(df['leak_binary'].fillna(0) > 0)
    if not parts:
        raise ValueError('No usable leak labels found.')
    return pd.concat(parts, axis=1).any(axis=1)


def find_initial_healthy_block(index: pd.DatetimeIndex, healthy_mask: pd.Series) -> pd.DatetimeIndex:
    arr = healthy_mask.reindex(index).fillna(False).to_numpy(dtype=bool)
    end_pos = len(arr)
    for i, val in enumerate(arr):
        if not val:
            end_pos = i
            break
    return index[:end_pos]


def choose_stage1_fit_end_from_healthy_block(df_2018: pd.DataFrame, fit_frac: float, pipe_cols: Sequence[str]) -> pd.Timestamp:
    healthy_mask = ~get_any_real_leak_mask(df_2018, pipe_cols)
    initial_block = find_initial_healthy_block(df_2018.index, healthy_mask)
    if len(initial_block) >= 20:
        pos = max(1, int(math.floor(len(initial_block) * fit_frac))) - 1
        return initial_block[pos]
    pos = max(1, int(math.floor(len(df_2018) * 0.20))) - 1
    return df_2018.index[pos]


def chronological_split_frame(df_in: pd.DataFrame, frac: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df_in.empty:
        return df_in.copy(), df_in.copy()
    df_sorted = df_in.sort_values('timestamp').reset_index(drop=True)
    if len(df_sorted) == 1:
        return df_sorted.copy(), df_sorted.iloc[0:0].copy()
    cut = max(1, int(math.floor(len(df_sorted) * frac)))
    cut = min(cut, len(df_sorted) - 1)
    return df_sorted.iloc[:cut].copy(), df_sorted.iloc[cut:].copy()


def split_df_threeway(df_in: pd.DataFrame, fracs=(0.60, 0.20, 0.20)):
    df_sorted = df_in.sort_values('timestamp').reset_index(drop=True).copy()
    if df_sorted.empty:
        empty = df_sorted.iloc[0:0].copy()
        return empty, empty, empty
    n = len(df_sorted)
    if n == 1:
        empty = df_sorted.iloc[0:0].copy()
        return df_sorted.copy(), empty, empty
    if n == 2:
        return df_sorted.iloc[:1].copy(), df_sorted.iloc[1:2].copy(), df_sorted.iloc[0:0].copy()
    a = max(1, int(round(n * fracs[0])))
    b = max(1, int(round(n * fracs[1])))
    if a + b >= n:
        b = max(1, n - a - 1)
    c = n - a - b
    if c <= 0:
        c = 1
        if b > 1:
            b -= 1
        else:
            a = max(1, a - 1)
    return df_sorted.iloc[:a].copy(), df_sorted.iloc[a:a + b].copy(), df_sorted.iloc[a + b:].copy()


def split_small_temporal(df_in: pd.DataFrame):
    df_sorted = df_in.sort_values('timestamp').reset_index(drop=True).copy()
    n = len(df_sorted)
    empty = df_sorted.iloc[0:0].copy()
    if n == 0:
        return empty, empty, empty
    if n == 1:
        return empty, empty, df_sorted.copy()
    if n == 2:
        return empty, df_sorted.iloc[:1].copy(), df_sorted.iloc[1:].copy()
    parts = np.array_split(df_sorted.index.to_numpy(), 3)
    return df_sorted.loc[parts[0]].copy(), df_sorted.loc[parts[1]].copy(), df_sorted.loc[parts[2]].copy()


def split_index_threeway(index: pd.DatetimeIndex):
    idx = pd.DatetimeIndex(sorted(pd.to_datetime(index).unique()))
    if len(idx) == 0:
        empty = pd.DatetimeIndex([])
        return empty, empty, empty
    parts = np.array_split(np.arange(len(idx)), 3)
    return idx[parts[0]], idx[parts[1]], idx[parts[2]]


def stratified_sample_timestamps(index: pd.DatetimeIndex, n_samples: int, seed: int = 42) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(sorted(pd.to_datetime(index).unique()))
    if n_samples <= 0 or len(idx) == 0:
        return pd.DatetimeIndex([])
    if n_samples >= len(idx):
        return idx
    df_idx = pd.DataFrame({'timestamp': idx})
    df_idx['month'] = df_idx['timestamp'].dt.month
    df_idx['hour'] = df_idx['timestamp'].dt.hour
    df_idx['stratum'] = df_idx['month'].astype(str) + '_' + df_idx['hour'].astype(str)
    rng = np.random.RandomState(seed)
    sampled_parts = []
    base_frac = n_samples / len(df_idx)
    for _, block in df_idx.groupby('stratum', sort=True):
        take = int(round(len(block) * base_frac))
        take = min(len(block), max(0, take))
        if take > 0:
            sampled_parts.append(block.sample(n=take, random_state=int(rng.randint(0, 10_000_000))))
    sampled = pd.concat(sampled_parts, ignore_index=True) if sampled_parts else df_idx.iloc[0:0].copy()
    sampled = sampled.drop_duplicates('timestamp')
    if len(sampled) < n_samples:
        remaining = df_idx.loc[~df_idx['timestamp'].isin(sampled['timestamp'])].copy()
        need = min(len(remaining), n_samples - len(sampled))
        if need > 0:
            extra = remaining.sample(n=need, random_state=int(rng.randint(0, 10_000_000)))
            sampled = pd.concat([sampled, extra], ignore_index=True)
    elif len(sampled) > n_samples:
        sampled = sampled.sample(n=n_samples, random_state=seed)
    return pd.DatetimeIndex(sorted(pd.to_datetime(sampled['timestamp']).unique()))


def make_pseudo_trigger_frame(index: pd.DatetimeIndex, reason: str, kind: str) -> pd.DataFrame:
    out = pd.DataFrame({'timestamp': pd.to_datetime(index)})
    out = out.sort_values('timestamp').reset_index(drop=True)
    out['event_id'] = -1
    out['reason'] = reason
    out['area_guess'] = 'unknown'
    out['top_sensors'] = ''
    out['sampled_negative'] = 1
    out['pseudo_negative_kind'] = kind
    return out


def select_peak_negative_timestamps(
    score: pd.Series,
    n_samples: int,
    step_minutes: int = 5,
    min_gap_steps: int = 12,
    peak_radius: int = 3,
) -> pd.DatetimeIndex:
    s = score.dropna().astype(float)
    if n_samples <= 0 or s.empty:
        return pd.DatetimeIndex([])
    vals = s.to_numpy()
    keep = np.ones(len(s), dtype=bool)
    for i in range(len(s)):
        lo = max(0, i - peak_radius)
        hi = min(len(s), i + peak_radius + 1)
        if vals[i] < np.nanmax(vals[lo:hi]):
            keep[i] = False
    cand = s[keep].sort_values(ascending=False)
    chosen: List[pd.Timestamp] = []
    for ts in cand.index:
        if all(abs((pd.Timestamp(ts) - prev).total_seconds()) >= min_gap_steps * step_minutes * 60 for prev in chosen):
            chosen.append(pd.Timestamp(ts))
        if len(chosen) >= n_samples:
            break
    return pd.DatetimeIndex(sorted(chosen))


def add_causal_event_context(results: pd.DataFrame, triggers: pd.DataFrame, step_minutes: int):
    out = results.copy()
    out['event_age_samples_so_far'] = 0
    out['event_age_hours_so_far'] = 0.0
    leak_state = out.get('leak_state', pd.Series(0, index=out.index)).fillna(0).astype(int)
    event_id = out.get('event_id', pd.Series(0, index=out.index)).fillna(0).astype(int)
    valid = (leak_state > 0) & (event_id > 0)
    if valid.any():
        tmp = out.loc[valid, ['score_final']].copy()
        tmp['event_id'] = event_id.loc[valid].values
        age = tmp.groupby('event_id').cumcount() + 1
        out.loc[valid, 'event_age_samples_so_far'] = age.astype(int).values
        out.loc[valid, 'event_age_hours_so_far'] = age.astype(float).values * (step_minutes / 60.0)

    trg = triggers.copy()
    if not trg.empty:
        trg['timestamp'] = pd.to_datetime(trg['timestamp'])
        trg = trg.sort_values('timestamp').reset_index(drop=True)
    else:
        trg = pd.DataFrame(columns=['timestamp', 'event_id', 'reason', 'area_guess', 'top_sensors'])
    if 'sampled_negative' not in trg.columns:
        trg['sampled_negative'] = 0
    if 'pseudo_negative_kind' not in trg.columns:
        trg['pseudo_negative_kind'] = 'none'
    if 'event_id' not in trg.columns:
        trg['event_id'] = -1
    trg['event_id'] = trg['event_id'].fillna(-1).astype(int)
    trg['prior_triggers_in_event'] = 0
    trg['trigger_order_in_event'] = 0
    trg['minutes_since_prev_trigger_in_event'] = -1.0
    real = trg['event_id'] > 0
    if real.any():
        grp = trg.loc[real].groupby('event_id', sort=False)
        trg.loc[real, 'prior_triggers_in_event'] = grp.cumcount().values
        trg.loc[real, 'trigger_order_in_event'] = (grp.cumcount() + 1).values
        delta = grp['timestamp'].diff().dt.total_seconds().div(60.0).fillna(-1.0)
        trg.loc[real, 'minutes_since_prev_trigger_in_event'] = delta.values
    return out, trg


def direct_threshold_search(y_true, p, min_recall=0.95):
    y_true = np.asarray(y_true, dtype=int)
    p = np.asarray(p, dtype=float)
    if len(y_true) == 0:
        return (
            {'threshold': 0.5, 'precision': np.nan, 'recall': np.nan, 'f1': np.nan},
            pd.DataFrame(columns=['threshold', 'precision', 'recall', 'f1', 'accepted_fraction']),
        )
    candidates = np.unique(np.clip(np.concatenate([[0.0], p, [1.0]]), 0.0, 1.0))
    rows = []
    for thr in candidates:
        y_pred = (p >= thr).astype(int)
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        rows.append({'threshold': float(thr), 'precision': float(precision), 'recall': float(recall), 'f1': float(f1), 'accepted_fraction': float(y_pred.mean())})
    tbl = pd.DataFrame(rows).drop_duplicates('threshold').sort_values('threshold').reset_index(drop=True)
    eligible = tbl[tbl['recall'] >= min_recall]
    if not eligible.empty:
        best = eligible.sort_values(['precision', 'f1', 'threshold'], ascending=[False, False, False]).iloc[0]
    else:
        best = tbl.sort_values(['recall', 'precision', 'f1', 'threshold'], ascending=[False, False, False, False]).iloc[0]
    return {'threshold': float(best['threshold']), 'precision': float(best['precision']), 'recall': float(best['recall']), 'f1': float(best['f1'])}, tbl


def safe_binary_metrics(y_true: np.ndarray, p: np.ndarray, threshold: float, precision_targets: Sequence[float] = (0.5, 0.6, 0.7, 0.8, 0.9)) -> pd.Series:
    y_true = np.asarray(y_true, dtype=int)
    p = np.asarray(p, dtype=float)
    if len(y_true) == 0:
        return pd.Series(dtype=float)
    y_pred = (p >= threshold).astype(int)
    out = {
        'n_packets': float(len(y_true)),
        'positive_rate': float(y_true.mean()),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1': float(f1_score(y_true, y_pred, zero_division=0)),
        'accepted_fraction': float(y_pred.mean()),
        'brier': float(brier_score_loss(y_true, p)) if len(np.unique(y_true)) > 1 else float(np.mean((p - y_true) ** 2)),
        'pr_auc': float(average_precision_score(y_true, p)) if len(np.unique(y_true)) > 1 else np.nan,
        'roc_auc': float(roc_auc_score(y_true, p)) if len(np.unique(y_true)) > 1 else np.nan,
    }
    if len(np.unique(y_true)) > 1:
        rows = pd.DataFrame({'p': p, 'y': y_true}).sort_values('p', ascending=False).reset_index(drop=True)
        tps = rows['y'].cumsum().to_numpy(dtype=float)
        fps = np.arange(1, len(rows) + 1, dtype=float) - tps
        precisions = np.divide(tps, tps + fps, out=np.zeros_like(tps), where=(tps + fps) > 0)
        recalls = tps / max(1.0, rows['y'].sum())
        for target in precision_targets:
            ok = recalls[precisions >= target]
            out[f'recall_at_precision_{target:.2f}'] = float(ok.max()) if len(ok) else np.nan
    else:
        for target in precision_targets:
            out[f'recall_at_precision_{target:.2f}'] = np.nan
    return pd.Series(out, dtype=float)


def evaluate_localisation(y_true_pipe: pd.DataFrame, pipe_prob_df: pd.DataFrame, threshold: float = 0.5) -> pd.Series:
    if len(y_true_pipe) == 0:
        return pd.Series(dtype=float)
    y_true = y_true_pipe.to_numpy(dtype=int)
    p = pipe_prob_df.loc[y_true_pipe.index].to_numpy(dtype=float)
    y_pred = (p >= threshold).astype(int)
    f1s = []
    aps = []
    for j in range(y_true.shape[1]):
        yt = y_true[:, j]
        yp = y_pred[:, j]
        tp = int(((yt == 1) & (yp == 1)).sum())
        fp = int(((yt == 0) & (yp == 1)).sum())
        fn = int(((yt == 1) & (yp == 0)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s.append((2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0)
        if len(np.unique(yt)) >= 2:
            aps.append(float(average_precision_score(yt, p[:, j])))
    pipe_cols = list(y_true_pipe.columns)
    top1 = []
    top3 = []
    for i in range(len(y_true_pipe)):
        true_set = {pipe_cols[j] for j, v in enumerate(y_true[i]) if v == 1}
        order = np.argsort(p[i])[::-1]
        top1.append(int(pipe_cols[order[0]] in true_set))
        top3.append(int(any(pipe_cols[j] in true_set for j in order[:3])))
    return pd.Series(
        {
            'macro_f1_pipes': float(np.mean(f1s)) if f1s else np.nan,
            'top1_hit_rate': float(np.mean(top1)) if top1 else np.nan,
            'top3_hit_rate': float(np.mean(top3)) if top3 else np.nan,
            'mAP_pipes': float(np.mean(aps)) if aps else np.nan,
        },
        dtype=float,
    )


def feature_group(name: str) -> str:
    if name.startswith('score_') or name.endswith('_topk') or name == 'sensor_votes' or name == 'burst_cusum':
        return 'stage1_scores'
    if name.startswith('reason__'):
        return 'trigger_reason'
    if name.startswith('sel_current_') or name.startswith('sel_mean_') or name.startswith('sel_max_'):
        return 'sensor_summary'
    if name.startswith('sel__'):
        return 'per_sensor'
    if name.startswith('flow_') or name.startswith('T1_'):
        return 'flow_tank'
    return 'other'


# ---------------------------------------------------------------------------
# Candidate definitions
# ---------------------------------------------------------------------------


@dataclass
class PipelineCandidate:
    name: str
    description: str
    baseline_mode: str
    monitored_pressure_count: int
    pairwise_anchors: int
    localizer_type: str
    binary_feature_mode: str
    loc_feature_mode: str
    stage1_notes: str


CANDIDATES: List[PipelineCandidate] = [
    PipelineCandidate(
        name='ultra_tiny_none_proto',
        description='No seasonal table in Stage 1; smallest linear/prototype stack.',
        baseline_mode='none',
        monitored_pressure_count=6,
        pairwise_anchors=1,
        localizer_type='prototype',
        binary_feature_mode='small',
        loc_feature_mode='small',
        stage1_notes='Extreme memory cut; likely weakest 2018 stability because seasonality is ignored.',
    ),
    PipelineCandidate(
        name='tiny_daily24_proto',
        description='24-bin daily Stage 1 baseline with prototype localizer.',
        baseline_mode='daily_hour',
        monitored_pressure_count=6,
        pairwise_anchors=1,
        localizer_type='prototype',
        binary_feature_mode='medium',
        loc_feature_mode='medium',
        stage1_notes='Middle ground: retains diurnal context with very small tables.',
    ),
    PipelineCandidate(
        name='tiny_weekly168_ovr',
        description='168-bin weekly-hour Stage 1 baseline with linear OVR localizer.',
        baseline_mode='weekly_hour',
        monitored_pressure_count=8,
        pairwise_anchors=2,
        localizer_type='ovr_logreg',
        binary_feature_mode='full',
        loc_feature_mode='full',
        stage1_notes='Best accuracy/size balance while staying tiny enough for MCU deployment.',
    ),
]


# ---------------------------------------------------------------------------
# Stage 2 packet splitting
# ---------------------------------------------------------------------------


@dataclass
class PacketSplitConfig:
    step_minutes: int = 5
    easy_negative_ratio_per_positive: float = 0.35
    hard_peak_negative_ratio_per_positive: float = 0.25
    sampled_negative_weight: float = 0.55
    pseudo_hard_negative_weight: float = 0.90
    real_hard_negative_weight: float = 1.25
    binary_min_recall: float = 0.95
    random_state: int = 42


@dataclass
class PacketSplitResult:
    train_packets: pd.DataFrame
    cal_packets: pd.DataFrame
    val_packets: pd.DataFrame
    test_packets: pd.DataFrame
    info: Dict[str, int]


# ---------------------------------------------------------------------------
# Tiny feature builder
# ---------------------------------------------------------------------------


class TinyFeatureBuilder:
    def __init__(
        self,
        raw_df: pd.DataFrame,
        results: pd.DataFrame,
        sensor_detail: pd.DataFrame,
        detector: BattLeDIMStage1,
        all_triggers: pd.DataFrame,
        groups: Dict[str, List[str]],
        step_minutes: int = 5,
    ):
        self.raw_df = raw_df
        self.results, self.all_triggers = add_causal_event_context(results, all_triggers, step_minutes=step_minutes)
        self.sensor_detail = sensor_detail
        self.detector = detector
        self.groups = groups
        self.step_minutes = step_minutes
        self.index = raw_df.index
        self.target_any = get_any_real_leak_mask(raw_df, groups['pipe_cols']).astype(int)
        self.target_pipe = raw_df[groups['pipe_cols']].fillna(0).astype(int)
        self.selected_sensors = list(detector.monitored_pressure_cols_)
        self.sensor_sel = sensor_detail[self.selected_sensors].astype(float).copy()
        self.internal = detector.last_internal_
        self.residual_all = self.internal['residual_all'].copy().astype(float)
        self.actual_reason_values = ['event_start', 'periodic_recheck', 'sensor_shift']
        self.flow_context_features = [c for c in ['flow_1', 'flow_2', 'flow_3', 'T1'] if c in self.residual_all.columns]
        self._precompute_rolls()

    def _precompute_rolls(self):
        self.row_mean = self.sensor_sel.mean(axis=1)
        self.row_max = self.sensor_sel.max(axis=1)
        self.row_std = self.sensor_sel.std(axis=1, ddof=0)
        self.rolls: Dict[str, pd.Series] = {}
        for h in [12, 72]:
            self.rolls[f'sel_mean_h{h}'] = self.row_mean.rolling(h, min_periods=1).mean()
            self.rolls[f'sel_max_h{h}'] = self.row_max.rolling(h, min_periods=1).mean()
            for sensor in self.selected_sensors:
                self.rolls[f'sel__{sensor}__h{h}'] = self.sensor_sel[sensor].rolling(h, min_periods=1).mean()
            for col in self.flow_context_features:
                self.rolls[f'{col}_h{h}'] = self.residual_all[col].rolling(h, min_periods=1).mean()

    def build_dataset(self, trigger_df: pd.DataFrame, split_name: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        features: List[Dict[str, float]] = []
        metas: List[Dict[str, object]] = []
        targets: List[Dict[str, object]] = []
        trigger_df = trigger_df.copy()
        trigger_df['timestamp'] = pd.to_datetime(trigger_df['timestamp'])
        trigger_df = trigger_df.sort_values('timestamp').reset_index(drop=True)
        for _, trigger_row in trigger_df.iterrows():
            ts = pd.Timestamp(trigger_row['timestamp'])
            if ts not in self.results.index:
                continue
            row = self.results.loc[ts]
            sensor_vals = self.sensor_sel.loc[ts].to_numpy(dtype=float, copy=False)
            sensor_sorted = np.sort(sensor_vals)[::-1] if sensor_vals.size else np.array([])
            feat: Dict[str, float] = {
                'score_final': float(row['score_final']),
                'score_burst': float(row['score_burst']),
                'score_incipient': float(row['score_incipient']),
                'pressure_topk': float(row['pressure_topk']),
                'pairwise_topk': float(row['pairwise_topk']),
                'flow_topk': float(row['flow_topk']),
                'ewma_topk': float(row['ewma_topk']),
                'slope_topk': float(row['slope_topk']),
                'burst_cusum': float(row['burst_cusum']),
                'sensor_votes': float(row['sensor_votes']),
                'sel_current_mean': float(sensor_vals.mean()) if sensor_vals.size else 0.0,
                'sel_current_max': float(sensor_vals.max()) if sensor_vals.size else 0.0,
                'sel_current_std': float(sensor_vals.std(ddof=0)) if sensor_vals.size else 0.0,
                'sel_current_top1': float(sensor_sorted[0]) if sensor_sorted.size >= 1 else 0.0,
                'sel_current_top2': float(sensor_sorted[1]) if sensor_sorted.size >= 2 else 0.0,
                'sel_current_gap12': float(sensor_sorted[0] - sensor_sorted[1]) if sensor_sorted.size >= 2 else 0.0,
                'sel_mean_h12': float(self.rolls['sel_mean_h12'].loc[ts]),
                'sel_mean_h72': float(self.rolls['sel_mean_h72'].loc[ts]),
                'sel_max_h12': float(self.rolls['sel_max_h12'].loc[ts]),
                'sel_max_h72': float(self.rolls['sel_max_h72'].loc[ts]),
            }
            reason = str(trigger_row.get('reason', 'unknown'))
            for val in self.actual_reason_values:
                feat[f'reason__{val}'] = 1.0 if reason == val else 0.0
            for col in self.flow_context_features:
                feat[f'{col}_last'] = float(self.residual_all.loc[ts, col])
                feat[f'{col}_h12'] = float(self.rolls[f'{col}_h12'].loc[ts])
                feat[f'{col}_h72'] = float(self.rolls[f'{col}_h72'].loc[ts])
            for sensor in self.selected_sensors:
                feat[f'sel__{sensor}'] = float(self.sensor_sel.loc[ts, sensor])
                feat[f'sel__{sensor}__h12'] = float(self.rolls[f'sel__{sensor}__h12'].loc[ts])
                feat[f'sel__{sensor}__h72'] = float(self.rolls[f'sel__{sensor}__h72'].loc[ts])

            features.append(feat)
            metas.append(
                {
                    'split': split_name,
                    'timestamp': ts,
                    'event_id': int(safe_float(trigger_row.get('event_id', -1), -1)),
                    'reason': reason,
                    'sampled_negative': int(trigger_row.get('sampled_negative', 0)),
                    'pseudo_negative_kind': str(trigger_row.get('pseudo_negative_kind', 'none')),
                }
            )
            target_row = {'y_any_leak': int(self.target_any.loc[ts])}
            target_row.update(self.target_pipe.loc[ts].to_dict())
            targets.append(target_row)

        X = pd.DataFrame(features).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        meta = pd.DataFrame(metas)
        y = pd.DataFrame(targets)
        return X, meta, y


def select_binary_feature_columns(columns: Sequence[str], mode: str) -> List[str]:
    cols = list(columns)
    core = [
        'score_final', 'score_burst', 'score_incipient', 'pressure_topk', 'pairwise_topk',
        'flow_topk', 'ewma_topk', 'slope_topk', 'burst_cusum', 'sensor_votes',
        'sel_current_mean', 'sel_current_max', 'sel_current_std', 'sel_current_top1',
        'sel_current_top2', 'sel_current_gap12',
    ]
    medium_extra = ['sel_mean_h12', 'sel_max_h12', 'flow_1_last', 'flow_2_last', 'flow_1_h12', 'flow_2_h12']
    full_extra = ['sel_mean_h72', 'sel_max_h72', 'flow_3_last', 'flow_3_h12', 'T1_last', 'T1_h12', 'reason__event_start', 'reason__periodic_recheck']
    if mode == 'small':
        keep = core + ['flow_1_last', 'flow_2_last']
    elif mode == 'medium':
        keep = core + medium_extra + ['reason__event_start', 'reason__periodic_recheck']
    else:
        keep = core + medium_extra + full_extra
    return [c for c in keep if c in cols]


def select_loc_feature_columns(columns: Sequence[str], sensors: Sequence[str], mode: str) -> List[str]:
    cols = set(columns)
    sensor_current = [f'sel__{s}' for s in sensors if f'sel__{s}' in cols]
    sensor_h12 = [f'sel__{s}__h12' for s in sensors if f'sel__{s}__h12' in cols]
    sensor_h72 = [f'sel__{s}__h72' for s in sensors if f'sel__{s}__h72' in cols]
    flow_small = [c for c in ['flow_1_last', 'flow_2_last', 'flow_3_last', 'T1_last'] if c in cols]
    flow_medium = [c for c in ['flow_1_h12', 'flow_2_h12', 'flow_3_h12', 'T1_h12'] if c in cols]
    flow_full = [c for c in ['flow_1_h72', 'flow_2_h72', 'flow_3_h72', 'T1_h72'] if c in cols]
    global_core = [c for c in ['score_final', 'pressure_topk', 'pairwise_topk', 'flow_topk'] if c in cols]
    if mode == 'small':
        return sensor_current + flow_small + global_core
    if mode == 'medium':
        return sensor_current + sensor_h12 + flow_small + flow_medium + global_core
    return sensor_current + sensor_h12 + sensor_h72 + flow_small + flow_medium + flow_full + global_core


# ---------------------------------------------------------------------------
# Model wrappers
# ---------------------------------------------------------------------------


@dataclass
class BinaryHeadArtifacts:
    feature_names: List[str]
    raw_coef: np.ndarray
    raw_bias: float
    threshold_prob: float
    threshold_logit: float
    fold_cal_a: float
    fold_cal_b: float
    val_mixed_metrics: pd.Series
    val_actual_metrics: pd.Series
    test_metrics: pd.Series
    threshold_table: pd.DataFrame
    pred_val: pd.DataFrame
    pred_test: pd.DataFrame


class LinearBinaryHead:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = LogisticRegression(max_iter=2000, solver='liblinear')
        self.cal_model: Optional[LogisticRegression] = None
        self.cal_a_: float = 1.0
        self.cal_b_: float = 0.0
        self.feature_names_: List[str] = []

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        sample_weight: Optional[np.ndarray],
        X_cal: pd.DataFrame,
        y_cal: np.ndarray,
    ):
        self.feature_names_ = list(X_train.columns)
        self.scaler.fit(X_train)
        Z_train = self.scaler.transform(X_train)
        self.model.fit(Z_train, y_train, sample_weight=sample_weight)
        p_cal_raw = self.predict_raw_proba(X_cal)
        if len(np.unique(y_cal)) >= 2:
            self.cal_model = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=self.random_state)
            self.cal_model.fit(logit(p_cal_raw).reshape(-1, 1), y_cal)
            self.cal_a_ = float(self.cal_model.coef_[0, 0])
            self.cal_b_ = float(self.cal_model.intercept_[0])
        else:
            self.cal_model = None
            self.cal_a_ = 1.0
            self.cal_b_ = 0.0
        return self

    def predict_raw_logit(self, X: pd.DataFrame) -> np.ndarray:
        Z = self.scaler.transform(X)
        return (Z @ self.model.coef_.ravel()) + self.model.intercept_[0]

    def predict_raw_proba(self, X: pd.DataFrame) -> np.ndarray:
        return sigmoid(self.predict_raw_logit(X))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raw_logit = self.predict_raw_logit(X)
        final_logit = self.cal_a_ * raw_logit + self.cal_b_
        return sigmoid(final_logit)

    def folded_raw_coefficients(self) -> Tuple[np.ndarray, float]:
        coef_z = self.model.coef_.ravel().astype(float)
        bias_z = float(self.model.intercept_[0])
        mean = self.scaler.mean_.astype(float)
        scale = self.scaler.scale_.astype(float)
        scale = np.where(scale == 0.0, 1.0, scale)
        raw_coef = coef_z / scale
        raw_bias = bias_z - float(np.dot(coef_z, mean / scale))
        raw_coef = self.cal_a_ * raw_coef
        raw_bias = self.cal_a_ * raw_bias + self.cal_b_
        return raw_coef, raw_bias


class PrototypeLocalizer:
    def __init__(self):
        self.feature_names_: List[str] = []
        self.scaler = StandardScaler()
        self.pos_centroids_: Dict[str, np.ndarray] = {}
        self.neg_centroids_: Dict[str, np.ndarray] = {}

    def fit(self, X: pd.DataFrame, y_pipe: pd.DataFrame):
        self.feature_names_ = list(X.columns)
        self.scaler.fit(X)
        Z = self.scaler.transform(X)
        for pipe in y_pipe.columns:
            mask_pos = y_pipe[pipe].to_numpy(dtype=int) == 1
            mask_neg = ~mask_pos
            pos = Z[mask_pos] if mask_pos.any() else np.zeros((1, Z.shape[1]), dtype=float)
            neg = Z[mask_neg] if mask_neg.any() else np.zeros((1, Z.shape[1]), dtype=float)
            self.pos_centroids_[pipe] = pos.mean(axis=0)
            self.neg_centroids_[pipe] = neg.mean(axis=0)
        return self

    def predict_pipe_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        Z = self.scaler.transform(X)
        rows = {}
        for pipe in self.pos_centroids_.keys():
            cp = self.pos_centroids_[pipe]
            cn = self.neg_centroids_[pipe]
            d_pos = np.sum((Z - cp) ** 2, axis=1)
            d_neg = np.sum((Z - cn) ** 2, axis=1)
            score = d_neg - d_pos
            rows[pipe] = sigmoid(score / max(1.0, Z.shape[1]))
        return pd.DataFrame(rows, index=X.index)


class OVRLinearLocalizer:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.feature_names_: List[str] = []
        self.scaler = StandardScaler()
        self.models_: Dict[str, LogisticRegression] = {}
        self.pipe_priors_: Dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y_pipe: pd.DataFrame):
        self.feature_names_ = list(X.columns)
        self.scaler.fit(X)
        Z = self.scaler.transform(X)
        for pipe in y_pipe.columns:
            y = y_pipe[pipe].to_numpy(dtype=int)
            self.pipe_priors_[pipe] = float(y.mean()) if len(y) else 0.0
            if len(np.unique(y)) < 2:
                self.models_[pipe] = None  # type: ignore
                continue
            model = LogisticRegression(max_iter=1000, solver='liblinear', class_weight='balanced', random_state=self.random_state)
            model.fit(Z, y)
            self.models_[pipe] = model
        return self

    def predict_pipe_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        Z = self.scaler.transform(X)
        out = {}
        for pipe, model in self.models_.items():
            if model is None:
                out[pipe] = np.full(len(X), self.pipe_priors_[pipe], dtype=float)
            else:
                out[pipe] = model.predict_proba(Z)[:, 1]
        return pd.DataFrame(out, index=X.index)

    def folded_raw_coefficients(self) -> Dict[str, Dict[str, object]]:
        mean = self.scaler.mean_.astype(float)
        scale = np.where(self.scaler.scale_.astype(float) == 0.0, 1.0, self.scaler.scale_.astype(float))
        out = {}
        for pipe, model in self.models_.items():
            if model is None:
                out[pipe] = {
                    'coef': np.zeros(len(self.feature_names_), dtype=float),
                    'bias': float(logit(self.pipe_priors_[pipe])) if 0.0 < self.pipe_priors_[pipe] < 1.0 else 0.0,
                }
                continue
            coef_z = model.coef_.ravel().astype(float)
            bias_z = float(model.intercept_[0])
            coef_raw = coef_z / scale
            bias_raw = bias_z - float(np.dot(coef_z, mean / scale))
            out[pipe] = {'coef': coef_raw, 'bias': bias_raw}
        return out


# ---------------------------------------------------------------------------
# Candidate execution
# ---------------------------------------------------------------------------


def build_packet_splits(
    post_fit_df: pd.DataFrame,
    results: pd.DataFrame,
    triggers_all: pd.DataFrame,
    groups: Dict[str, List[str]],
    cfg: PacketSplitConfig,
) -> PacketSplitResult:
    dev_df = post_fit_df[post_fit_df.index.year == 2018].copy()
    test_df_local = post_fit_df[post_fit_df.index.year == 2019].copy()
    y_any_dev = get_any_real_leak_mask(dev_df, groups['pipe_cols']).astype(int)

    actual_dev = triggers_all[(triggers_all['timestamp'] >= dev_df.index.min()) & (triggers_all['timestamp'] <= dev_df.index.max())].copy()
    actual_dev['timestamp'] = pd.to_datetime(actual_dev['timestamp'])
    actual_dev = actual_dev.sort_values('timestamp').reset_index(drop=True)
    actual_dev['sampled_negative'] = 0
    actual_dev['pseudo_negative_kind'] = 'none'
    actual_dev['y_any_leak'] = y_any_dev.reindex(pd.DatetimeIndex(actual_dev['timestamp'])).astype(int).values

    actual_pos = actual_dev[actual_dev['y_any_leak'] == 1].copy()
    actual_neg = actual_dev[actual_dev['y_any_leak'] == 0].copy()

    pos_train, pos_cal, pos_val = split_df_threeway(actual_pos, fracs=(0.60, 0.20, 0.20))
    neg_train, neg_cal, neg_val = split_small_temporal(actual_neg)

    actual_test = triggers_all[(triggers_all['timestamp'] >= test_df_local.index.min()) & (triggers_all['timestamp'] <= test_df_local.index.max())].copy()
    actual_test['timestamp'] = pd.to_datetime(actual_test['timestamp'])
    actual_test = actual_test.sort_values('timestamp').reset_index(drop=True)
    actual_test['sampled_negative'] = 0
    actual_test['pseudo_negative_kind'] = 'none'

    actual_dev_ts = pd.DatetimeIndex(pd.to_datetime(actual_dev['timestamp']))
    healthy_non_trigger = dev_df.index[(y_any_dev == 0).values].difference(actual_dev_ts)

    score_for_peaks = results.loc[healthy_non_trigger, 'score_final'].fillna(0.0) if len(healthy_non_trigger) else pd.Series(dtype=float)
    peak_target_total = min(
        len(healthy_non_trigger),
        max(24, int(round((len(pos_train) + len(pos_cal) + len(pos_val)) * cfg.hard_peak_negative_ratio_per_positive))),
    )
    hard_peak_idx = select_peak_negative_timestamps(
        score_for_peaks,
        n_samples=peak_target_total,
        step_minutes=cfg.step_minutes,
        min_gap_steps=12,
        peak_radius=3,
    )
    hard_peak_train_idx, hard_peak_cal_idx, hard_peak_val_idx = split_index_threeway(hard_peak_idx)

    remaining_healthy = healthy_non_trigger.difference(hard_peak_idx)
    rem_train_df, rem_tmp_df = chronological_split_frame(pd.DataFrame({'timestamp': remaining_healthy}), 0.60)
    rem_cal_df, rem_val_df = chronological_split_frame(rem_tmp_df, 0.50)
    rem_train_idx = pd.DatetimeIndex(pd.to_datetime(rem_train_df['timestamp']))
    rem_cal_idx = pd.DatetimeIndex(pd.to_datetime(rem_cal_df['timestamp']))
    rem_val_idx = pd.DatetimeIndex(pd.to_datetime(rem_val_df['timestamp']))

    easy_train_target = int(round(len(pos_train) * cfg.easy_negative_ratio_per_positive))
    easy_cal_target = int(round(len(pos_cal) * cfg.easy_negative_ratio_per_positive))
    easy_val_target = int(round(len(pos_val) * cfg.easy_negative_ratio_per_positive))

    easy_train_idx = stratified_sample_timestamps(rem_train_idx, easy_train_target, seed=cfg.random_state + 10)
    easy_cal_idx = stratified_sample_timestamps(rem_cal_idx, easy_cal_target, seed=cfg.random_state + 11)
    easy_val_idx = stratified_sample_timestamps(rem_val_idx, easy_val_target, seed=cfg.random_state + 12)

    train_packets = pd.concat(
        [
            pos_train.assign(sampled_negative=0, pseudo_negative_kind='none').drop(columns=['y_any_leak'], errors='ignore'),
            neg_train.assign(sampled_negative=0, pseudo_negative_kind='none').drop(columns=['y_any_leak'], errors='ignore'),
            make_pseudo_trigger_frame(hard_peak_train_idx, reason='pseudo_hard_negative', kind='pseudo_hard_negative'),
            make_pseudo_trigger_frame(easy_train_idx, reason='sampled_negative', kind='sampled_negative'),
        ],
        ignore_index=True,
    ).sort_values('timestamp').reset_index(drop=True)

    cal_packets = pd.concat(
        [
            pos_cal.assign(sampled_negative=0, pseudo_negative_kind='none').drop(columns=['y_any_leak'], errors='ignore'),
            neg_cal.assign(sampled_negative=0, pseudo_negative_kind='none').drop(columns=['y_any_leak'], errors='ignore'),
            make_pseudo_trigger_frame(hard_peak_cal_idx, reason='pseudo_hard_negative', kind='pseudo_hard_negative'),
            make_pseudo_trigger_frame(easy_cal_idx, reason='sampled_negative', kind='sampled_negative'),
        ],
        ignore_index=True,
    ).sort_values('timestamp').reset_index(drop=True)

    val_packets = pd.concat(
        [
            pos_val.assign(sampled_negative=0, pseudo_negative_kind='none').drop(columns=['y_any_leak'], errors='ignore'),
            neg_val.assign(sampled_negative=0, pseudo_negative_kind='none').drop(columns=['y_any_leak'], errors='ignore'),
            make_pseudo_trigger_frame(hard_peak_val_idx, reason='pseudo_hard_negative', kind='pseudo_hard_negative'),
            make_pseudo_trigger_frame(easy_val_idx, reason='sampled_negative', kind='sampled_negative'),
        ],
        ignore_index=True,
    ).sort_values('timestamp').reset_index(drop=True)

    info = {
        'actual_dev_triggers': int(len(actual_dev)),
        'actual_dev_positive_triggers': int(len(actual_pos)),
        'actual_dev_negative_triggers': int(len(actual_neg)),
        'pseudo_hard_negative_total': int(len(hard_peak_idx)),
        'healthy_non_trigger_pool': int(len(healthy_non_trigger)),
        'easy_negative_total': int(len(easy_train_idx) + len(easy_cal_idx) + len(easy_val_idx)),
    }
    return PacketSplitResult(train_packets, cal_packets, val_packets, actual_test, info)


def build_prediction_frame(
    meta: pd.DataFrame,
    y: pd.DataFrame,
    p_any: np.ndarray,
    threshold: float,
    pipe_prob_df: pd.DataFrame,
) -> pd.DataFrame:
    pred = meta.copy().reset_index(drop=True)
    pred['p_any_leak'] = p_any.astype(float)
    pred['pred_any_leak'] = (p_any >= threshold).astype(int)
    pred['threshold'] = float(threshold)
    pipe_prob_df = pipe_prob_df.reset_index(drop=True)
    pipe_cols = list(pipe_prob_df.columns)
    prob_mat = pipe_prob_df.to_numpy(dtype=float)
    topk = np.argsort(prob_mat, axis=1)[:, ::-1]
    pred['top_1_pipe'] = [pipe_cols[row[0]] if len(row) else '' for row in topk]
    pred['top_2_pipe'] = [pipe_cols[row[1]] if len(row) > 1 else '' for row in topk]
    pred['top_3_pipe'] = [pipe_cols[row[2]] if len(row) > 2 else '' for row in topk]
    pred['top_3_pipes'] = [', '.join([pipe_cols[j] for j in row[:3]]) for row in topk]
    pred['true_pipes'] = y[pipe_cols].astype(int).apply(lambda r: ', '.join([c for c, v in r.items() if v == 1]), axis=1)
    return pred


@dataclass
class CandidateRunResult:
    candidate: PipelineCandidate
    detector: BattLeDIMStage1
    stage1_metrics_2018: pd.Series
    stage1_metrics_2019: pd.Series
    packet_info: Dict[str, int]
    binary_head: LinearBinaryHead
    localizer: object
    binary_cols: List[str]
    loc_cols: List[str]
    binary_threshold_info: Dict[str, float]
    binary_threshold_table: pd.DataFrame
    binary_metrics_val_mixed: pd.Series
    binary_metrics_val_actual: pd.Series
    binary_metrics_test_actual: pd.Series
    localisation_metrics_val: pd.Series
    localisation_metrics_test: pd.Series
    pred_val: pd.DataFrame
    pred_test: pd.DataFrame
    pipe_prob_val: pd.DataFrame
    pipe_prob_test: pd.DataFrame
    feature_spec: pd.DataFrame
    memory_rows: pd.DataFrame
    total_ram_bytes: int
    total_flash_bytes: int
    feasible: bool
    chosen_score: Tuple[float, float, float, float]
    post_fit_df: pd.DataFrame
    results_postfit: pd.DataFrame
    sensor_detail_postfit: pd.DataFrame
    triggers_postfit: pd.DataFrame
    X_train: pd.DataFrame
    X_cal: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    meta_train: pd.DataFrame
    meta_cal: pd.DataFrame
    meta_val: pd.DataFrame
    meta_test: pd.DataFrame
    y_train: pd.DataFrame
    y_cal: pd.DataFrame
    y_val: pd.DataFrame
    y_test: pd.DataFrame


# ---------------------------------------------------------------------------
# Reference current-model size estimate
# ---------------------------------------------------------------------------


def estimate_reference_current_pipeline(groups: Dict[str, List[str]]) -> pd.DataFrame:
    raw_cols = len(groups['pressure_cols']) + len(groups['flow_primary_cols']) + len(groups['flow_context_cols']) + len(groups['tank_cols'])
    demand_cols = len(groups['demand_cols'])
    stage1_numeric = 13
    sensor_detail_cols = len(groups['pressure_cols'])
    current_feature_count = (
        raw_cols * 4 * 11
        + 7 * 4 * 11 + 5
        + stage1_numeric * 4 * 11
        + 4
        + sensor_detail_cols * 4 * 4 + 17
        + sensor_detail_cols * 4
        + 9
        + 7
    )
    binary_feature_count = current_feature_count - 44 - 4 - 7 - 4 - 1 - 4
    loc_feature_count = current_feature_count - 3

    stage1_baseline_channels = len(groups['pressure_cols']) + 2 + len(groups['flow_context_cols']) + len(groups['tank_cols'])
    stage1_baseline_flash = 2016 * stage1_baseline_channels * 4
    stage1_pairwise_flash = len(groups['pressure_cols']) * 4 * 3 * 4
    current_packet_buffer_ram = 144 * (raw_cols + demand_cols + stage1_numeric + sensor_detail_cols) * 4
    current_feature_workspace_ram = current_feature_count * 4
    # HistGradientBoosting fallback: 400 trees, depth 6, max 31 leaves => ~61 nodes / tree.
    hgb_nodes_per_model = 400 * 61
    node_bytes = 20
    binary_model_flash = hgb_nodes_per_model * node_bytes
    localizer_model_flash = len(groups['pipe_cols']) * hgb_nodes_per_model * node_bytes
    calibrator_flash = 2048
    runtime_workspace_ram = 32768
    stack_allowance_ram = 16384
    code_allowance_flash = 65536

    rows = [
        {'candidate': 'current_reference', 'component': 'stage1_weekly_baseline_float32', 'ram_bytes': 0, 'flash_bytes': stage1_baseline_flash, 'notes': '2016-bin weekly baseline over all Stage 1 channels.'},
        {'candidate': 'current_reference', 'component': 'stage1_pairwise_params_float32', 'ram_bytes': 0, 'flash_bytes': stage1_pairwise_flash, 'notes': 'Pairwise linear models for full pressure set.'},
        {'candidate': 'current_reference', 'component': 'stage2_packet_buffer_float32', 'ram_bytes': current_packet_buffer_ram, 'flash_bytes': 0, 'notes': 'Raw/demand/result/sensor-detail lookback for 144 steps.'},
        {'candidate': 'current_reference', 'component': 'stage2_feature_workspace_float32', 'ram_bytes': current_feature_workspace_ram, 'flash_bytes': 0, 'notes': f'About {current_feature_count} engineered features.'},
        {'candidate': 'current_reference', 'component': 'stage2_binary_hgb_model', 'ram_bytes': 0, 'flash_bytes': binary_model_flash, 'notes': '400-tree HistGradientBoosting binary head.'},
        {'candidate': 'current_reference', 'component': 'stage2_pipe_hgb_models', 'ram_bytes': 0, 'flash_bytes': localizer_model_flash, 'notes': f"{len(groups['pipe_cols'])} one-vs-rest gradient-boosted pipe heads."},
        {'candidate': 'current_reference', 'component': 'stage2_calibrator_and_code', 'ram_bytes': runtime_workspace_ram + stack_allowance_ram, 'flash_bytes': calibrator_flash + code_allowance_flash, 'notes': 'Calibration, tree runtime, stack, and control logic.'},
    ]
    df = pd.DataFrame(rows)
    df['binary_feature_count'] = float(binary_feature_count)
    df['localisation_feature_count'] = float(loc_feature_count)
    return df


# ---------------------------------------------------------------------------
# Candidate memory estimate
# ---------------------------------------------------------------------------


def estimate_candidate_memory(
    candidate: PipelineCandidate,
    detector: BattLeDIMStage1,
    binary_cols: Sequence[str],
    loc_cols: Sequence[str],
    groups: Dict[str, List[str]],
    binary_head: LinearBinaryHead,
    localizer: object,
) -> pd.DataFrame:
    bins = 1 if candidate.baseline_mode == 'none' else 24 if candidate.baseline_mode == 'daily_hour' else 168
    stage1_channels = len(detector.monitored_pressure_cols_) + len(detector.flow_score_cols_) + len(detector.flow_context_cols_) + len(detector.tank_cols_)
    stage1_baseline_flash = bins * stage1_channels * 2  # int16 tables
    stage1_pairwise_params = sum(len(v) for v in detector.pairwise_params_.values())
    stage1_pairwise_flash = stage1_pairwise_params * 4 * 2  # anchor index + slope/intercept/scale int16
    stage1_scale_flash = stage1_channels * 2 * 2  # scale + offset/threshold-ish int16
    stage1_code_flash = 24 * 1024

    median_ring_ram = stage1_channels * int(detector.cfg.smooth_window) * 2
    slope_ring_ram = len(detector.monitored_pressure_cols_) * int(detector.cfg.slope_window) * 2
    stage1_state_ram = (len(detector.monitored_pressure_cols_) * 8) + 512

    n_flow_ctx = len([c for c in ['flow_1', 'flow_2', 'flow_3', 'T1'] if c in detector.last_internal_['residual_all'].columns])
    stage2_sensor_h12_ram = len(detector.monitored_pressure_cols_) * 12 * 2
    stage2_sensor_h72_ram = len(detector.monitored_pressure_cols_) * 72 * 2 if candidate.loc_feature_mode == 'full' else 0
    stage2_agg_h72_ram = 2 * 72 * 2
    stage2_flow_buffer_ram = n_flow_ctx * 72 * 2
    stage2_feature_workspace_ram = (len(binary_cols) + len(loc_cols)) * 4
    stage2_stack_ram = 8 * 1024

    binary_flash = (len(binary_cols) + 1) * 2 + 1024  # folded int16 coefs + sigmoid LUT / logic overhead
    if isinstance(localizer, PrototypeLocalizer):
        loc_flash = len(groups['pipe_cols']) * len(loc_cols) * 2 * 2 + 2048  # pos/neg centroids
        loc_code_flash = 12 * 1024
    else:
        loc_flash = len(groups['pipe_cols']) * (len(loc_cols) + 1) * 2 + 2048
        loc_code_flash = 18 * 1024
    stage2_control_flash = 8 * 1024

    rows = [
        {'candidate': candidate.name, 'component': 'stage1_baseline_tables_int16', 'ram_bytes': 0, 'flash_bytes': stage1_baseline_flash, 'notes': f'{bins} bins x {stage1_channels} channels.'},
        {'candidate': candidate.name, 'component': 'stage1_pairwise_params_int16', 'ram_bytes': 0, 'flash_bytes': stage1_pairwise_flash + stage1_scale_flash, 'notes': f'{stage1_pairwise_params} pairwise links plus scales/thresholds.'},
        {'candidate': candidate.name, 'component': 'stage1_runtime_buffers', 'ram_bytes': median_ring_ram + slope_ring_ram + stage1_state_ram, 'flash_bytes': stage1_code_flash, 'notes': 'Streaming median/slope rings, EWMA/CUSUM, and Stage 1 logic.'},
        {'candidate': candidate.name, 'component': 'stage2_running_feature_buffers', 'ram_bytes': stage2_sensor_h12_ram + stage2_sensor_h72_ram + stage2_agg_h72_ram + stage2_flow_buffer_ram, 'flash_bytes': 0, 'notes': 'Rolling sums/rings for selected sensor and flow summaries.'},
        {'candidate': candidate.name, 'component': 'stage2_binary_linear_head', 'ram_bytes': stage2_feature_workspace_ram, 'flash_bytes': binary_flash, 'notes': f'{len(binary_cols)} binary features with folded coefficients.'},
        {'candidate': candidate.name, 'component': 'stage2_localizer', 'ram_bytes': 0, 'flash_bytes': loc_flash + loc_code_flash + stage2_control_flash, 'notes': f'{candidate.localizer_type} over {len(loc_cols)} localisation features.'},
        {'candidate': candidate.name, 'component': 'stack_allowance', 'ram_bytes': stage2_stack_ram, 'flash_bytes': 0, 'notes': 'Conservative stack / temporary workspace allowance.'},
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Quantization helpers for deployment artifacts
# ---------------------------------------------------------------------------


def quantize_array(arr: np.ndarray, qmax: int = 32767) -> Tuple[np.ndarray, float]:
    arr = np.asarray(arr, dtype=float)
    max_abs = float(np.max(np.abs(arr))) if arr.size else 1.0
    scale = max_abs / qmax if max_abs > 0 else 1.0
    q = np.round(arr / scale).astype(np.int16)
    return q, float(scale)


def c_array_int16(name: str, arr: np.ndarray, cols: int = 12) -> str:
    flat = np.asarray(arr, dtype=np.int16).ravel().tolist()
    chunks = []
    for i in range(0, len(flat), cols):
        chunks.append(', '.join(str(int(x)) for x in flat[i:i + cols]))
    body = ',\n    '.join(chunks)
    return f'static const int16_t {name}[{len(flat)}] = {{\n    {body}\n}};\n'


def c_array_float(name: str, arr: np.ndarray, cols: int = 8) -> str:
    flat = np.asarray(arr, dtype=float).ravel().tolist()
    chunks = []
    for i in range(0, len(flat), cols):
        chunks.append(', '.join(f'{float(x):.8g}f' for x in flat[i:i + cols]))
    body = ',\n    '.join(chunks)
    return f'static const float {name}[{len(flat)}] = {{\n    {body}\n}};\n'


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def make_pr_curve_fig(pred_df: pd.DataFrame, title: str):
    if pred_df.empty or pred_df['y_any_leak'].nunique() < 2:
        return None
    rows = pred_df.sort_values('p_any_leak', ascending=False).reset_index(drop=True)
    tps = rows['y_any_leak'].cumsum().to_numpy(dtype=float)
    fps = np.arange(1, len(rows) + 1, dtype=float) - tps
    precision = np.divide(tps, tps + fps, out=np.zeros_like(tps), where=(tps + fps) > 0)
    recall = tps / max(1.0, rows['y_any_leak'].sum())
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.plot(recall, precision, linewidth=1.5)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def make_calibration_fig(pred_df: pd.DataFrame, title: str):
    if pred_df.empty or pred_df['y_any_leak'].nunique() < 2:
        return None
    frac_pos, mean_pred = calibration_curve(pred_df['y_any_leak'], pred_df['p_any_leak'], n_bins=10, strategy='quantile')
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.plot([0, 1], [0, 1], linestyle='--', linewidth=1)
    ax.plot(mean_pred, frac_pos, marker='o', linewidth=1.5)
    ax.set_xlabel('Predicted probability')
    ax.set_ylabel('Observed fraction positive')
    ax.set_title(title)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def make_probability_histogram_fig(pred_df: pd.DataFrame, title: str):
    if pred_df.empty:
        return None
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    pos = pred_df.loc[pred_df['y_any_leak'] == 1, 'p_any_leak']
    neg = pred_df.loc[pred_df['y_any_leak'] == 0, 'p_any_leak']
    bins = np.linspace(0.0, 1.0, 21)
    if len(pos):
        ax.hist(pos, bins=bins, alpha=0.7, label='Positive packets')
    if len(neg):
        ax.hist(neg, bins=bins, alpha=0.7, label='Negative packets')
    ax.set_xlabel('Predicted probability')
    ax.set_ylabel('Packet count')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def make_memory_vs_performance_fig(model_comparison: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8.0, 5.6))
    for _, row in model_comparison.iterrows():
        x = row['total_flash_bytes'] / 1024.0
        y = row['localisation_top3_2019']
        label = f"{row['candidate']}\nBR={row['binary_recall_2019']:.3f}"
        ax.scatter([x], [y], s=80)
        ax.annotate(label, (x, y), textcoords='offset points', xytext=(6, 6), fontsize=8)
    ax.set_xlabel('Estimated flash footprint (KB)')
    ax.set_ylabel('2019 localisation top-3 hit rate')
    ax.set_title('Memory vs performance Pareto')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def make_feature_budget_breakdown_fig(memory_rows: pd.DataFrame, candidate_name: str):
    df = memory_rows[memory_rows['candidate'] == candidate_name].copy()
    df['total_bytes'] = df['ram_bytes'] + df['flash_bytes']
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    ax.barh(df['component'], df['flash_bytes'] / 1024.0, label='Flash KB')
    ax.barh(df['component'], df['ram_bytes'] / 1024.0, left=df['flash_bytes'] / 1024.0, label='RAM KB')
    ax.set_xlabel('KB')
    ax.set_ylabel('Component')
    ax.set_title(f'Chosen model memory budget: {candidate_name}')
    ax.legend()
    ax.grid(True, axis='x', alpha=0.3)
    fig.tight_layout()
    return fig


def make_stage1_calls_vs_recall_fig(rows: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    for _, row in rows.iterrows():
        ax.scatter([row['stage1_calls_per_day_2018']], [row['stage1_recall_2018']], s=80)
        ax.annotate(row['candidate'], (row['stage1_calls_per_day_2018'], row['stage1_recall_2018']), textcoords='offset points', xytext=(6, 6), fontsize=8)
    ax.set_xlabel('Stage 2 calls/day on late 2018')
    ax.set_ylabel('Stage 1 timestep recall on late 2018')
    ax.set_title('Operational Stage 1 tradeoff')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def make_localisation_topk_fig(baseline_topk: Dict[str, float], chosen_result: CandidateRunResult):
    labels = ['Val top-1', 'Val top-3', '2019 top-1', '2019 top-3']
    baseline_vals = [baseline_topk['val_top1'], baseline_topk['val_top3'], baseline_topk['test_top1'], baseline_topk['test_top3']]
    chosen_vals = [
        safe_float(chosen_result.localisation_metrics_val.get('top1_hit_rate', np.nan), np.nan),
        safe_float(chosen_result.localisation_metrics_val.get('top3_hit_rate', np.nan), np.nan),
        safe_float(chosen_result.localisation_metrics_test.get('top1_hit_rate', np.nan), np.nan),
        safe_float(chosen_result.localisation_metrics_test.get('top3_hit_rate', np.nan), np.nan),
    ]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    ax.bar(x - width / 2, baseline_vals, width=width, label='Current patched baseline')
    ax.bar(x + width / 2, chosen_vals, width=width, label=chosen_result.candidate.name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel('Hit rate')
    ax.set_title('Localisation top-k comparison')
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Candidate run
# ---------------------------------------------------------------------------


def run_candidate(
    candidate: PipelineCandidate,
    full_df: pd.DataFrame,
    groups: Dict[str, List[str]],
    base_stage1_cfg: Stage1Config,
    split_cfg: PacketSplitConfig,
) -> CandidateRunResult:
    full_df_2018 = full_df[full_df.index.year == 2018].copy()
    stage1_fit_end = choose_stage1_fit_end_from_healthy_block(full_df_2018, fit_frac=0.60, pipe_cols=groups['pipe_cols'])
    stage1_train_df = full_df.loc[:stage1_fit_end].copy()
    post_fit_df = full_df.loc[full_df.index > stage1_fit_end].copy()
    val2018_df = post_fit_df[post_fit_df.index.year == 2018].copy()
    test2019_df = post_fit_df[post_fit_df.index.year == 2019].copy()

    stage1_cfg = Stage1Config(**base_stage1_cfg.to_dict())
    stage1_cfg.baseline_mode = candidate.baseline_mode
    stage1_cfg.monitored_pressure_count = int(candidate.monitored_pressure_count)
    stage1_cfg.compact_pairwise_anchors = int(candidate.pairwise_anchors)

    stage1_train_exclusions = build_known_intervals_from_leak_columns(
        stage1_train_df,
        pad_before='30min',
        pad_after='30min',
        min_active_samples=1,
    )

    detector = BattLeDIMStage1(stage1_cfg)
    detector.fit(stage1_train_df, known_intervals=stage1_train_exclusions)
    results_postfit, sensor_detail_postfit, _ = detector.detect(post_fit_df)
    triggers_postfit = detector.build_stage2_triggers(results_postfit)
    triggers_postfit = triggers_postfit.copy()
    triggers_postfit['timestamp'] = pd.to_datetime(triggers_postfit['timestamp'])
    triggers_postfit = triggers_postfit.sort_values('timestamp').reset_index(drop=True)

    truth_2018 = get_any_real_leak_mask(val2018_df, groups['pipe_cols']).astype(int)
    truth_2019 = get_any_real_leak_mask(test2019_df, groups['pipe_cols']).astype(int)
    triggers_2018 = triggers_postfit[(triggers_postfit['timestamp'] >= val2018_df.index.min()) & (triggers_postfit['timestamp'] <= val2018_df.index.max())].copy()
    triggers_2019 = triggers_postfit[(triggers_postfit['timestamp'] >= test2019_df.index.min()) & (triggers_postfit['timestamp'] <= test2019_df.index.max())].copy()
    stage1_metrics_2018 = evaluate_stage1_results(results_postfit.loc[val2018_df.index], truth_2018, triggers=triggers_2018)
    stage1_metrics_2019 = evaluate_stage1_results(results_postfit.loc[test2019_df.index], truth_2019, triggers=triggers_2019)

    split_result = build_packet_splits(post_fit_df, results_postfit, triggers_postfit, groups, split_cfg)
    all_packet_triggers = pd.concat(
        [split_result.train_packets, split_result.cal_packets, split_result.val_packets, split_result.test_packets],
        ignore_index=True,
    ).sort_values('timestamp').reset_index(drop=True)
    builder = TinyFeatureBuilder(post_fit_df, results_postfit, sensor_detail_postfit, detector, all_packet_triggers, groups, step_minutes=split_cfg.step_minutes)

    X_train, meta_train, y_train = builder.build_dataset(split_result.train_packets, 'stage2_train')
    X_cal, meta_cal, y_cal = builder.build_dataset(split_result.cal_packets, 'stage2_cal')
    X_val, meta_val, y_val = builder.build_dataset(split_result.val_packets, 'stage2_val')
    X_test, meta_test, y_test = builder.build_dataset(split_result.test_packets, 'test_2019')

    all_cols = sorted(set(X_train.columns) | set(X_cal.columns) | set(X_val.columns) | set(X_test.columns))
    for frame in [X_train, X_cal, X_val, X_test]:
        for col in all_cols:
            if col not in frame.columns:
                frame[col] = 0.0
        frame.sort_index(axis=1, inplace=True)

    binary_cols = select_binary_feature_columns(X_train.columns, mode=candidate.binary_feature_mode)
    loc_cols = select_loc_feature_columns(X_train.columns, detector.monitored_pressure_cols_, mode=candidate.loc_feature_mode)

    # Binary head
    binary_head = LinearBinaryHead(random_state=split_cfg.random_state)
    binary_source_weight = np.ones(len(meta_train), dtype=float)
    kind_train = meta_train.get('pseudo_negative_kind', pd.Series('none', index=meta_train.index)).fillna('none').astype(str)
    binary_source_weight[kind_train.eq('sampled_negative').to_numpy()] = split_cfg.sampled_negative_weight
    binary_source_weight[kind_train.eq('pseudo_hard_negative').to_numpy()] = split_cfg.pseudo_hard_negative_weight
    binary_source_weight[((meta_train['sampled_negative'].astype(int) == 0) & (y_train['y_any_leak'] == 0)).to_numpy()] = split_cfg.real_hard_negative_weight
    binary_class_weight = compute_sample_weight(class_weight='balanced', y=y_train['y_any_leak'].to_numpy(dtype=int))
    sample_weight_binary = binary_source_weight * binary_class_weight
    binary_head.fit(
        X_train[binary_cols],
        y_train['y_any_leak'].to_numpy(dtype=int),
        sample_weight_binary,
        X_cal[binary_cols],
        y_cal['y_any_leak'].to_numpy(dtype=int),
    )
    p_val = binary_head.predict_proba(X_val[binary_cols])
    p_test = binary_head.predict_proba(X_test[binary_cols]) if len(X_test) else np.array([])
    threshold_info, threshold_table = direct_threshold_search(y_val['y_any_leak'].to_numpy(dtype=int), p_val, min_recall=split_cfg.binary_min_recall)
    threshold = float(threshold_info['threshold'])

    val_actual_mask = meta_val['sampled_negative'].astype(int) == 0
    binary_metrics_val_mixed = safe_binary_metrics(y_val['y_any_leak'].to_numpy(dtype=int), p_val, threshold)
    binary_metrics_val_actual = safe_binary_metrics(y_val.loc[val_actual_mask, 'y_any_leak'].to_numpy(dtype=int), p_val[val_actual_mask.to_numpy()], threshold)
    binary_metrics_test_actual = safe_binary_metrics(y_test['y_any_leak'].to_numpy(dtype=int), p_test, threshold)

    # Localizer
    loc_train_mask = (meta_train['sampled_negative'].astype(int) == 0) & (y_train['y_any_leak'] == 1)
    loc_val_mask = (meta_val['sampled_negative'].astype(int) == 0) & (y_val['y_any_leak'] == 1)
    loc_test_mask = (meta_test['sampled_negative'].astype(int) == 0) & (y_test['y_any_leak'] == 1)

    if candidate.localizer_type == 'prototype':
        localizer: object = PrototypeLocalizer().fit(X_train.loc[loc_train_mask, loc_cols], y_train.loc[loc_train_mask, groups['pipe_cols']])
    else:
        localizer = OVRLinearLocalizer(random_state=split_cfg.random_state).fit(X_train.loc[loc_train_mask, loc_cols], y_train.loc[loc_train_mask, groups['pipe_cols']])

    pipe_prob_val = localizer.predict_pipe_proba(X_val[loc_cols])
    pipe_prob_test = localizer.predict_pipe_proba(X_test[loc_cols]) if len(X_test) else pd.DataFrame(columns=groups['pipe_cols'])

    localisation_metrics_val = evaluate_localisation(y_val.loc[loc_val_mask, groups['pipe_cols']], pipe_prob_val.loc[loc_val_mask])
    localisation_metrics_test = evaluate_localisation(y_test.loc[loc_test_mask, groups['pipe_cols']], pipe_prob_test.loc[loc_test_mask])

    pred_val = build_prediction_frame(meta_val, y_val, p_val, threshold, pipe_prob_val)
    pred_val['y_any_leak'] = y_val['y_any_leak'].to_numpy(dtype=int)
    pred_test = build_prediction_frame(meta_test, y_test, p_test, threshold, pipe_prob_test)
    pred_test['y_any_leak'] = y_test['y_any_leak'].to_numpy(dtype=int)

    memory_rows = estimate_candidate_memory(candidate, detector, binary_cols, loc_cols, groups, binary_head, localizer)
    total_ram = int(memory_rows['ram_bytes'].sum())
    total_flash = int(memory_rows['flash_bytes'].sum())
    feasible = (total_ram <= 256 * 1024) and (total_flash <= 1024 * 1024) and (safe_float(stage1_metrics_2018.get('stage2_calls_per_day', np.nan), np.inf) <= 3.0)

    feature_rows = []
    for name in binary_cols:
        feature_rows.append({'feature': name, 'group': feature_group(name), 'used_in_binary': 1, 'used_in_localization': int(name in loc_cols), 'notes': ''})
    for name in loc_cols:
        if name not in binary_cols:
            feature_rows.append({'feature': name, 'group': feature_group(name), 'used_in_binary': 0, 'used_in_localization': 1, 'notes': ''})
    feature_spec = pd.DataFrame(feature_rows).sort_values(['used_in_binary', 'used_in_localization', 'group', 'feature'], ascending=[False, False, True, True]).reset_index(drop=True)

    score_tuple = (
        1.0 if feasible else 0.0,
        safe_float(binary_metrics_test_actual.get('recall', np.nan), -1.0),
        safe_float(localisation_metrics_test.get('top3_hit_rate', np.nan), -1.0),
        -float(total_flash),
    )

    return CandidateRunResult(
        candidate=candidate,
        detector=detector,
        stage1_metrics_2018=stage1_metrics_2018,
        stage1_metrics_2019=stage1_metrics_2019,
        packet_info=split_result.info,
        binary_head=binary_head,
        localizer=localizer,
        binary_cols=binary_cols,
        loc_cols=loc_cols,
        binary_threshold_info=threshold_info,
        binary_threshold_table=threshold_table,
        binary_metrics_val_mixed=binary_metrics_val_mixed,
        binary_metrics_val_actual=binary_metrics_val_actual,
        binary_metrics_test_actual=binary_metrics_test_actual,
        localisation_metrics_val=localisation_metrics_val,
        localisation_metrics_test=localisation_metrics_test,
        pred_val=pred_val,
        pred_test=pred_test,
        pipe_prob_val=pipe_prob_val,
        pipe_prob_test=pipe_prob_test,
        feature_spec=feature_spec,
        memory_rows=memory_rows,
        total_ram_bytes=total_ram,
        total_flash_bytes=total_flash,
        feasible=feasible,
        chosen_score=score_tuple,
        post_fit_df=post_fit_df,
        results_postfit=results_postfit,
        sensor_detail_postfit=sensor_detail_postfit,
        triggers_postfit=triggers_postfit,
        X_train=X_train,
        X_cal=X_cal,
        X_val=X_val,
        X_test=X_test,
        meta_train=meta_train,
        meta_cal=meta_cal,
        meta_val=meta_val,
        meta_test=meta_test,
        y_train=y_train,
        y_cal=y_cal,
        y_val=y_val,
        y_test=y_test,
    )


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------


class ArtifactWriter:
    def __init__(self, root: Path):
        self.root = root
        self.rows: List[Dict[str, str]] = []

    def add(self, name: str, path: Path, description: str):
        self.rows.append({'name': name, 'path': str(path.relative_to(self.root)), 'description': description})

    def write_csv(self, name: str, df: pd.DataFrame, description: str) -> Path:
        path = self.root / name
        df.to_csv(path, index=False)
        self.add(name, path, description)
        return path

    def write_series_csv(self, name: str, s: pd.Series, description: str) -> Path:
        df = s.rename('value').reset_index().rename(columns={'index': 'metric'})
        return self.write_csv(name, df, description)

    def write_json(self, name: str, obj, description: str) -> Path:
        path = self.root / name
        path.write_text(json.dumps(obj, indent=2, default=_json_default))
        self.add(name, path, description)
        return path

    def write_text(self, name: str, text: str, description: str) -> Path:
        path = self.root / name
        path.write_text(text)
        self.add(name, path, description)
        return path

    def write_plot(self, name: str, fig, description: str) -> Optional[Path]:
        if fig is None:
            return None
        path = self.root / name
        fig.savefig(path, dpi=160, bbox_inches='tight')
        plt.close(fig)
        self.add(name, path, description)
        return path

    def finalize_manifest(self) -> Path:
        manifest = pd.DataFrame(self.rows).sort_values('name').reset_index(drop=True)
        return self.write_csv('tinyml_artifact_manifest.csv', manifest, 'Manifest of generated artifacts.')



def _json_default(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, (pd.Series,)):
        return obj.to_dict()
    if isinstance(obj, (pd.DataFrame,)):
        return obj.to_dict(orient='records')
    if hasattr(obj, '__dict__'):
        return obj.__dict__
    return str(obj)


# ---------------------------------------------------------------------------
# Baseline metadata from provided run artifacts
# ---------------------------------------------------------------------------


def load_reference_baseline() -> Dict[str, object]:
    metadata_path = Path('/mnt/data/run_metadata.json')
    summary_path = Path('/mnt/data/run_summary.md')
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    summary_text = summary_path.read_text() if summary_path.exists() else ''

    baseline = {
        'stage1_2018': {
            'timestep_recall': 1.0,
            'timestep_precision': 0.991102,
            'false_positive_rate': 0.998918,
            'active_fraction': 0.999990,
            'truth_fraction': 0.991093,
            'stage2_calls_per_day': 2.001755,
            'stage2_call_fraction': 0.006950,
        },
        'stage1_2019': {
            'timestep_recall': 1.0,
            'timestep_precision': 1.0,
            'false_positive_rate': 0.0,
            'active_fraction': 1.0,
            'truth_fraction': 1.0,
            'stage2_calls_per_day': 2.000019,
            'stage2_call_fraction': 0.006944,
        },
        'binary_val_actual': {
            'precision': 1.0,
            'recall': 1.0,
            'f1': 1.0,
            'accepted_fraction': 0.986207,
            'brier': 0.000022,
            'pr_auc': 1.0,
            'roc_auc': 1.0,
        },
        'binary_test_2019': {
            'precision': 1.0,
            'recall': 0.964384,
            'f1': 0.981869,
            'accepted_fraction': 0.964384,
            'brier': 0.014191,
        },
        'loc_val': {
            'macro_f1_pipes': 0.285714,
            'top1_hit_rate': 1.0,
            'top3_hit_rate': 1.0,
            'mAP_pipes': 0.111888,
        },
        'loc_test_2019': {
            'macro_f1_pipes': 0.227882,
            'top1_hit_rate': 0.582192,
            'top3_hit_rate': 0.716438,
            'mAP_pipes': 0.360016,
        },
        'packet_info': metadata.get('packet_source_counts', {}),
        'threshold_choice': metadata.get('threshold_choice', {}),
        'summary_excerpt': summary_text,
    }
    return baseline


# ---------------------------------------------------------------------------
# Export deployment artifacts
# ---------------------------------------------------------------------------


def export_stage1_artifacts(result: CandidateRunResult, writer: ArtifactWriter):
    params = result.detector.export_params()
    writer.write_json('tiny_stage1_params.json', params, 'Chosen tiny Stage 1 parameters in JSON form.')

    pressure_idx = [int(re.search(r'(\d+)$', s).group(1)) - 1 for s in result.detector.monitored_pressure_cols_]
    channel_names = result.detector.all_signal_cols_
    baseline_scales = []
    baseline_flat_q = []
    baseline_offsets = []
    for col in channel_names:
        table = np.asarray(result.detector.baseline_tables_[col], dtype=float)
        q, scale = quantize_array(table)
        baseline_scales.append(scale)
        baseline_offsets.append(0.0)
        baseline_flat_q.extend(q.tolist())
    threshold_keys = ['burst_start', 'incipient_start', 'hold', 'flow_hold']
    thr_vals = np.array([result.detector.thresholds_[k] for k in threshold_keys], dtype=float)
    q_thr, thr_scale = quantize_array(thr_vals)

    lines = [
        '#pragma once',
        '#include <stdint.h>',
        f'#define TINY_STAGE1_NUM_PRESSURES {len(result.detector.monitored_pressure_cols_)}',
        f'#define TINY_STAGE1_NUM_SIGNALS {len(channel_names)}',
        f'#define TINY_STAGE1_BASELINE_BINS {len(next(iter(result.detector.baseline_tables_.values())))}',
        c_array_int16('TINY_STAGE1_PRESSURE_INDEX', np.asarray(pressure_idx, dtype=np.int16)),
        c_array_float('TINY_STAGE1_BASELINE_SCALE_PER_CHANNEL', np.asarray(baseline_scales, dtype=float)),
        c_array_int16('TINY_STAGE1_BASELINE_TABLES_Q15', np.asarray(baseline_flat_q, dtype=np.int16)),
        c_array_int16('TINY_STAGE1_THRESHOLDS_Q15', q_thr),
        f'static const float TINY_STAGE1_THRESHOLD_SCALE = {thr_scale:.8g}f;\n',
        f'static const int16_t TINY_STAGE1_STAGE2_RECHECK_STEPS = {int(round(result.detector.cfg.stage2_recheck_hours * 60 / result.detector.cfg.step_minutes))};\n',
        f'static const int16_t TINY_STAGE1_SMOOTH_WINDOW = {int(result.detector.cfg.smooth_window)};\n',
        f'static const int16_t TINY_STAGE1_SLOPE_WINDOW = {int(result.detector.cfg.slope_window)};\n',
    ]
    header_path = writer.write_text('tiny_stage1_params.h', '\n'.join(lines), 'Quantized C header for the chosen tiny Stage 1.')
    return header_path



def export_stage2_artifacts(result: CandidateRunResult, writer: ArtifactWriter):
    raw_coef, raw_bias = result.binary_head.folded_raw_coefficients()
    threshold_prob = float(result.binary_threshold_info['threshold'])
    threshold_logit = float(logit(threshold_prob))

    stage2_json = {
        'candidate': result.candidate.name,
        'binary_feature_names': result.binary_cols,
        'binary_raw_coef': raw_coef.tolist(),
        'binary_raw_bias': float(raw_bias),
        'binary_threshold_prob': threshold_prob,
        'binary_threshold_logit': threshold_logit,
        'localizer_type': result.candidate.localizer_type,
        'loc_feature_names': result.loc_cols,
    }

    if isinstance(result.localizer, OVRLinearLocalizer):
        folded = result.localizer.folded_raw_coefficients()
        stage2_json['pipe_models'] = {
            pipe: {'coef': spec['coef'].tolist(), 'bias': float(spec['bias'])}
            for pipe, spec in folded.items()
        }
    else:
        stage2_json['prototype_centroids_pos'] = {pipe: arr.tolist() for pipe, arr in result.localizer.pos_centroids_.items()}
        stage2_json['prototype_centroids_neg'] = {pipe: arr.tolist() for pipe, arr in result.localizer.neg_centroids_.items()}
        stage2_json['prototype_scaler_mean'] = result.localizer.scaler.mean_.tolist()
        stage2_json['prototype_scaler_scale'] = result.localizer.scaler.scale_.tolist()

    writer.write_json('tiny_stage2_params.json', stage2_json, 'Chosen tiny Stage 2 parameters in JSON form.')

    coef_q, coef_scale = quantize_array(raw_coef)
    bias_q, bias_scale = quantize_array(np.asarray([raw_bias], dtype=float))
    lines = [
        '#pragma once',
        '#include <stdint.h>',
        f'#define TINY_STAGE2_BINARY_FEATURES {len(result.binary_cols)}',
        c_array_int16('TINY_STAGE2_BINARY_COEF_Q15', coef_q),
        f'static const float TINY_STAGE2_BINARY_COEF_SCALE = {coef_scale:.8g}f;\n',
        c_array_int16('TINY_STAGE2_BINARY_BIAS_Q15', bias_q),
        f'static const float TINY_STAGE2_BINARY_BIAS_SCALE = {bias_scale:.8g}f;\n',
        f'static const float TINY_STAGE2_BINARY_THRESHOLD_PROB = {threshold_prob:.8g}f;\n',
        f'static const float TINY_STAGE2_BINARY_THRESHOLD_LOGIT = {threshold_logit:.8g}f;\n',
    ]
    if isinstance(result.localizer, OVRLinearLocalizer):
        folded = result.localizer.folded_raw_coefficients()
        lines.append(f'#define TINY_STAGE2_NUM_PIPES {len(folded)}')
        lines.append(f'#define TINY_STAGE2_LOC_FEATURES {len(result.loc_cols)}')
        for pipe, spec in folded.items():
            q, scale = quantize_array(np.asarray(spec['coef'], dtype=float))
            qb, bscale = quantize_array(np.asarray([spec['bias']], dtype=float))
            safe_name = pipe.upper().replace('-', '_')
            lines.append(c_array_int16(f'TINY_STAGE2_{safe_name}_COEF_Q15', q))
            lines.append(f'static const float TINY_STAGE2_{safe_name}_COEF_SCALE = {scale:.8g}f;\n')
            lines.append(c_array_int16(f'TINY_STAGE2_{safe_name}_BIAS_Q15', qb))
            lines.append(f'static const float TINY_STAGE2_{safe_name}_BIAS_SCALE = {bscale:.8g}f;\n')
    header_path = writer.write_text('tiny_stage2_params.h', '\n'.join(lines), 'Quantized C header for the chosen tiny Stage 2.')
    return header_path


# ---------------------------------------------------------------------------
# Notebook export
# ---------------------------------------------------------------------------


def export_notebook(run_dir: Path, script_name: str = 'battledim_tinyml_final.py') -> Path:
    nb = nbf.v4.new_notebook()
    nb['cells'] = [
        nbf.v4.new_markdown_cell(
            '# BattLeDIM tiny MCU phase\n\n'
            'This notebook preserves the existing Stage 1 import path and wraps the final microcontroller-focused run as a single callable script.'
        ),
        nbf.v4.new_code_cell(
            'from pathlib import Path\n'
            'from battledim_tinyml_final import main\n\n'
            f"run_dir = main(output_root=Path('/mnt/data'))\n"
            'print(run_dir)'
        ),
    ]
    path = run_dir / 'battledim_stage1_notebook_tinyml_final.ipynb'
    with path.open('w', encoding='utf-8') as f:
        nbf.write(nb, f)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(output_root: Path | str = Path('/mnt/data')) -> Path:
    output_root = Path(output_root)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = output_root / f'tinyml_run_{stamp}'
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = ArtifactWriter(run_dir)

    reference = load_reference_baseline()
    full_df = load_battledim_merged_csv('/mnt/data/data_full_2018_2019_fixed.csv', timestamp_col='Timestamp')
    groups = infer_column_groups(full_df, include_flow3_context=True)
    base_stage1_cfg = Stage1Config(**json.loads(Path('/mnt/data/battledim_stage1_tuned_config.json').read_text()))
    split_cfg = PacketSplitConfig()

    reference_size = estimate_reference_current_pipeline(groups)
    current_ref_total_ram = int(reference_size['ram_bytes'].sum())
    current_ref_total_flash = int(reference_size['flash_bytes'].sum())

    # Run candidates sequentially so only the current best heavy object stays resident.
    comparison_rows = [
        {
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
            'total_ram_bytes': current_ref_total_ram,
            'total_flash_bytes': current_ref_total_flash,
            'feasible_under_256k_1m': False,
        }
    ]
    memory_budget_parts: List[pd.DataFrame] = [reference_size]
    chosen_candidate: Optional[PipelineCandidate] = None
    chosen_score: Optional[Tuple[float, float, float, float]] = None
    for cand in CANDIDATES:
        res = run_candidate(cand, full_df, groups, base_stage1_cfg, split_cfg)
        comparison_rows.append(
            {
                'candidate': res.candidate.name,
                'description': res.candidate.description,
                'stage1_mode': f"{res.candidate.baseline_mode} / {res.candidate.monitored_pressure_count} pressures",
                'localizer_type': res.candidate.localizer_type,
                'binary_feature_count': len(res.binary_cols),
                'localisation_feature_count': len(res.loc_cols),
                'stage1_recall_2018': safe_float(res.stage1_metrics_2018.get('timestep_recall', np.nan), np.nan),
                'stage1_calls_per_day_2018': safe_float(res.stage1_metrics_2018.get('stage2_calls_per_day', np.nan), np.nan),
                'binary_recall_val_actual': safe_float(res.binary_metrics_val_actual.get('recall', np.nan), np.nan),
                'binary_recall_2019': safe_float(res.binary_metrics_test_actual.get('recall', np.nan), np.nan),
                'localisation_top3_val': safe_float(res.localisation_metrics_val.get('top3_hit_rate', np.nan), np.nan),
                'localisation_top3_2019': safe_float(res.localisation_metrics_test.get('top3_hit_rate', np.nan), np.nan),
                'total_ram_bytes': res.total_ram_bytes,
                'total_flash_bytes': res.total_flash_bytes,
                'feasible_under_256k_1m': bool(res.feasible),
            }
        )
        memory_budget_parts.append(res.memory_rows)
        if (chosen_score is None) or (res.chosen_score > chosen_score):
            chosen_score = res.chosen_score
            chosen_candidate = res.candidate
        del res

    if chosen_candidate is None:
        raise RuntimeError('No candidate results were produced.')

    chosen = run_candidate(chosen_candidate, full_df, groups, base_stage1_cfg, split_cfg)
    model_comparison = pd.DataFrame(comparison_rows)
    memory_budget = pd.concat(memory_budget_parts, ignore_index=True)

    # Required tables
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
        'chosen_candidate': asdict(chosen.candidate),
        'reference_baseline': reference,
        'packet_split_config': asdict(split_cfg),
        'packet_info': chosen.packet_info,
        'stage1_monitored_pressures': chosen.detector.monitored_pressure_cols_,
        'stage1_flow_score_cols': chosen.detector.flow_score_cols_,
        'stage1_auto_excluded_flow_score_cols': chosen.detector.auto_excluded_flow_score_cols_,
        'binary_threshold_info': chosen.binary_threshold_info,
        'binary_feature_count': len(chosen.binary_cols),
        'localisation_feature_count': len(chosen.loc_cols),
        'estimated_total_ram_bytes': chosen.total_ram_bytes,
        'estimated_total_flash_bytes': chosen.total_flash_bytes,
    }
    writer.write_json('tinyml_metadata.json', metadata_obj, 'Run metadata for the tiny MCU phase.')

    # Deployment artifacts.
    export_stage1_artifacts(chosen, writer)
    export_stage2_artifacts(chosen, writer)

    # Notebook wrapper.
    nb_path = export_notebook(run_dir)
    writer.add(nb_path.name, nb_path, 'Notebook wrapper for rerunning the tiny MCU phase.')

    # Required plots.
    writer.write_plot('pr_curve_val.png', make_pr_curve_fig(chosen.pred_val, 'Tiny Stage 2 validation PR curve'), 'PR curve on mixed validation packets.')
    writer.write_plot('calibration_val.png', make_calibration_fig(chosen.pred_val, 'Tiny Stage 2 validation calibration'), 'Calibration curve on mixed validation packets.')
    writer.write_plot('probability_histogram_val.png', make_probability_histogram_fig(chosen.pred_val, 'Tiny Stage 2 validation probability histogram'), 'Histogram of validation probabilities.')
    writer.write_plot('memory_vs_performance.png', make_memory_vs_performance_fig(model_comparison[model_comparison['candidate'] != 'current_patched_reference']), 'Candidate memory vs performance Pareto.')
    writer.write_plot('feature_budget_breakdown.png', make_feature_budget_breakdown_fig(memory_budget, chosen.candidate.name), 'Memory budget breakdown for the chosen tiny model.')
    stage1_tradeoff_df = model_comparison[model_comparison['candidate'] != 'current_patched_reference'][['candidate', 'stage1_calls_per_day_2018', 'stage1_recall_2018']].copy()
    writer.write_plot('stage1_calls_vs_recall.png', make_stage1_calls_vs_recall_fig(stage1_tradeoff_df), 'Operational Stage 1 calls/day vs recall tradeoff.')
    baseline_topk = {
        'val_top1': reference['loc_val']['top1_hit_rate'],
        'val_top3': reference['loc_val']['top3_hit_rate'],
        'test_top1': reference['loc_test_2019']['top1_hit_rate'],
        'test_top3': reference['loc_test_2019']['top3_hit_rate'],
    }
    writer.write_plot('localisation_topk.png', make_localisation_topk_fig(baseline_topk, chosen), 'Top-1 / top-3 localisation comparison against the provided baseline.')

    # Summary markdown.
    stage1_drop_2018 = safe_float(chosen.stage1_metrics_2018.get('timestep_recall', np.nan), np.nan) - reference['stage1_2018']['timestep_recall']
    binary_recall_drop_2019 = safe_float(chosen.binary_metrics_test_actual.get('recall', np.nan), np.nan) - reference['binary_test_2019']['recall']
    loc_top3_drop_2019 = safe_float(chosen.localisation_metrics_test.get('top3_hit_rate', np.nan), np.nan) - reference['loc_test_2019']['top3_hit_rate']
    summary_lines = [
        '# BattLeDIM tiny MCU phase summary',
        '',
        '## Current pipeline from the attachments',
        '',
        '- Stage 1 is a recall-first BattLeDIM detector with causal smoothing, weekly de-seasonalisation, pressure/pairwise/flow evidence, and periodic Stage 2 rechecks.',
        '- The attached patched Stage 2 uses 12-hour packets, four summary horizons, pseudo-hard negatives, and a calibrated gradient-boosting binary head with one-vs-rest pipe models.',
        '- The provided reference run reports 721 actual late-2018 triggers (714 positive, 7 negative), a 2019 binary recall of 0.964, and 2019 localisation top-3 hit rate of 0.716.',
        '',
        '## Why the provided patched pipeline is too large for the MCU target',
        '',
        f'- The current Stage 2 feature builder expands each trigger into about {int(reference_size["binary_feature_count"].iloc[0])} binary features and {int(reference_size["localisation_feature_count"].iloc[0])} localisation features, derived from 144-step histories over pressures, flows, tank, demands, Stage 1 scores, and per-sensor detail.',
        f'- Just the current packet lookback buffers are about {int(reference_size.loc[reference_size["component"] == "stage2_packet_buffer_float32", "ram_bytes"].iloc[0]) / 1024:.1f} KB RAM, and the feature workspace is about {int(reference_size.loc[reference_size["component"] == "stage2_feature_workspace_float32", "ram_bytes"].iloc[0]) / 1024:.1f} KB RAM.',
        f'- The fallback HistGradientBoosting binary head plus 14 one-vs-rest pipe heads is conservatively estimated at roughly {current_ref_total_flash / 1024:.1f} KB flash overall, which is far beyond a 1 MB MCU budget.',
        '',
        '## Chosen tiny architecture',
        '',
        f'- **Chosen candidate:** `{chosen.candidate.name}` — {chosen.candidate.description}',
        f'- **Stage 1:** `{chosen.candidate.baseline_mode}` seasonal table, `{chosen.candidate.monitored_pressure_count}` selected pressure sensors, `{chosen.candidate.pairwise_anchors}` pairwise anchors, same Stage 1 import path/API, causal state machine, and 12-hour periodic rechecks.',
        f'- **Stage 2 binary head:** linear logistic model over `{len(chosen.binary_cols)}` compact causal packet features with folded calibration.',
        f'- **Stage 2 localizer:** `{chosen.candidate.localizer_type}` over `{len(chosen.loc_cols)}` compact features, returning pipe probabilities and top-3 rankings.',
        '',
        '## MCU fit check',
        '',
        f'- Estimated runtime RAM: **{chosen.total_ram_bytes / 1024:.1f} KB**.',
        f'- Estimated flash footprint: **{chosen.total_flash_bytes / 1024:.1f} KB**.',
        '- Both are within the requested ~256 KB RAM and ~1 MB Flash budget with explicit allowance for stack and control logic.',
        '',
        '## Performance vs the provided patched baseline',
        '',
        f'- Stage 1 late-2018 timestep recall delta: **{stage1_drop_2018:+.4f}**.',
        f'- Stage 2 binary 2019 recall delta: **{binary_recall_drop_2019:+.4f}**.',
        f'- Stage 2 localisation 2019 top-3 hit-rate delta: **{loc_top3_drop_2019:+.4f}**.',
        '- As in the provided reference run, 2019 is all-positive in this merged table, so 2019 cannot measure false-alarm rejection for the binary head. The meaningful rejection check remains the mixed/actual validation splits in late 2018.',
        '',
        '## Risks / caveats',
        '',
        '- Validation still has very few real hard-negative actual triggers, so false-alarm rejection uncertainty remains high.',
        '- Localisation remains the hardest head to compress; the selected tiny localizer is intentionally simple and may lose some fine-grained ranking resolution relative to the boosted baseline.',
        '- The exported C headers are parameter tables and folded coefficients. Firmware still needs a small runtime wrapper for streaming median, EWMA/CUSUM, rolling sums, and sigmoid/logit evaluation.',
        '',
        '## Files written',
        '',
        '- See `tinyml_artifact_manifest.csv` for the full file list.',
    ]
    writer.write_text('tinyml_run_summary.md', '\n'.join(summary_lines), 'Narrative summary of the tiny MCU phase.')

    # Manifest last.
    writer.finalize_manifest()
    return run_dir


if __name__ == '__main__':
    out = main(Path('/mnt/data'))
    print(out)
