from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _natural_key(text: str):
    return [int(tok) if tok.isdigit() else tok for tok in re.split(r'(\d+)', str(text))]


def _infer_column_groups(df: pd.DataFrame) -> Dict[str, List[str]]:
    pressure_cols = sorted([c for c in df.columns if re.fullmatch(r'press_\d+', c)], key=_natural_key)
    flow_cols = sorted([c for c in df.columns if re.fullmatch(r'flow_\d+', c)], key=_natural_key)
    tank_cols = [c for c in ['T1'] if c in df.columns]
    demand_cols = sorted([c for c in df.columns if re.fullmatch(r'n\d+', c)], key=_natural_key)
    pipe_cols = sorted([c for c in df.columns if re.fullmatch(r'pipe_\d+_bin', c)], key=_natural_key)
    label_cols = [c for c in ['leak_mag_total', 'leak_count', 'leak_binary'] if c in df.columns]
    return {
        'pressure_cols': pressure_cols,
        'flow_cols': flow_cols,
        'tank_cols': tank_cols,
        'demand_cols': demand_cols,
        'pipe_cols': pipe_cols,
        'label_cols': label_cols,
    }


def _robust_scale(arr: np.ndarray, min_scale: float = 1e-3) -> float:
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 1.0
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < min_scale:
        std = float(np.std(arr))
        scale = std if np.isfinite(std) and std >= min_scale else min_scale
    return float(scale)


def _safe_quantile(arr: np.ndarray, q: float, default: float = 0.0) -> float:
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float(default)
    return float(np.quantile(arr, q))


def _topk_mean(matrix: np.ndarray, k: int) -> np.ndarray:
    if matrix.ndim != 2:
        raise ValueError('matrix must be 2D')
    if matrix.shape[1] == 0:
        return np.zeros(matrix.shape[0], dtype=float)
    k = int(max(1, min(k, matrix.shape[1])))
    part = np.partition(matrix, -k, axis=1)[:, -k:]
    return np.nanmean(part, axis=1)


def _rolling_sum_bool(series: pd.Series, window: int) -> pd.Series:
    return series.astype(int).rolling(window=window, min_periods=1).sum()


def _hour_bin(index: pd.DatetimeIndex) -> np.ndarray:
    return (index.dayofweek.to_numpy(dtype=int) * 24 + index.hour.to_numpy(dtype=int)).astype(int)


def _day_hour_bin(index: pd.DatetimeIndex) -> np.ndarray:
    return index.hour.to_numpy(dtype=int).astype(int)


def _week_5min_bin(index: pd.DatetimeIndex, step_minutes: int = 5) -> np.ndarray:
    step = max(1, int(step_minutes))
    return (
        index.dayofweek.to_numpy(dtype=int) * (24 * 60 // step)
        + index.hour.to_numpy(dtype=int) * (60 // step)
        + (index.minute.to_numpy(dtype=int) // step)
    ).astype(int)


def _timebins(index: pd.DatetimeIndex, mode: str, step_minutes: int, weekly_period: int) -> Tuple[np.ndarray, int]:
    mode = str(mode)
    if mode == 'none':
        return np.zeros(len(index), dtype=int), 1
    if mode == 'daily_hour':
        return _day_hour_bin(index), 24
    if mode == 'weekly_hour':
        return _hour_bin(index), 168
    if mode == 'legacy_weekly_5min':
        return _week_5min_bin(index, step_minutes=step_minutes), int(weekly_period)
    if mode == 'auto':
        return _week_5min_bin(index, step_minutes=step_minutes), int(weekly_period)
    raise ValueError(f'Unsupported baseline mode: {mode}')


def _group_median_lookup(index: pd.DatetimeIndex, values: pd.Series, mode: str, step_minutes: int, weekly_period: int) -> Tuple[np.ndarray, np.ndarray]:
    bins, n_bins = _timebins(index, mode=mode, step_minutes=step_minutes, weekly_period=weekly_period)
    table = np.full(n_bins, np.nan, dtype=float)
    arr = values.to_numpy(dtype=float, copy=False)
    for b in range(n_bins):
        mask = bins == b
        if np.any(mask):
            val = np.nanmedian(arr[mask])
            table[b] = float(val) if np.isfinite(val) else np.nan
    finite = np.isfinite(table)
    if not finite.all():
        fallback = float(np.nanmedian(arr)) if np.isfinite(np.nanmedian(arr)) else 0.0
        table[~finite] = fallback
    return bins, table


# ---------------------------------------------------------------------------
# Public API helpers
# ---------------------------------------------------------------------------


def load_battledim_merged_csv(path: str | Path, timestamp_col: str = 'Timestamp') -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, parse_dates=[timestamp_col])
    df = df.sort_values(timestamp_col).reset_index(drop=True)
    df = df.set_index(timestamp_col)
    df.index = pd.to_datetime(df.index)
    return df



def build_known_intervals_from_leak_columns(
    df: pd.DataFrame,
    pad_before: str | pd.Timedelta = '0min',
    pad_after: str | pd.Timedelta = '0min',
    min_active_samples: int = 1,
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    groups = _infer_column_groups(df)
    masks = []
    if 'leak_count' in df.columns:
        masks.append(df['leak_count'].fillna(0) > 0)
    if 'leak_mag_total' in df.columns:
        masks.append(df['leak_mag_total'].fillna(0) > 0)
    if groups['pipe_cols']:
        masks.append(df[groups['pipe_cols']].fillna(0).sum(axis=1) > 0)
    if not masks and 'leak_binary' in df.columns:
        masks.append(df['leak_binary'].fillna(0) > 0)
    if not masks:
        return []

    active = pd.concat(masks, axis=1).any(axis=1).astype(int)
    pad_before = pd.Timedelta(pad_before)
    pad_after = pd.Timedelta(pad_after)
    intervals: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    if active.empty:
        return intervals

    in_run = False
    start = None
    count = 0
    last_ts = None
    for ts, flag in active.items():
        if flag and not in_run:
            in_run = True
            start = ts
            count = 1
        elif flag and in_run:
            count += 1
        elif (not flag) and in_run:
            if start is not None and count >= int(min_active_samples):
                intervals.append((pd.Timestamp(start) - pad_before, pd.Timestamp(last_ts) + pad_after))
            in_run = False
            start = None
            count = 0
        last_ts = ts
    if in_run and start is not None and count >= int(min_active_samples):
        end_ts = active.index[-1]
        intervals.append((pd.Timestamp(start) - pad_before, pd.Timestamp(end_ts) + pad_after))
    return intervals



def _mask_from_known_intervals(index: pd.DatetimeIndex, known_intervals: Optional[Sequence[Tuple[pd.Timestamp, pd.Timestamp]]]) -> pd.Series:
    mask = pd.Series(False, index=index)
    if not known_intervals:
        return mask
    for start, end in known_intervals:
        mask.loc[(mask.index >= pd.Timestamp(start)) & (mask.index <= pd.Timestamp(end))] = True
    return mask



def evaluate_stage1_results(results: pd.DataFrame, truth: pd.Series, triggers: Optional[pd.DataFrame] = None) -> pd.Series:
    truth = pd.Series(truth).reindex(results.index).fillna(0).astype(int)
    pred = results['leak_state'].reindex(results.index).fillna(0).astype(int)

    tp = int(((truth == 1) & (pred == 1)).sum())
    fp = int(((truth == 0) & (pred == 1)).sum())
    fn = int(((truth == 1) & (pred == 0)).sum())
    tn = int(((truth == 0) & (pred == 0)).sum())

    timestep_recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    timestep_precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    active_fraction = float(pred.mean()) if len(pred) else np.nan
    truth_fraction = float(truth.mean()) if len(truth) else np.nan

    def count_events(binary: pd.Series) -> Tuple[int, List[Tuple[pd.Timestamp, pd.Timestamp]]]:
        binary = binary.astype(int)
        intervals = []
        in_run = False
        start = None
        prev_ts = None
        for ts, val in binary.items():
            if val == 1 and not in_run:
                start = ts
                in_run = True
            elif val == 0 and in_run:
                intervals.append((pd.Timestamp(start), pd.Timestamp(prev_ts)))
                start = None
                in_run = False
            prev_ts = ts
        if in_run and start is not None:
            intervals.append((pd.Timestamp(start), pd.Timestamp(binary.index[-1])))
        return len(intervals), intervals

    n_truth_events, truth_intervals = count_events(truth)
    n_detected_events, pred_intervals = count_events(pred)
    matched = 0
    for t0, t1 in truth_intervals:
        overlap = any((p0 <= t1) and (p1 >= t0) for p0, p1 in pred_intervals)
        matched += int(overlap)
    event_recall = matched / n_truth_events if n_truth_events > 0 else np.nan

    out = {
        'timestep_recall': float(timestep_recall) if np.isfinite(timestep_recall) else np.nan,
        'timestep_precision': float(timestep_precision) if np.isfinite(timestep_precision) else np.nan,
        'false_positive_rate': float(false_positive_rate) if np.isfinite(false_positive_rate) else np.nan,
        'active_fraction': float(active_fraction) if np.isfinite(active_fraction) else np.nan,
        'truth_fraction': float(truth_fraction) if np.isfinite(truth_fraction) else np.nan,
        'n_truth_events': float(n_truth_events),
        'n_detected_events': float(n_detected_events),
        'event_recall': float(event_recall) if np.isfinite(event_recall) else np.nan,
    }
    if triggers is not None and len(results) > 0:
        tr = triggers.copy()
        if not tr.empty and 'timestamp' in tr.columns:
            tr['timestamp'] = pd.to_datetime(tr['timestamp'])
            tr = tr[tr['timestamp'].isin(results.index)]
            calls_per_day = tr.groupby(tr['timestamp'].dt.floor('D')).size().mean() if len(tr) else 0.0
            call_fraction = len(tr) / len(results)
        else:
            calls_per_day = 0.0
            call_fraction = 0.0
        out['stage2_calls_per_day'] = float(calls_per_day)
        out['stage2_call_fraction'] = float(call_fraction)
    return pd.Series(out, dtype=float)


# ---------------------------------------------------------------------------
# Detector config
# ---------------------------------------------------------------------------


@dataclass
class Stage1Config:
    smooth_window: int = 3
    step_minutes: int = 5
    weekly_period: int = 2016
    top_n_anchor_pressures: int = 6
    pairwise_anchors_per_sensor: int = 4
    min_top_k: int = 3
    top_k_fraction: float = 0.15
    top_sensors_output: int = 5
    huber_delta: float = 1.5
    ridge_lambda: float = 0.001
    max_irls_iter: int = 30
    trim_quantile: float = 0.97
    ewma_alpha: float = 0.15
    slope_window: int = 24
    sensor_vote_threshold: float = 0.9
    start_sensor_votes: int = 2
    hold_sensor_votes: int = 1
    weight_pressure: float = 1.0
    weight_pairwise: float = 0.8
    weight_flow: float = 0.4
    weight_night_flow: float = 0.25
    weight_slope: float = 0.8
    cusum_allowance: float = 0.5
    cusum_decay: float = 0.85
    burst_start_quantile: float = 0.99
    incipient_start_quantile: float = 0.97
    hold_quantile: float = 0.90
    flow_hold_quantile: float = 0.88
    burst_confirm_window: int = 3
    burst_confirm_needed: int = 2
    incipient_confirm_window: int = 12
    incipient_confirm_needed: int = 6
    quiet_end_samples: int = 72
    merge_gap_samples: int = 144
    night_start_hour: int = 1
    night_end_hour: int = 4
    stage2_recheck_hours: float = 12.0
    min_sensor_shift_minutes: int = 720
    sensor_shift_jaccard_threshold: float = 0.2
    flow_score_exclude_cols: List[str] = field(default_factory=list)
    auto_exclude_binary_like_flows: bool = True
    binary_like_round_decimals: int = 2
    binary_like_edge_fraction_threshold: float = 0.85
    binary_like_unique_ratio_threshold: float = 0.02
    sensor_areas: Dict[str, str] = field(default_factory=dict)
    baseline_mode: str = 'auto'  # auto, legacy_weekly_5min, weekly_hour, daily_hour, none
    monitored_pressure_count: Optional[int] = None  # None => all
    compact_pairwise_anchors: Optional[int] = None

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Detector implementation
# ---------------------------------------------------------------------------


class BattLeDIMStage1:
    """
    Recall-first detector.

    This implementation is deliberately deployment-aware:
    - `baseline_mode='legacy_weekly_5min'` approximates the original weekly 5-minute baseline idea.
    - `baseline_mode='weekly_hour'|'daily_hour'|'none'` enables compact MCU-style variants.
    - `monitored_pressure_count` can reduce the pressure subset for deployment.
    """

    def __init__(self, config: Stage1Config):
        self.cfg = config
        self.groups_: Dict[str, List[str]] = {}
        self.is_fitted_: bool = False
        self.pressure_cols_: List[str] = []
        self.flow_cols_: List[str] = []
        self.tank_cols_: List[str] = []
        self.all_signal_cols_: List[str] = []
        self.monitored_pressure_cols_: List[str] = []
        self.anchor_pressure_cols_: List[str] = []
        self.flow_score_cols_: List[str] = []
        self.flow_context_cols_: List[str] = []
        self.auto_excluded_flow_score_cols_: List[str] = []
        self.baseline_mode_: str = 'auto'
        self.baseline_tables_: Dict[str, np.ndarray] = {}
        self.scale_: Dict[str, float] = {}
        self.pairwise_params_: Dict[str, List[Tuple[str, float, float, float]]] = {}
        self.thresholds_: Dict[str, float] = {}
        self.train_metrics_: Dict[str, float] = {}
        self.last_internal_: Dict[str, pd.DataFrame | pd.Series | np.ndarray] = {}

    # ----------------------------
    # Fitting helpers
    # ----------------------------

    def _resolve_baseline_mode(self) -> str:
        mode = str(self.cfg.baseline_mode)
        if mode != 'auto':
            return mode
        return 'legacy_weekly_5min' if int(self.cfg.weekly_period) == 2016 else 'weekly_hour'

    def _auto_exclude_binary_like_flows(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        keep = []
        excluded = []
        for col in self.flow_cols_:
            if col in set(self.cfg.flow_score_exclude_cols):
                excluded.append(col)
                continue
            if not self.cfg.auto_exclude_binary_like_flows:
                keep.append(col)
                continue
            s = df[col].dropna().round(int(self.cfg.binary_like_round_decimals))
            if len(s) == 0:
                excluded.append(col)
                continue
            unique_ratio = s.nunique() / max(1, len(s))
            mn = float(s.min())
            mx = float(s.max())
            if mx > mn:
                edge_fraction = float(((s == mn) | (s == mx)).mean())
            else:
                edge_fraction = 1.0
            if (
                unique_ratio <= float(self.cfg.binary_like_unique_ratio_threshold)
                and edge_fraction >= float(self.cfg.binary_like_edge_fraction_threshold)
            ):
                excluded.append(col)
            else:
                keep.append(col)
        return keep, excluded

    def _select_pressure_subset(self, smooth_train: pd.DataFrame) -> Tuple[List[str], List[str]]:
        pressure_cols = self.pressure_cols_
        n_total = len(pressure_cols)
        target = self.cfg.monitored_pressure_count
        if target is None or int(target) >= n_total:
            monitored = list(pressure_cols)
        else:
            target = max(1, int(target))
            arr = smooth_train[pressure_cols].copy()
            corr = arr.corr().fillna(0.0).abs()
            corr_arr = corr.to_numpy(copy=True)
            np.fill_diagonal(corr_arr, 0.0)
            corr = pd.DataFrame(corr_arr, index=corr.index, columns=corr.columns)
            centrality = corr.mean(axis=1).sort_values(ascending=False)
            monitored = centrality.head(target).index.tolist()
        anchor_target = max(1, min(int(self.cfg.top_n_anchor_pressures), len(monitored)))
        if len(monitored) == 1:
            anchors = monitored[:]
        else:
            arr = smooth_train[monitored].copy()
            corr = arr.corr().fillna(0.0).abs()
            corr_arr = corr.to_numpy(copy=True)
            np.fill_diagonal(corr_arr, 0.0)
            corr = pd.DataFrame(corr_arr, index=corr.index, columns=corr.columns)
            centrality = corr.mean(axis=1).sort_values(ascending=False)
            anchors = centrality.head(anchor_target).index.tolist()
        return monitored, anchors

    def _fit_pairwise_models(self, smooth_train: pd.DataFrame) -> Dict[str, List[Tuple[str, float, float, float]]]:
        params: Dict[str, List[Tuple[str, float, float, float]]] = {}
        n_anchors = self.cfg.compact_pairwise_anchors if self.cfg.compact_pairwise_anchors is not None else self.cfg.pairwise_anchors_per_sensor
        n_anchors = max(0, int(n_anchors))
        if n_anchors == 0 or len(self.monitored_pressure_cols_) <= 1:
            return {c: [] for c in self.monitored_pressure_cols_}

        arr = smooth_train[self.monitored_pressure_cols_].copy()
        corr = arr.corr().fillna(0.0).abs()
        corr_arr = corr.to_numpy(copy=True)
        np.fill_diagonal(corr_arr, 0.0)
        corr = pd.DataFrame(corr_arr, index=corr.index, columns=corr.columns)
        anchor_pool = list(self.anchor_pressure_cols_)
        if not anchor_pool:
            anchor_pool = list(self.monitored_pressure_cols_)

        for sensor in self.monitored_pressure_cols_:
            choices = [c for c in anchor_pool if c != sensor]
            if not choices:
                params[sensor] = []
                continue
            ranked = corr.loc[sensor, choices].sort_values(ascending=False)
            chosen = ranked.head(min(n_anchors, len(ranked))).index.tolist()
            rows: List[Tuple[str, float, float, float]] = []
            y = arr[sensor].to_numpy(dtype=float, copy=False)
            for anchor in chosen:
                x = arr[anchor].to_numpy(dtype=float, copy=False)
                mask = np.isfinite(x) & np.isfinite(y)
                if mask.sum() < 8:
                    continue
                xm = x[mask].mean()
                ym = y[mask].mean()
                xv = x[mask] - xm
                yv = y[mask] - ym
                denom = float(np.dot(xv, xv) + self.cfg.ridge_lambda)
                if denom <= 0:
                    slope = 0.0
                else:
                    slope = float(np.dot(xv, yv) / denom)
                intercept = float(ym - slope * xm)
                resid = y[mask] - (slope * x[mask] + intercept)
                scale = _robust_scale(resid)
                rows.append((anchor, slope, intercept, scale))
            params[sensor] = rows
        return params

    def _compute_preprocessed(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, pd.Series], Dict[str, pd.Series]]:
        relevant_cols = sorted(set(self.monitored_pressure_cols_ + self.flow_score_cols_ + self.flow_context_cols_ + self.tank_cols_), key=_natural_key)
        smooth = df[relevant_cols].astype(float).rolling(window=int(self.cfg.smooth_window), min_periods=1).median()
        bins, _ = _timebins(df.index, mode=self.baseline_mode_, step_minutes=int(self.cfg.step_minutes), weekly_period=int(self.cfg.weekly_period))
        residuals: Dict[str, pd.Series] = {}
        zscores: Dict[str, pd.Series] = {}
        pairwise_scores: Dict[str, pd.Series] = {}
        for col in relevant_cols:
            baseline = self.baseline_tables_[col][bins]
            resid = smooth[col] - baseline
            residuals[col] = resid.astype(float)
            zscores[col] = (resid.abs() / max(self.scale_.get(col, 1.0), 1e-6)).astype(float)
        for sensor in self.monitored_pressure_cols_:
            rows = self.pairwise_params_.get(sensor, [])
            if not rows:
                pairwise_scores[sensor] = pd.Series(0.0, index=df.index)
                continue
            vals = []
            y = smooth[sensor]
            for anchor, slope, intercept, scale in rows:
                pred = slope * smooth[anchor] + intercept
                val = (y - pred).abs() / max(scale, 1e-6)
                vals.append(val.to_numpy(dtype=float, copy=False))
            score = np.mean(np.vstack(vals), axis=0) if vals else np.zeros(len(df), dtype=float)
            pairwise_scores[sensor] = pd.Series(score, index=df.index)
        return smooth, {'timebins': bins}, residuals, zscores | pairwise_scores

    def _compute_score_frames(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame | pd.Series | np.ndarray]:
        if not self.is_fitted_:
            raise RuntimeError('Detector is not fitted.')
        smooth, meta, residuals, zplus = self._compute_preprocessed(df)
        idx = df.index

        pressure_z = pd.DataFrame({c: zplus[c] for c in self.monitored_pressure_cols_}, index=idx)
        pairwise_z = pd.DataFrame({c: zplus.get(c, pd.Series(0.0, index=idx)) for c in self.monitored_pressure_cols_}, index=idx)
        pressure_resid = pd.DataFrame({c: residuals[c] for c in self.monitored_pressure_cols_}, index=idx)
        residual_all = pd.DataFrame({c: residuals[c] for c in residuals.keys()}, index=idx)
        zscore_all = pd.DataFrame({c: zplus[c] for c in residuals.keys() if c in zplus}, index=idx)

        slope_raw = pressure_resid.diff(int(self.cfg.slope_window)).abs().fillna(0.0)
        for col in slope_raw.columns:
            slope_raw[col] = slope_raw[col] / max(self.scale_.get(col, 1.0), 1e-6)
        ewma_z = pressure_z.ewm(alpha=float(self.cfg.ewma_alpha), adjust=False).mean()

        if self.flow_score_cols_:
            flow_pos = {}
            for col in self.flow_score_cols_:
                resid = residuals[col]
                flow_pos[col] = (resid.clip(lower=0.0) / max(self.scale_.get(col, 1.0), 1e-6)).astype(float)
            flow_z = pd.DataFrame(flow_pos, index=idx)
        else:
            flow_z = pd.DataFrame(index=idx)

        night_mask = ((idx.hour >= int(self.cfg.night_start_hour)) & (idx.hour < int(self.cfg.night_end_hour))).astype(int)
        if len(flow_z.columns):
            flow_topk = _topk_mean(flow_z.to_numpy(dtype=float, copy=False), max(1, min(len(flow_z.columns), 1)))
            night_flow_topk = flow_topk * night_mask
        else:
            flow_topk = np.zeros(len(idx), dtype=float)
            night_flow_topk = np.zeros(len(idx), dtype=float)

        topk_pressure = max(1, min(len(self.monitored_pressure_cols_), max(int(self.cfg.min_top_k), int(math.ceil(len(self.monitored_pressure_cols_) * float(self.cfg.top_k_fraction))))))
        pressure_topk = _topk_mean(pressure_z.to_numpy(dtype=float, copy=False), topk_pressure) if len(pressure_z.columns) else np.zeros(len(idx), dtype=float)
        pairwise_topk = _topk_mean(pairwise_z.to_numpy(dtype=float, copy=False), topk_pressure) if len(pairwise_z.columns) else np.zeros(len(idx), dtype=float)
        ewma_topk = _topk_mean(ewma_z.to_numpy(dtype=float, copy=False), topk_pressure) if len(ewma_z.columns) else np.zeros(len(idx), dtype=float)
        slope_topk = _topk_mean(slope_raw.to_numpy(dtype=float, copy=False), topk_pressure) if len(slope_raw.columns) else np.zeros(len(idx), dtype=float)

        sensor_score = (
            float(self.cfg.weight_pressure) * pressure_z
            + float(self.cfg.weight_pairwise) * pairwise_z
            + float(self.cfg.weight_slope) * slope_raw
        )
        sensor_votes = (sensor_score > float(self.cfg.sensor_vote_threshold)).sum(axis=1).astype(int)

        burst_raw = (
            float(self.cfg.weight_pressure) * pressure_topk
            + float(self.cfg.weight_pairwise) * pairwise_topk
            + float(self.cfg.weight_flow) * flow_topk
        )
        score_incipient = (
            0.50 * ewma_topk
            + 0.50 * slope_topk
            + float(self.cfg.weight_night_flow) * night_flow_topk
        )
        burst_cusum = np.zeros(len(idx), dtype=float)
        for i in range(len(idx)):
            prev = burst_cusum[i - 1] if i > 0 else 0.0
            burst_cusum[i] = max(0.0, float(self.cfg.cusum_decay) * prev + burst_raw[i] - float(self.cfg.cusum_allowance))
        score_burst = burst_cusum.copy()
        score_final = np.maximum(score_burst, score_incipient)

        order = np.argsort(sensor_score.to_numpy(dtype=float, copy=False), axis=1)[:, ::-1] if len(sensor_score.columns) else np.empty((len(idx), 0), dtype=int)
        top_sensor_names: List[List[str]] = []
        sensor_cols = list(sensor_score.columns)
        for i in range(len(idx)):
            names = [sensor_cols[j] for j in order[i, :min(len(sensor_cols), int(self.cfg.top_sensors_output))]] if len(sensor_cols) else []
            top_sensor_names.append(names)

        frames: Dict[str, pd.DataFrame | pd.Series | np.ndarray] = {
            'smooth': smooth,
            'residual_all': residual_all,
            'zscore_all': zscore_all,
            'pressure_resid': pressure_resid,
            'pressure_z': pressure_z,
            'pairwise_z': pairwise_z,
            'ewma_z': ewma_z,
            'slope_z': slope_raw,
            'flow_z': flow_z,
            'sensor_score': sensor_score,
            'pressure_topk': pd.Series(pressure_topk, index=idx),
            'pairwise_topk': pd.Series(pairwise_topk, index=idx),
            'flow_topk': pd.Series(flow_topk, index=idx),
            'ewma_topk': pd.Series(ewma_topk, index=idx),
            'slope_topk': pd.Series(slope_topk, index=idx),
            'burst_raw': pd.Series(burst_raw, index=idx),
            'burst_cusum': pd.Series(burst_cusum, index=idx),
            'score_burst': pd.Series(score_burst, index=idx),
            'score_incipient': pd.Series(score_incipient, index=idx),
            'score_final': pd.Series(score_final, index=idx),
            'sensor_votes': sensor_votes,
            'top_sensor_names': top_sensor_names,
            'timebins': meta['timebins'],
        }
        return frames

    def fit(self, df: pd.DataFrame, known_intervals: Optional[Sequence[Tuple[pd.Timestamp, pd.Timestamp]]] = None) -> 'BattLeDIMStage1':
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError('fit(df): df must have a DatetimeIndex.')
        self.groups_ = _infer_column_groups(df)
        self.pressure_cols_ = list(self.groups_['pressure_cols'])
        self.flow_cols_ = list(self.groups_['flow_cols'])
        self.tank_cols_ = list(self.groups_['tank_cols'])
        self.baseline_mode_ = self._resolve_baseline_mode()

        exclusion_mask = _mask_from_known_intervals(df.index, known_intervals)
        train_mask = ~exclusion_mask
        if train_mask.sum() < max(10, len(df) // 10):
            train_mask[:] = True
        train_df = df.loc[train_mask].copy()

        # Flow evidence selection.
        keep_flows, auto_excluded = self._auto_exclude_binary_like_flows(train_df)
        self.flow_score_cols_ = list(keep_flows)
        self.auto_excluded_flow_score_cols_ = [c for c in auto_excluded if c not in set(self.cfg.flow_score_exclude_cols)]
        self.flow_context_cols_ = [c for c in self.flow_cols_ if c not in self.flow_score_cols_]

        # Smooth first so correlation ranking is less noisy.
        smooth_train_all = train_df[self.pressure_cols_].astype(float).rolling(window=int(self.cfg.smooth_window), min_periods=1).median()
        self.monitored_pressure_cols_, self.anchor_pressure_cols_ = self._select_pressure_subset(smooth_train_all)

        # Baselines and scales.
        relevant_cols = sorted(set(self.monitored_pressure_cols_ + self.flow_score_cols_ + self.flow_context_cols_ + self.tank_cols_), key=_natural_key)
        self.all_signal_cols_ = relevant_cols
        smooth_train = train_df[relevant_cols].astype(float).rolling(window=int(self.cfg.smooth_window), min_periods=1).median()
        for col in relevant_cols:
            bins, table = _group_median_lookup(train_df.index, smooth_train[col], mode=self.baseline_mode_, step_minutes=int(self.cfg.step_minutes), weekly_period=int(self.cfg.weekly_period))
            resid = smooth_train[col].to_numpy(dtype=float, copy=False) - table[bins]
            self.baseline_tables_[col] = table.astype(float)
            self.scale_[col] = _robust_scale(resid)

        self.pairwise_params_ = self._fit_pairwise_models(smooth_train_all.loc[train_df.index])
        self.is_fitted_ = True

        train_frames = self._compute_score_frames(train_df)
        burst_q = _safe_quantile(train_frames['score_burst'].to_numpy(dtype=float, copy=False), float(self.cfg.burst_start_quantile), default=1.0)
        incipient_q = _safe_quantile(train_frames['score_incipient'].to_numpy(dtype=float, copy=False), float(self.cfg.incipient_start_quantile), default=1.0)
        hold_q = _safe_quantile(train_frames['score_final'].to_numpy(dtype=float, copy=False), float(self.cfg.hold_quantile), default=0.5)
        flow_hold_q = _safe_quantile(train_frames['flow_topk'].to_numpy(dtype=float, copy=False), float(self.cfg.flow_hold_quantile), default=0.25)

        self.thresholds_ = {
            'burst_start': max(0.10, burst_q),
            'incipient_start': max(0.10, incipient_q),
            'hold': max(0.05, hold_q),
            'flow_hold': max(0.05, flow_hold_q),
        }
        self.train_metrics_ = {
            'fit_rows': int(len(train_df)),
            'excluded_rows': int(exclusion_mask.sum()),
            'monitored_pressure_count': int(len(self.monitored_pressure_cols_)),
            'flow_score_count': int(len(self.flow_score_cols_)),
        }
        return self

    # ----------------------------
    # Detection
    # ----------------------------

    def detect(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if not self.is_fitted_:
            raise RuntimeError('Detector is not fitted.')
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError('detect(df): df must have a DatetimeIndex.')
        frames = self._compute_score_frames(df)
        idx = df.index

        burst_above = pd.Series((frames['score_burst'].to_numpy(dtype=float, copy=False) >= self.thresholds_['burst_start']).astype(int), index=idx)
        incipient_above = pd.Series((frames['score_incipient'].to_numpy(dtype=float, copy=False) >= self.thresholds_['incipient_start']).astype(int), index=idx)
        burst_confirmed = _rolling_sum_bool(burst_above, int(self.cfg.burst_confirm_window)) >= int(self.cfg.burst_confirm_needed)
        incipient_confirmed = _rolling_sum_bool(incipient_above, int(self.cfg.incipient_confirm_window)) >= int(self.cfg.incipient_confirm_needed)

        leak_state = np.zeros(len(idx), dtype=int)
        event_ids = np.zeros(len(idx), dtype=int)
        event_id = 0
        quiet_count = 0
        quiet_limit = max(int(self.cfg.quiet_end_samples), int(self.cfg.merge_gap_samples))
        for i, ts in enumerate(idx):
            score_final = float(frames['score_final'].iat[i])
            flow_topk = float(frames['flow_topk'].iat[i])
            sensor_votes = int(frames['sensor_votes'].iat[i])
            start_now = (
                sensor_votes >= int(self.cfg.start_sensor_votes)
                and (bool(burst_confirmed.iat[i]) or bool(incipient_confirmed.iat[i]))
            )
            hold_now = (
                sensor_votes >= int(self.cfg.hold_sensor_votes)
                and (
                    score_final >= float(self.thresholds_['hold'])
                    or flow_topk >= float(self.thresholds_['flow_hold'])
                    or bool(incipient_confirmed.iat[i])
                )
            )
            prev = leak_state[i - 1] if i > 0 else 0
            if prev == 0:
                if start_now:
                    event_id += 1
                    leak_state[i] = 1
                    quiet_count = 0
                else:
                    leak_state[i] = 0
            else:
                if hold_now:
                    leak_state[i] = 1
                    quiet_count = 0
                else:
                    quiet_count += 1
                    if quiet_count >= quiet_limit:
                        leak_state[i] = 0
                        quiet_count = 0
                    else:
                        leak_state[i] = 1
            event_ids[i] = event_id if leak_state[i] == 1 else 0

        sensor_score = frames['sensor_score'].copy()
        sensor_detail = pd.DataFrame(index=idx)
        for col in self.pressure_cols_:
            if col in sensor_score.columns:
                sensor_detail[col] = sensor_score[col].astype(float)
            else:
                sensor_detail[col] = 0.0
        sensor_detail['event_id'] = event_ids
        sensor_detail['leak_state'] = leak_state

        # Results table.
        results = pd.DataFrame(index=idx)
        for key in ['pressure_topk', 'pairwise_topk', 'flow_topk', 'ewma_topk', 'slope_topk', 'burst_raw', 'burst_cusum', 'score_burst', 'score_incipient', 'score_final']:
            results[key] = frames[key].astype(float)
        results['score_burst_trigger'] = burst_confirmed.astype(int).astype(float)
        results['sensor_votes'] = frames['sensor_votes'].astype(int)
        results['leak_state'] = leak_state.astype(int)
        results['event_id'] = event_ids.astype(int)

        top_sensor_names: List[List[str]] = frames['top_sensor_names']
        area_guess = []
        for names in top_sensor_names:
            votes: Dict[str, int] = {}
            for n in names:
                area = self.cfg.sensor_areas.get(n, 'unknown') if isinstance(self.cfg.sensor_areas, dict) else 'unknown'
                votes[area] = votes.get(area, 0) + 1
            area_guess.append(max(votes, key=votes.get) if votes else 'unknown')
        results['area_guess'] = area_guess
        for i in range(1, int(self.cfg.top_sensors_output) + 1):
            results[f'top_sensor_{i}'] = [names[i - 1] if len(names) >= i else '' for names in top_sensor_names]

        events = self._build_events_from_results(results)

        self.last_internal_ = {
            **frames,
            'results': results,
            'sensor_detail': sensor_detail,
            'events': events,
        }
        return results, sensor_detail, events

    def _build_events_from_results(self, results: pd.DataFrame) -> pd.DataFrame:
        rows = []
        active = results['leak_state'].fillna(0).astype(int)
        if active.empty:
            return pd.DataFrame(columns=['event_id', 'start_time', 'end_time', 'duration_samples', 'duration_hours', 'max_score_final', 'area_guess', 'top_sensors'])
        for eid in sorted([int(x) for x in results['event_id'].unique() if int(x) > 0]):
            block = results[results['event_id'] == eid]
            if block.empty:
                continue
            start = pd.Timestamp(block.index[0])
            end = pd.Timestamp(block.index[-1])
            duration_samples = int(len(block))
            duration_hours = duration_samples * float(self.cfg.step_minutes) / 60.0
            max_score = float(block['score_final'].max()) if 'score_final' in block.columns else np.nan
            area = str(block['area_guess'].mode().iloc[0]) if 'area_guess' in block.columns and not block['area_guess'].mode().empty else 'unknown'
            last_row = block.iloc[-1]
            top_sensors = [str(last_row.get(f'top_sensor_{i}', '')) for i in range(1, int(self.cfg.top_sensors_output) + 1)]
            rows.append({
                'event_id': eid,
                'start_time': start,
                'end_time': end,
                'duration_samples': duration_samples,
                'duration_hours': duration_hours,
                'max_score_final': max_score,
                'area_guess': area,
                'top_sensors': ', '.join([x for x in top_sensors if x]),
            })
        return pd.DataFrame(rows)

    def build_stage2_triggers(self, results: pd.DataFrame) -> pd.DataFrame:
        if results.empty:
            return pd.DataFrame(columns=['timestamp', 'event_id', 'reason', 'area_guess', 'top_sensors'])
        idx = results.index
        leak_state = results['leak_state'].fillna(0).to_numpy(dtype=int, copy=False)
        event_id = results['event_id'].fillna(0).to_numpy(dtype=int, copy=False)
        area_arr = results['area_guess'].fillna('unknown').astype(str).to_numpy(copy=False) if 'area_guess' in results.columns else np.array(['unknown'] * len(results), dtype=object)
        top_cols = [c for c in [f'top_sensor_{j}' for j in range(1, int(self.cfg.top_sensors_output) + 1)] if c in results.columns]
        top_mat = results[top_cols].fillna('').astype(str).to_numpy(copy=False) if top_cols else np.empty((len(results), 0), dtype=object)

        recheck_steps = max(1, int(round(float(self.cfg.stage2_recheck_hours) * 60.0 / float(self.cfg.step_minutes))))
        shift_gap_steps = max(1, int(round(float(self.cfg.min_sensor_shift_minutes) / float(self.cfg.step_minutes))))
        rows = []
        last_trigger_pos_by_event: Dict[int, int] = {}
        last_top3_by_event: Dict[int, set] = {}

        for i, ts in enumerate(idx):
            if leak_state[i] != 1:
                continue
            eid = int(event_id[i])
            if eid <= 0:
                continue
            prev = int(leak_state[i - 1]) if i > 0 else 0
            top_values = top_mat[i] if len(top_mat) else []
            top3 = {str(x) for x in top_values[:min(3, len(top_values))] if str(x)}
            reason = None
            if prev == 0:
                reason = 'event_start'
            else:
                last_pos = last_trigger_pos_by_event.get(eid)
                if last_pos is not None and (i - last_pos) >= recheck_steps:
                    reason = 'periodic_recheck'
                elif last_pos is not None and (i - last_pos) >= shift_gap_steps:
                    prev_top3 = last_top3_by_event.get(eid, set())
                    union = len(prev_top3 | top3)
                    jaccard = (len(prev_top3 & top3) / union) if union > 0 else 1.0
                    if jaccard < float(self.cfg.sensor_shift_jaccard_threshold):
                        reason = 'sensor_shift'
            if reason is not None:
                top_sensors = ', '.join([str(x) for x in top_values if str(x)]) if len(top_values) else ''
                rows.append({
                    'timestamp': pd.Timestamp(ts),
                    'event_id': eid,
                    'reason': reason,
                    'area_guess': str(area_arr[i]),
                    'top_sensors': top_sensors,
                })
                last_trigger_pos_by_event[eid] = i
                last_top3_by_event[eid] = top3
        return pd.DataFrame(rows, columns=['timestamp', 'event_id', 'reason', 'area_guess', 'top_sensors'])

    # ----------------------------
    # Export helpers
    # ----------------------------

    def export_params(self) -> Dict[str, object]:
        if not self.is_fitted_:
            raise RuntimeError('Detector is not fitted.')
        return {
            'config': self.cfg.to_dict(),
            'baseline_mode': self.baseline_mode_,
            'pressure_cols': self.pressure_cols_,
            'monitored_pressure_cols': self.monitored_pressure_cols_,
            'anchor_pressure_cols': self.anchor_pressure_cols_,
            'flow_score_cols': self.flow_score_cols_,
            'flow_context_cols': self.flow_context_cols_,
            'auto_excluded_flow_score_cols': self.auto_excluded_flow_score_cols_,
            'baseline_tables': {k: [float(x) for x in v] for k, v in self.baseline_tables_.items()},
            'scale': {k: float(v) for k, v in self.scale_.items()},
            'pairwise_params': {
                sensor: [
                    {'anchor': anchor, 'slope': float(slope), 'intercept': float(intercept), 'scale': float(scale)}
                    for anchor, slope, intercept, scale in rows
                ]
                for sensor, rows in self.pairwise_params_.items()
            },
            'thresholds': {k: float(v) for k, v in self.thresholds_.items()},
            'train_metrics': self.train_metrics_,
        }

    def save_params_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(json.dumps(self.export_params(), indent=2))
        return path
