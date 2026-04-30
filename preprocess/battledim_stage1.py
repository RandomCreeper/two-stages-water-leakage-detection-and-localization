from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


EPS = 1e-9


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------


def robust_sigma(x: pd.Series | np.ndarray) -> float:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 1.0
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
    sigma = 1.4826 * mad
    return float(sigma if sigma > EPS else max(np.std(arr), 1.0))



def robust_sigma_frame(df: pd.DataFrame) -> pd.Series:
    return df.apply(robust_sigma, axis=0)



def topk_mean(df: pd.DataFrame, k: int) -> pd.Series:
    if df.empty:
        return pd.Series(0.0, index=df.index)
    k = max(1, min(k, df.shape[1]))
    arr = np.nan_to_num(df.to_numpy(dtype=float), nan=0.0)
    part = np.partition(arr, arr.shape[1] - k, axis=1)[:, -k:]
    return pd.Series(part.mean(axis=1), index=df.index)



def rolling_vote(mask: pd.Series, window: int, needed: int) -> pd.Series:
    return mask.astype(int).rolling(window=window, min_periods=1).sum() >= needed



def jaccard_similarity(a: Sequence[str], b: Sequence[str]) -> float:
    sa = {x for x in a if x}
    sb = {x for x in b if x}
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(1, len(sa | sb))



def _try_parse_timestamp_column(df: pd.DataFrame, explicit: Optional[str] = None) -> Tuple[pd.DataFrame, str]:
    if explicit is not None:
        ts = pd.to_datetime(df[explicit], errors="coerce")
        if ts.notna().mean() < 0.8:
            raise ValueError(f"Could not parse timestamp column '{explicit}'.")
        out = df.copy()
        out.index = ts
        out = out.drop(columns=[explicit])
        return out.sort_index(), explicit

    candidates = [
        c for c in df.columns
        if any(tok in c.lower() for tok in ["time", "date", "timestamp", "datetime"])
    ]
    for c in candidates + [df.columns[0]]:
        ts = pd.to_datetime(df[c], errors="coerce")
        if ts.notna().mean() >= 0.8:
            out = df.copy()
            out.index = ts
            out = out.drop(columns=[c])
            return out.sort_index(), c
    raise ValueError("Could not infer a timestamp column. Pass timestamp_col explicitly.")



def load_scada_csv(path: str | Path, timestamp_col: Optional[str] = None) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)
    df, _ = _try_parse_timestamp_column(df, explicit=timestamp_col)
    # keep only numeric columns after the timestamp was moved to index
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    df = df[numeric_cols].astype(float)
    return df



def _natural_sort_key(name: str) -> Tuple[str, int, str]:
    name_str = str(name)
    match = re.search(r"(\d+)(?!.*\d)", name_str)
    number = int(match.group(1)) if match else -1
    stem = re.sub(r"\d+", "", name_str.lower())
    return stem, number, name_str.lower()



def infer_battledim_merged_schema(df: pd.DataFrame) -> Dict[str, List[str]]:
    cols = list(df.columns)
    pressure_cols = sorted([c for c in cols if re.fullmatch(r"press_\d+", str(c), flags=re.I)], key=_natural_sort_key)
    flow_cols = sorted([c for c in cols if re.fullmatch(r"flow_\d+", str(c), flags=re.I)], key=_natural_sort_key)
    level_cols = sorted([c for c in cols if re.fullmatch(r"t\d+", str(c), flags=re.I)], key=_natural_sort_key)
    demand_cols = sorted([c for c in cols if re.fullmatch(r"n\d+", str(c), flags=re.I)], key=_natural_sort_key)
    pipe_bin_cols = sorted([c for c in cols if re.fullmatch(r"pipe_\d+_bin", str(c), flags=re.I)], key=_natural_sort_key)
    leak_label_cols = [
        c for c in cols
        if c in {"leak_mag_total", "total_leak_mag", "leak_count", "leak_binary"} or c in pipe_bin_cols
    ]
    return {
        "pressure_cols": pressure_cols,
        "flow_cols": flow_cols,
        "level_cols": level_cols,
        "demand_cols": demand_cols,
        "pipe_bin_cols": pipe_bin_cols,
        "label_cols": leak_label_cols,
    }



def load_battledim_merged_csv(path: str | Path, timestamp_col: Optional[str] = None) -> pd.DataFrame:
    """
    Load a single merged BattLeDIM-style CSV like the schema the user shared:
      Timestamp, press_*, pipe_*_bin, leak_mag_total, leak_count, leak_binary,
      flow_*, T1, n*.

    The returned dataframe uses a DatetimeIndex and stores inferred column groups in
    dataframe attrs so BattLeDIMStage1.fit() can auto-wire itself.
    """
    path = Path(path)
    df = pd.read_csv(path)
    df, used_ts_col = _try_parse_timestamp_column(df, explicit=timestamp_col)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    numeric_cols = [c for c in df.columns if df[c].notna().any()]
    df = df[numeric_cols].astype(float)
    schema = infer_battledim_merged_schema(df)
    df.attrs.update(schema)
    df.attrs["timestamp_col"] = used_ts_col
    return df



def boolean_series_to_intervals(
    mask: pd.Series,
    pad_before: str | pd.Timedelta = "0min",
    pad_after: str | pd.Timedelta = "0min",
    min_active_samples: int = 1,
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    if not isinstance(mask.index, pd.DatetimeIndex):
        raise TypeError("boolean_series_to_intervals requires a DatetimeIndex.")
    x = mask.fillna(False).astype(bool)
    before = pd.Timedelta(pad_before)
    after = pd.Timedelta(pad_after)

    intervals: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    start: Optional[pd.Timestamp] = None
    prev_ts: Optional[pd.Timestamp] = None
    active_count = 0

    for ts, val in x.items():
        if val:
            if start is None:
                start = ts
                active_count = 1
            else:
                active_count += 1
        else:
            if start is not None and prev_ts is not None and active_count >= max(1, int(min_active_samples)):
                intervals.append((start - before, prev_ts + after))
            start = None
            active_count = 0
        prev_ts = ts

    if start is not None and prev_ts is not None and active_count >= max(1, int(min_active_samples)):
        intervals.append((start - before, prev_ts + after))
    return intervals



def build_known_intervals_from_leak_columns(
    df: pd.DataFrame,
    leak_count_col: str = "leak_count",
    leak_mag_col: str = "leak_mag_total",
    leak_binary_col: str = "leak_binary",
    pipe_bin_cols: Optional[Sequence[str]] = None,
    pad_before: str | pd.Timedelta = "0min",
    pad_after: str | pd.Timedelta = "0min",
    min_active_samples: int = 1,
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Turn BattLeDIM leak labels into exclusion windows for training.

    Priority:
      1) leak_count > 0
      2) leak_mag_total > 0 (or total_leak_mag > 0)
      3) any pipe_*_bin == 1
      4) leak_binary > 0 (fallback only; often over-active in BattLeDIM-derived tables)
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        df, _ = _try_parse_timestamp_column(df)

    schema = infer_battledim_merged_schema(df)
    if pipe_bin_cols is None:
        pipe_bin_cols = schema.get("pipe_bin_cols", [])

    mask = pd.Series(False, index=df.index)
    usable_primary = False

    if leak_count_col in df.columns:
        mask |= pd.to_numeric(df[leak_count_col], errors="coerce").fillna(0.0) > 0.0
        usable_primary = True

    for col in [leak_mag_col, "total_leak_mag"]:
        if col in df.columns:
            mask |= pd.to_numeric(df[col], errors="coerce").fillna(0.0) > 0.0
            usable_primary = True
            break

    if pipe_bin_cols:
        pipe_mask = df[list(pipe_bin_cols)].apply(pd.to_numeric, errors="coerce").fillna(0.0).max(axis=1) > 0.0
        mask |= pipe_mask
        usable_primary = True

    if (not usable_primary) and (leak_binary_col in df.columns):
        mask |= pd.to_numeric(df[leak_binary_col], errors="coerce").fillna(0.0) > 0.0

    return boolean_series_to_intervals(
        mask,
        pad_before=pad_before,
        pad_after=pad_after,
        min_active_samples=min_active_samples,
    )



def evaluate_stage1_results(
    results: pd.DataFrame,
    truth_mask: pd.Series,
    triggers: Optional[pd.DataFrame] = None,
) -> pd.Series:
    """Simple tuning metrics for recall-first Stage 1 evaluation."""
    if "leak_state" not in results.columns:
        raise ValueError("results must contain a 'leak_state' column from detect()/score().")
    truth = truth_mask.reindex(results.index).fillna(False).astype(bool)
    pred = results["leak_state"].reindex(results.index).fillna(0).astype(bool)

    tp = int((pred & truth).sum())
    fn = int((~pred & truth).sum())
    fp = int((pred & ~truth).sum())
    tn = int((~pred & ~truth).sum())

    truth_intervals = boolean_series_to_intervals(truth)
    pred_intervals = boolean_series_to_intervals(pred)

    event_hits = 0
    for t0, t1 in truth_intervals:
        if any((p0 <= t1) and (p1 >= t0) for p0, p1 in pred_intervals):
            event_hits += 1

    out = {
        "timestep_recall": tp / max(1, tp + fn),
        "timestep_precision": tp / max(1, tp + fp),
        "false_positive_rate": fp / max(1, fp + tn),
        "active_fraction": float(pred.mean()),
        "truth_fraction": float(truth.mean()),
        "n_truth_events": len(truth_intervals),
        "n_detected_events": len(pred_intervals),
        "event_recall": event_hits / max(1, len(truth_intervals)),
    }
    if triggers is not None:
        duration_days = max(1e-9, (results.index.max() - results.index.min()).total_seconds() / 86400.0)
        out["stage2_calls_per_day"] = float(len(triggers) / duration_days)
        out["stage2_call_fraction"] = float(len(triggers) / max(1, len(results)))
    return pd.Series(out)



def load_battledim_split(
    root_dir: str | Path,
    year: int,
    timestamp_col: Optional[str] = None,
    join_how: str = "outer",
) -> pd.DataFrame:
    """
    Load BattLeDIM split files for a given year.

    Expected default files (official Zenodo packaging):
      - {year}_SCADA_Pressures.csv
      - {year}_SCADA_Flows.csv
      - {year}_SCADA_Levels.csv
      - {year}_SCADA_Demands.csv  (optional for this stage-1 code)
    """
    root = Path(root_dir)
    file_map = {
        "pressure_cols": root / f"{year}_SCADA_Pressures.csv",
        "flow_cols": root / f"{year}_SCADA_Flows.csv",
        "level_cols": root / f"{year}_SCADA_Levels.csv",
        "demand_cols": root / f"{year}_SCADA_Demands.csv",
    }

    frames: List[pd.DataFrame] = []
    attrs: Dict[str, List[str]] = {}

    for group_name, path in file_map.items():
        if not path.exists():
            continue
        frame = load_scada_csv(path, timestamp_col=timestamp_col)
        attrs[group_name] = frame.columns.tolist()
        frames.append(frame)

    if not frames:
        raise FileNotFoundError(f"No BattLeDIM split CSVs found in {root!s} for year {year}.")

    merged = pd.concat(frames, axis=1, join=join_how).sort_index()
    merged = merged[~merged.index.duplicated(keep="first")]
    merged.attrs.update(attrs)
    return merged



def parse_fixed_leak_report(
    path: str | Path,
    pad_before: str | pd.Timedelta = "1D",
    pad_after: str | pd.Timedelta = "1D",
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Parse a BattLeDIM 2018 fixed-leak/burst report and turn repair timestamps
    into exclusion windows around the repair time.

    The official BattLeDIM data page states that the 2018 leak report contains
    the repair times of fixed pipe bursts. This function extracts ISO-like
    timestamps and pads them to remove nearby abnormal training windows.
    """
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    ts_strings = re.findall(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(?::\d{2})?", text)
    if not ts_strings:
        return []
    before = pd.Timedelta(pad_before)
    after = pd.Timedelta(pad_after)
    windows: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    for s in ts_strings:
        t = pd.Timestamp(s)
        windows.append((t - before, t + after))
    return sorted(windows)



def build_interval_mask(index: pd.DatetimeIndex, intervals: Sequence[Tuple[pd.Timestamp, pd.Timestamp]]) -> pd.Series:
    mask = pd.Series(False, index=index)
    for start, end in intervals:
        mask |= (index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end))
    return mask



def infer_column_groups(
    df: pd.DataFrame,
    pressure_cols: Optional[Sequence[str]] = None,
    flow_cols: Optional[Sequence[str]] = None,
    level_cols: Optional[Sequence[str]] = None,
    demand_cols: Optional[Sequence[str]] = None,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    schema = infer_battledim_merged_schema(df)
    if pressure_cols is None:
        pressure_cols = df.attrs.get("pressure_cols") or schema.get("pressure_cols")
    if flow_cols is None:
        flow_cols = df.attrs.get("flow_cols") or schema.get("flow_cols")
    if level_cols is None:
        level_cols = df.attrs.get("level_cols") or schema.get("level_cols")
    if demand_cols is None:
        demand_cols = df.attrs.get("demand_cols") or schema.get("demand_cols")

    def _ensure(cols: Optional[Sequence[str]], label: str) -> List[str]:
        if cols is None:
            return []
        out = [c for c in cols if c in df.columns]
        if label in {"pressure", "flow"} and not out:
            raise ValueError(
                f"Could not resolve {label} columns. Pass them explicitly, use load_battledim_split(), or use load_battledim_merged_csv()."
            )
        return sorted(out, key=_natural_sort_key)

    return (
        _ensure(pressure_cols, "pressure"),
        _ensure(flow_cols, "flow"),
        _ensure(level_cols, "level"),
        _ensure(demand_cols, "demand"),
    )



def time_of_week_slot(index: pd.DatetimeIndex, step_minutes: int = 5) -> np.ndarray:
    minutes = index.dayofweek.values * 24 * 60 + index.hour.values * 60 + index.minute.values
    return minutes // step_minutes



def causal_median_smooth(df: pd.DataFrame, window: int) -> pd.DataFrame:
    if window <= 1:
        return df.copy()
    return df.rolling(window=window, min_periods=1).median()


# -----------------------------------------------------------------------------
# Seasonal profile
# -----------------------------------------------------------------------------


class WeeklySeasonalProfile:
    def __init__(self, step_minutes: int = 5):
        self.step_minutes = step_minutes
        self.period = int(7 * 24 * 60 / step_minutes)
        self.location_: Optional[pd.DataFrame] = None
        self.scale_: Optional[pd.DataFrame] = None
        self.global_location_: Optional[pd.Series] = None
        self.global_scale_: Optional[pd.Series] = None

    def fit(self, df: pd.DataFrame) -> "WeeklySeasonalProfile":
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("WeeklySeasonalProfile requires a DatetimeIndex.")
        slots = pd.Series(time_of_week_slot(df.index, self.step_minutes), index=df.index, name="slot")
        med = df.groupby(slots).median()
        aligned_med = med.loc[slots.values].set_index(df.index)
        abs_dev = (df - aligned_med).abs()
        mad = abs_dev.groupby(slots).median()

        full_index = pd.Index(np.arange(self.period), name="slot")
        med = med.reindex(full_index)
        mad = mad.reindex(full_index)

        global_location = df.median()
        global_scale = robust_sigma_frame(df)

        med = med.ffill().bfill().fillna(global_location)
        mad = mad.fillna(global_scale / 1.4826)
        mad = mad.replace(0.0, np.nan).fillna(global_scale / 1.4826)

        self.location_ = med
        self.scale_ = mad * 1.4826
        self.global_location_ = global_location
        self.global_scale_ = global_scale.replace(0.0, 1.0)
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if self.location_ is None or self.scale_ is None:
            raise RuntimeError("WeeklySeasonalProfile must be fit before transform().")
        slots = time_of_week_slot(df.index, self.step_minutes)
        loc = self.location_.loc[slots].set_index(df.index)
        scale = self.scale_.loc[slots].set_index(df.index).replace(0.0, 1.0)
        resid = df - loc
        return resid, loc, scale


# -----------------------------------------------------------------------------
# Robust linear model (IRLS Huber)
# -----------------------------------------------------------------------------


class RobustLinearModel:
    def __init__(self, delta: float = 1.5, ridge: float = 1e-3, max_iter: int = 30):
        self.delta = float(delta)
        self.ridge = float(ridge)
        self.max_iter = int(max_iter)
        self.columns_: List[str] = []
        self.x_center_: Optional[pd.Series] = None
        self.x_scale_: Optional[pd.Series] = None
        self.beta_: Optional[np.ndarray] = None
        self.resid_scale_: float = 1.0
        self.y_center_: float = 0.0

    def _prepare_X(self, X: pd.DataFrame) -> np.ndarray:
        if self.x_center_ is None or self.x_scale_ is None:
            raise RuntimeError("Model has not been fit.")
        X = X.reindex(columns=self.columns_).copy()
        for c in self.columns_:
            if c not in X.columns:
                X[c] = self.x_center_[c]
        X = X[self.columns_]
        X = X.fillna(self.x_center_)
        Xs = (X - self.x_center_) / self.x_scale_
        return Xs.to_numpy(dtype=float)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RobustLinearModel":
        data = pd.concat([X, y.rename("__y__")], axis=1).dropna()
        if data.empty:
            self.columns_ = list(X.columns)
            self.x_center_ = pd.Series(0.0, index=self.columns_)
            self.x_scale_ = pd.Series(1.0, index=self.columns_)
            self.beta_ = np.zeros(len(self.columns_) + 1, dtype=float)
            self.beta_[0] = float(y.median()) if len(y) else 0.0
            self.resid_scale_ = max(1.0, robust_sigma(y))
            return self

        Xd = data.drop(columns="__y__")
        yd = data["__y__"].astype(float)

        self.columns_ = list(Xd.columns)
        self.x_center_ = Xd.median()
        self.x_scale_ = robust_sigma_frame(Xd).replace(0.0, 1.0)

        Xs = ((Xd - self.x_center_) / self.x_scale_).to_numpy(dtype=float)
        yv = yd.to_numpy(dtype=float)
        X1 = np.column_stack([np.ones(len(Xs)), Xs])
        beta = np.linalg.lstsq(X1, yv, rcond=None)[0]

        reg = np.eye(X1.shape[1]) * self.ridge
        reg[0, 0] = 0.0

        for _ in range(self.max_iter):
            resid = yv - X1 @ beta
            scale = max(robust_sigma(resid), 1e-3)
            cutoff = self.delta * scale
            weights = np.minimum(1.0, cutoff / (np.abs(resid) + EPS))
            Xw = X1 * weights[:, None]
            A = X1.T @ Xw + reg
            b = X1.T @ (weights * yv)
            beta_new = np.linalg.solve(A, b)
            if np.linalg.norm(beta_new - beta) <= 1e-6 * (1.0 + np.linalg.norm(beta)):
                beta = beta_new
                break
            beta = beta_new

        resid = yv - X1 @ beta
        self.beta_ = beta
        self.resid_scale_ = max(robust_sigma(resid), 1e-3)
        self.y_center_ = float(np.median(yv))
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.beta_ is None:
            raise RuntimeError("Model has not been fit.")
        Xs = self._prepare_X(X)
        X1 = np.column_stack([np.ones(len(Xs)), Xs])
        pred = X1 @ self.beta_
        return pd.Series(pred, index=X.index)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


@dataclass
class Stage1Config:
    smooth_window: int = 3
    step_minutes: int = 5
    weekly_period: int = 2016  # 7 * 24 * 12 for 5-minute BattLeDIM data

    top_n_anchor_pressures: int = 5
    pairwise_anchors_per_sensor: int = 3
    min_top_k: int = 3
    top_k_fraction: float = 0.15
    top_sensors_output: int = 5

    huber_delta: float = 1.5
    ridge_lambda: float = 1e-3
    max_irls_iter: int = 30
    trim_quantile: float = 0.97

    ewma_alpha: float = 0.15
    slope_window: int = 24  # 24*5 min = 2 hours
    sensor_vote_threshold: float = 1.5
    start_sensor_votes: int = 2
    hold_sensor_votes: int = 2

    weight_pressure: float = 1.0
    weight_pairwise: float = 0.6
    weight_flow: float = 0.7
    weight_night_flow: float = 0.4
    weight_slope: float = 0.5

    cusum_allowance: float = 0.5
    cusum_decay: float = 0.85

    burst_start_quantile: float = 0.995
    incipient_start_quantile: float = 0.995
    hold_quantile: float = 0.97
    flow_hold_quantile: float = 0.95

    burst_confirm_window: int = 3
    burst_confirm_needed: int = 2
    incipient_confirm_window: int = 12
    incipient_confirm_needed: int = 6
    quiet_end_samples: int = 12
    merge_gap_samples: int = 6

    night_start_hour: int = 1
    night_end_hour: int = 4

    stage2_recheck_hours: float = 6.0
    min_sensor_shift_minutes: int = 60
    sensor_shift_jaccard_threshold: float = 0.5

    flow_score_exclude_cols: List[str] = field(default_factory=list)
    auto_exclude_binary_like_flows: bool = True
    binary_like_round_decimals: int = 2
    binary_like_edge_fraction_threshold: float = 0.85
    binary_like_unique_ratio_threshold: float = 0.02

    sensor_areas: Dict[str, str] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Stage 1 detector
# -----------------------------------------------------------------------------


class BattLeDIMStage1:
    """
    Recall-first Stage 1 detector for BattLeDIM-like SCADA data.

    Pipeline:
      1) causal median smoothing
      2) weekly seasonal profile removal
      3) robust pressure and flow residual models
      4) pairwise pressure reconstruction error monitoring
      5) burst + incipient scores
      6) hysteretic leak-state tracker
      7) Stage-2 trigger scheduler
    """

    def __init__(self, config: Optional[Stage1Config] = None):
        self.config = config or Stage1Config()
        self.fitted_ = False

        self.pressure_cols_: List[str] = []
        self.flow_cols_: List[str] = []
        self.flow_score_cols_: List[str] = []
        self.auto_excluded_flow_score_cols_: List[str] = []
        self.has_flow_evidence_: bool = False
        self.level_cols_: List[str] = []
        self.demand_cols_: List[str] = []
        self.all_cols_: List[str] = []

        self.seasonal_: Optional[WeeklySeasonalProfile] = None
        self.pressure_models_: Dict[str, RobustLinearModel] = {}
        self.pressure_model_features_: Dict[str, List[str]] = {}
        self.flow_models_: Dict[str, RobustLinearModel] = {}
        self.flow_model_features_: Dict[str, List[str]] = {}
        self.pairwise_models_: Dict[str, Dict[str, RobustLinearModel]] = {}
        self.pairwise_features_: Dict[str, Dict[str, List[str]]] = {}

        self.pressure_resid_scale_: pd.Series | None = None
        self.flow_resid_scale_: pd.Series | None = None
        self.pairwise_resid_scale_: pd.Series | None = None
        self.weekly_diff_scale_: pd.Series | None = None
        self.slope_diff_scale_: pd.Series | None = None

        self.cusum_allowance_: float = self.config.cusum_allowance
        self.burst_start_threshold_: float = 0.0
        self.incipient_start_threshold_: float = 0.0
        self.hold_threshold_: float = 0.0
        self.flow_hold_threshold_: float = 0.0

    # -----------------------------
    # Fitting
    # -----------------------------

    def fit(
        self,
        df: pd.DataFrame,
        pressure_cols: Optional[Sequence[str]] = None,
        flow_cols: Optional[Sequence[str]] = None,
        level_cols: Optional[Sequence[str]] = None,
        demand_cols: Optional[Sequence[str]] = None,
        known_intervals: Optional[Sequence[Tuple[pd.Timestamp, pd.Timestamp]]] = None,
    ) -> "BattLeDIMStage1":
        df = self._prepare_dataframe(df)
        self.pressure_cols_, self.flow_cols_, self.level_cols_, self.demand_cols_ = infer_column_groups(
            df,
            pressure_cols=pressure_cols,
            flow_cols=flow_cols,
            level_cols=level_cols,
            demand_cols=demand_cols,
        )
        self.all_cols_ = self.pressure_cols_ + self.flow_cols_ + self.level_cols_ + self.demand_cols_
        if not self.all_cols_:
            raise ValueError("No sensor columns available for training.")

        df = df[self.all_cols_].copy()
        smooth = causal_median_smooth(df, self.config.smooth_window)

        healthy_mask = pd.Series(True, index=smooth.index)
        if known_intervals:
            healthy_mask &= ~build_interval_mask(smooth.index, known_intervals)
        healthy = smooth.loc[healthy_mask]
        if len(healthy) < 1000:
            raise ValueError("Too little healthy training data after exclusions.")

        self.seasonal_ = WeeklySeasonalProfile(step_minutes=self.config.step_minutes).fit(healthy)
        healthy_ds, _, _ = self.seasonal_.transform(healthy)

        self._set_flow_score_columns(healthy[self.flow_cols_] if self.flow_cols_ else pd.DataFrame(index=healthy.index))
        self._fit_pressure_models(healthy_ds)
        self._fit_flow_models(healthy_ds)
        self._fit_pairwise_models(healthy_ds)

        train_features = self._compute_feature_tables(smooth)
        train_scores = self._aggregate_scores(train_features, cusum_allowance=self.config.cusum_allowance)
        self.cusum_allowance_ = float(train_scores["burst_raw"].median() + 0.5 * robust_sigma(train_scores["burst_raw"]))
        train_scores = self._aggregate_scores(train_features, cusum_allowance=self.cusum_allowance_)

        self.burst_start_threshold_ = float(
            np.nanquantile(train_scores["score_burst_trigger"].to_numpy(), self.config.burst_start_quantile)
        )
        self.incipient_start_threshold_ = float(
            np.nanquantile(train_scores["score_incipient"].to_numpy(), self.config.incipient_start_quantile)
        )
        self.hold_threshold_ = float(
            np.nanquantile(train_scores["score_final"].to_numpy(), self.config.hold_quantile)
        )
        self.flow_hold_threshold_ = float(
            np.nanquantile(train_scores["flow_topk"].to_numpy(), self.config.flow_hold_quantile)
        )

        self.fitted_ = True
        return self

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df.index, pd.DatetimeIndex):
            df, _ = _try_parse_timestamp_column(df)
        out = df.sort_index().copy()
        out = out[~out.index.duplicated(keep="first")]
        for c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        numeric_cols = [c for c in out.columns if out[c].notna().any()]
        out = out[numeric_cols].astype(float)
        return out

    def _select_top_correlated(self, X: pd.DataFrame, y: pd.Series, n: int) -> List[str]:
        if X.empty or n <= 0:
            return []
        scores = {}
        yv = y.to_numpy(dtype=float)
        ystd = np.nanstd(yv)
        for c in X.columns:
            xv = X[c].to_numpy(dtype=float)
            if np.nanstd(xv) <= EPS or ystd <= EPS:
                corr = 0.0
            else:
                corr = np.corrcoef(np.nan_to_num(xv), np.nan_to_num(yv))[0, 1]
                if not np.isfinite(corr):
                    corr = 0.0
            scores[c] = abs(float(corr))
        ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [c for c, _ in ordered[:n]]

    def _fit_trimmed_model(self, X: pd.DataFrame, y: pd.Series) -> RobustLinearModel:
        model = RobustLinearModel(
            delta=self.config.huber_delta,
            ridge=self.config.ridge_lambda,
            max_iter=self.config.max_irls_iter,
        ).fit(X, y)
        pred = model.predict(X)
        z = (y - pred).abs() / max(model.resid_scale_, 1e-3)
        cutoff = float(np.nanquantile(z.to_numpy(), self.config.trim_quantile))
        keep = z <= cutoff
        return RobustLinearModel(
            delta=self.config.huber_delta,
            ridge=self.config.ridge_lambda,
            max_iter=self.config.max_irls_iter,
        ).fit(X.loc[keep], y.loc[keep])

    def _identify_binary_like_flows(self, flow_df: pd.DataFrame) -> List[str]:
        flagged: List[str] = []
        if flow_df.empty:
            return flagged
        for c in flow_df.columns:
            s = pd.to_numeric(flow_df[c], errors="coerce").dropna()
            if len(s) < 100:
                continue
            q_low = float(s.quantile(0.05))
            q_high = float(s.quantile(0.95))
            spread = q_high - q_low
            if spread <= EPS:
                continue
            scaled = ((s - q_low) / spread).clip(0.0, 1.0)
            edge_fraction = float(((scaled <= 0.15) | (scaled >= 0.85)).mean())
            unique_ratio = float(s.round(self.config.binary_like_round_decimals).nunique() / max(1, len(s)))
            if (
                edge_fraction >= self.config.binary_like_edge_fraction_threshold
                and unique_ratio <= self.config.binary_like_unique_ratio_threshold
            ):
                flagged.append(c)
        return sorted(flagged, key=_natural_sort_key)

    def _set_flow_score_columns(self, healthy_flow: pd.DataFrame) -> None:
        explicit_excluded = {c for c in self.config.flow_score_exclude_cols if c in self.flow_cols_}
        auto_excluded: List[str] = []
        if self.config.auto_exclude_binary_like_flows and not healthy_flow.empty:
            auto_excluded = self._identify_binary_like_flows(healthy_flow)
        excluded = explicit_excluded | set(auto_excluded)
        self.auto_excluded_flow_score_cols_ = auto_excluded
        self.flow_score_cols_ = [c for c in self.flow_cols_ if c not in excluded]
        self.has_flow_evidence_ = bool(self.flow_score_cols_)

    def _fit_pressure_models(self, healthy_ds: pd.DataFrame) -> None:
        scales = {}
        for target in self.pressure_cols_:
            candidate_pressures = [c for c in self.pressure_cols_ if c != target]
            anchors = self._select_top_correlated(
                healthy_ds[candidate_pressures], healthy_ds[target], self.config.top_n_anchor_pressures
            )
            features = anchors + self.flow_cols_ + self.level_cols_ + self.demand_cols_
            if not features:
                features = self.flow_cols_ + self.level_cols_ + self.demand_cols_
            X = healthy_ds[features] if features else pd.DataFrame(index=healthy_ds.index)
            y = healthy_ds[target]
            model = self._fit_trimmed_model(X, y)
            self.pressure_models_[target] = model
            self.pressure_model_features_[target] = features
            pred = model.predict(X)
            scales[target] = robust_sigma(y - pred)
        self.pressure_resid_scale_ = pd.Series(scales).replace(0.0, 1.0)

    def _fit_flow_models(self, healthy_ds: pd.DataFrame) -> None:
        scales = {}
        pressure_summary = pd.DataFrame(index=healthy_ds.index)
        if self.pressure_cols_:
            pressure_summary["pressure_mean"] = healthy_ds[self.pressure_cols_].mean(axis=1)
            pressure_summary["pressure_min"] = healthy_ds[self.pressure_cols_].min(axis=1)
        for target in self.flow_cols_:
            other_flows = [c for c in self.flow_cols_ if c != target]
            features = other_flows + self.level_cols_ + self.demand_cols_
            X = pd.concat([healthy_ds[features], pressure_summary], axis=1) if features or not pressure_summary.empty else pressure_summary
            y = healthy_ds[target]
            model = self._fit_trimmed_model(X, y)
            self.flow_models_[target] = model
            self.flow_model_features_[target] = list(X.columns)
            pred = model.predict(X)
            scales[target] = robust_sigma(y - pred)
        self.flow_resid_scale_ = pd.Series(scales).replace(0.0, 1.0) if scales else pd.Series(dtype=float)

    def _fit_pairwise_models(self, healthy_ds: pd.DataFrame) -> None:
        scales = {}
        for target in self.pressure_cols_:
            candidate_pressures = [c for c in self.pressure_cols_ if c != target]
            anchors = self._select_top_correlated(
                healthy_ds[candidate_pressures], healthy_ds[target], self.config.pairwise_anchors_per_sensor
            )
            self.pairwise_models_[target] = {}
            self.pairwise_features_[target] = {}
            for anchor in anchors:
                features = [anchor] + self.flow_cols_ + self.level_cols_ + self.demand_cols_
                X = healthy_ds[features]
                y = healthy_ds[target]
                model = self._fit_trimmed_model(X, y)
                self.pairwise_models_[target][anchor] = model
                self.pairwise_features_[target][anchor] = features
                pred = model.predict(X)
                scales[f"{target}|{anchor}"] = robust_sigma(y - pred)
        self.pairwise_resid_scale_ = pd.Series(scales).replace(0.0, 1.0) if scales else pd.Series(dtype=float)

        # Train scales for slow/incipient detectors from residuals on healthy data.
        residuals = self._compute_pressure_residuals(healthy_ds)
        weekly_diff = residuals - residuals.shift(self.config.weekly_period)
        slope_diff = residuals - residuals.shift(max(1, self.config.slope_window - 1))
        self.weekly_diff_scale_ = weekly_diff.apply(robust_sigma, axis=0).replace(0.0, 1.0)
        self.slope_diff_scale_ = slope_diff.apply(robust_sigma, axis=0).replace(0.0, 1.0)

    # -----------------------------
    # Core feature computation
    # -----------------------------

    def _compute_pressure_residuals(self, ds: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=ds.index)
        for target, model in self.pressure_models_.items():
            feats = self.pressure_model_features_[target]
            X = ds[feats] if feats else pd.DataFrame(index=ds.index)
            out[target] = ds[target] - model.predict(X)
        return out

    def _compute_flow_residuals(self, ds: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=ds.index)
        if not self.flow_cols_:
            return out
        pressure_summary = pd.DataFrame(index=ds.index)
        if self.pressure_cols_:
            pressure_summary["pressure_mean"] = ds[self.pressure_cols_].mean(axis=1)
            pressure_summary["pressure_min"] = ds[self.pressure_cols_].min(axis=1)
        for target, model in self.flow_models_.items():
            feats = self.flow_model_features_[target]
            feat_blocks: List[pd.DataFrame] = []
            if feats:
                cols_from_ds = [c for c in feats if c in ds.columns]
                cols_from_summary = [c for c in feats if c in pressure_summary.columns]
                if cols_from_ds:
                    feat_blocks.append(ds[cols_from_ds])
                if cols_from_summary:
                    feat_blocks.append(pressure_summary[cols_from_summary])
            X = pd.concat(feat_blocks, axis=1) if feat_blocks else pd.DataFrame(index=ds.index)
            out[target] = ds[target] - model.predict(X)
        return out

    def _compute_pairwise_residual_scores(self, ds: pd.DataFrame) -> pd.DataFrame:
        scores = pd.DataFrame(0.0, index=ds.index, columns=self.pressure_cols_)
        for target in self.pressure_cols_:
            anchor_scores = []
            for anchor, model in self.pairwise_models_.get(target, {}).items():
                feats = self.pairwise_features_[target][anchor]
                X = ds[feats]
                resid = ds[target] - model.predict(X)
                scale = float(self.pairwise_resid_scale_.get(f"{target}|{anchor}", 1.0))
                # actual pressure lower than predicted => likely leak impact near target
                s = np.clip((-resid / max(scale, 1e-3)).to_numpy(dtype=float), 0.0, None)
                anchor_scores.append(pd.Series(s, index=ds.index))
            if anchor_scores:
                stacked = pd.concat(anchor_scores, axis=1)
                scores[target] = stacked.median(axis=1)
        return scores

    def _is_night(self, index: pd.DatetimeIndex) -> pd.Series:
        hours = index.hour
        return pd.Series((hours >= self.config.night_start_hour) & (hours < self.config.night_end_hour), index=index)

    def _compute_feature_tables(self, smooth: pd.DataFrame) -> Dict[str, pd.DataFrame | pd.Series]:
        if self.seasonal_ is None:
            raise RuntimeError("Model must be fit before scoring.")

        ds, _, _ = self.seasonal_.transform(smooth[self.all_cols_])

        pressure_resid = self._compute_pressure_residuals(ds)
        flow_resid = self._compute_flow_residuals(ds)

        pressure_score = pd.DataFrame(index=ds.index)
        for c in self.pressure_cols_:
            scale = float(self.pressure_resid_scale_.get(c, 1.0))
            pressure_score[c] = np.clip((-pressure_resid[c] / max(scale, 1e-3)).to_numpy(dtype=float), 0.0, None)

        flow_score = pd.DataFrame(index=ds.index)
        for c in self.flow_cols_:
            scale = float(self.flow_resid_scale_.get(c, 1.0))
            flow_score[c] = np.clip((flow_resid[c] / max(scale, 1e-3)).to_numpy(dtype=float), 0.0, None)

        pairwise_score = self._compute_pairwise_residual_scores(ds)

        weekly_diff = pressure_resid - pressure_resid.shift(self.config.weekly_period)
        weekly_score = pd.DataFrame(index=ds.index)
        for c in self.pressure_cols_:
            scale = float(self.weekly_diff_scale_.get(c, 1.0))
            weekly_score[c] = np.clip((-(weekly_diff[c]) / max(scale, 1e-3)).to_numpy(dtype=float), 0.0, None)

        ewma_score = weekly_score.ewm(alpha=self.config.ewma_alpha, adjust=False).mean()

        slope_diff = pressure_resid - pressure_resid.shift(max(1, self.config.slope_window - 1))
        slope_score = pd.DataFrame(index=ds.index)
        for c in self.pressure_cols_:
            scale = float(self.slope_diff_scale_.get(c, 1.0))
            slope_score[c] = np.clip((-(slope_diff[c]) / max(scale, 1e-3)).to_numpy(dtype=float), 0.0, None)

        night_mask = self._is_night(ds.index)

        return {
            "pressure_resid": pressure_resid,
            "flow_resid": flow_resid,
            "pressure_score": pressure_score.fillna(0.0),
            "flow_score": flow_score.fillna(0.0),
            "pairwise_score": pairwise_score.fillna(0.0),
            "weekly_score": weekly_score.fillna(0.0),
            "ewma_score": ewma_score.fillna(0.0),
            "slope_score": slope_score.fillna(0.0),
            "night_mask": night_mask,
        }

    def _aggregate_scores(
        self,
        feats: Dict[str, pd.DataFrame | pd.Series],
        cusum_allowance: Optional[float] = None,
    ) -> pd.DataFrame:
        pressure_score = feats["pressure_score"]
        flow_score = feats["flow_score"]
        pairwise_score = feats["pairwise_score"]
        ewma_score = feats["ewma_score"]
        slope_score = feats["slope_score"]
        night_mask = feats["night_mask"]

        k = max(self.config.min_top_k, int(math.ceil(self.config.top_k_fraction * max(1, len(self.pressure_cols_)))))
        flow_score_active = flow_score[self.flow_score_cols_] if self.flow_score_cols_ else pd.DataFrame(index=flow_score.index)
        flow_k = max(1, min(len(self.flow_score_cols_), k)) if self.flow_score_cols_ else 1

        pressure_topk = topk_mean(pressure_score, k)
        pairwise_topk = topk_mean(pairwise_score, k)
        flow_topk = topk_mean(flow_score_active, flow_k) if not flow_score_active.empty else pd.Series(0.0, index=flow_score.index)
        ewma_topk = topk_mean(ewma_score, k)
        slope_topk = topk_mean(slope_score, k)

        burst_raw = (
            self.config.weight_pressure * pressure_topk
            + self.config.weight_pairwise * pairwise_topk
            + self.config.weight_flow * flow_topk
            + self.config.weight_night_flow * (flow_topk * night_mask.astype(float))
        )
        allowance = self.cusum_allowance_ if cusum_allowance is None else float(cusum_allowance)
        burst_cusum = self._positive_cusum(
            burst_raw,
            allowance=allowance,
            decay=self.config.cusum_decay,
        )
        score_burst_trigger = pd.concat([burst_raw, burst_cusum], axis=1).max(axis=1)
        score_burst = burst_raw

        score_incipient = ewma_topk + self.config.weight_slope * slope_topk
        score_final = pd.concat([score_burst, score_incipient], axis=1).max(axis=1)

        return pd.DataFrame(
            {
                "pressure_topk": pressure_topk,
                "pairwise_topk": pairwise_topk,
                "flow_topk": flow_topk,
                "ewma_topk": ewma_topk,
                "slope_topk": slope_topk,
                "burst_raw": burst_raw,
                "burst_cusum": burst_cusum,
                "score_burst_trigger": score_burst_trigger,
                "score_burst": score_burst,
                "score_incipient": score_incipient,
                "score_final": score_final,
            },
            index=pressure_topk.index,
        )

    @staticmethod
    def _positive_cusum(x: pd.Series, allowance: float = 0.5, decay: float = 0.85) -> pd.Series:
        out = np.zeros(len(x), dtype=float)
        xv = x.to_numpy(dtype=float)
        for i in range(1, len(xv)):
            out[i] = max(0.0, out[i - 1] * decay + xv[i] - allowance)
        return pd.Series(out, index=x.index)

    # -----------------------------
    # Detection / scoring
    # -----------------------------

    def score(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if not self.fitted_:
            raise RuntimeError("Model must be fit before score().")
        df = self._prepare_dataframe(df)
        smooth = causal_median_smooth(df[self.all_cols_], self.config.smooth_window)
        feats = self._compute_feature_tables(smooth)
        agg = self._aggregate_scores(feats)

        sensor_evidence = (
            feats["pressure_score"]
            + feats["pairwise_score"]
            + 0.5 * feats["ewma_score"]
            + 0.25 * feats["slope_score"]
        )
        sensor_votes = (sensor_evidence >= self.config.sensor_vote_threshold).sum(axis=1)

        leak_state = self._state_machine(
            score_burst_trigger=agg["score_burst_trigger"],
            score_burst=agg["score_burst"],
            score_incipient=agg["score_incipient"],
            score_final=agg["score_final"],
            flow_topk=agg["flow_topk"],
            sensor_votes=sensor_votes,
        )
        leak_state = self._merge_small_gaps(leak_state, max_gap=self.config.merge_gap_samples)
        event_id = self._event_ids(leak_state)

        top_sensor_cols = {}
        top_area_guess = []
        top_sensor_lists = []
        n_out = min(self.config.top_sensors_output, max(1, len(self.pressure_cols_)))
        arr = sensor_evidence.to_numpy(dtype=float)
        cols = np.array(sensor_evidence.columns)
        for row in arr:
            idx = np.argsort(row)[::-1][:n_out]
            sensors = [str(cols[i]) for i in idx if row[i] > 0]
            sensors = sensors[:n_out]
            top_sensor_lists.append(sensors)
            areas = [self.config.sensor_areas.get(s) for s in sensors if self.config.sensor_areas.get(s)]
            top_area_guess.append(pd.Series(areas).mode().iat[0] if areas else None)
        for j in range(n_out):
            top_sensor_cols[f"top_sensor_{j + 1}"] = [lst[j] if j < len(lst) else None for lst in top_sensor_lists]

        results = pd.concat([agg], axis=1)
        results["sensor_votes"] = sensor_votes
        results["leak_state"] = leak_state.astype(int)
        results["event_id"] = event_id
        results["area_guess"] = top_area_guess
        for c, vals in top_sensor_cols.items():
            results[c] = vals

        sensor_detail = sensor_evidence.copy()
        sensor_detail["event_id"] = event_id
        sensor_detail["leak_state"] = leak_state.astype(int)

        return results, sensor_detail

    def detect(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        results, sensor_detail = self.score(df)
        events = self.summarize_events(results)
        return results, sensor_detail, events

    def _state_machine(
        self,
        score_burst_trigger: pd.Series,
        score_burst: pd.Series,
        score_incipient: pd.Series,
        score_final: pd.Series,
        flow_topk: pd.Series,
        sensor_votes: pd.Series,
    ) -> pd.Series:
        burst_fire = score_burst_trigger >= self.burst_start_threshold_
        incip_fire = score_incipient >= self.incipient_start_threshold_

        burst_confirm = rolling_vote(
            burst_fire,
            window=self.config.burst_confirm_window,
            needed=self.config.burst_confirm_needed,
        )
        incip_confirm = rolling_vote(
            incip_fire,
            window=self.config.incipient_confirm_window,
            needed=self.config.incipient_confirm_needed,
        )

        active = np.zeros(len(score_final), dtype=bool)
        quiet = 0
        for i in range(len(score_final)):
            flow_hit = bool(self.has_flow_evidence_ and (flow_topk.iat[i] >= self.flow_hold_threshold_))
            enough_start_votes = sensor_votes.iat[i] >= self.config.start_sensor_votes
            enough_hold_votes = sensor_votes.iat[i] >= self.config.hold_sensor_votes
            if not active[i - 1] if i > 0 else True:
                start_burst = bool(burst_confirm.iat[i] and (enough_start_votes or flow_hit))
                start_incip = bool(incip_confirm.iat[i] and enough_start_votes)
                if start_burst or start_incip:
                    active[i] = True
                    quiet = 0
                else:
                    active[i] = False
            else:
                active[i] = True
                keep = (
                    (score_final.iat[i] >= self.hold_threshold_)
                    or flow_hit
                    or enough_hold_votes
                )
                if keep:
                    quiet = 0
                else:
                    quiet += 1
                    if quiet >= self.config.quiet_end_samples:
                        active[i] = False
                        quiet = 0
        return pd.Series(active, index=score_final.index)

    @staticmethod
    def _merge_small_gaps(state: pd.Series, max_gap: int) -> pd.Series:
        x = state.astype(int).to_numpy()
        n = len(x)
        i = 0
        while i < n:
            if x[i] == 0:
                j = i
                while j < n and x[j] == 0:
                    j += 1
                gap = j - i
                if i > 0 and j < n and gap <= max_gap:
                    x[i:j] = 1
                i = j
            else:
                i += 1
        return pd.Series(x.astype(bool), index=state.index)

    @staticmethod
    def _event_ids(state: pd.Series) -> pd.Series:
        event_id = []
        cur = 0
        prev = False
        for v in state.astype(bool):
            if v and not prev:
                cur += 1
            event_id.append(cur if v else 0)
            prev = bool(v)
        return pd.Series(event_id, index=state.index)

    # -----------------------------
    # Event summaries and Stage-2 scheduling
    # -----------------------------

    def summarize_events(self, results: pd.DataFrame) -> pd.DataFrame:
        active = results[results["event_id"] > 0]
        if active.empty:
            return pd.DataFrame(
                columns=[
                    "event_id",
                    "start_time",
                    "end_time",
                    "duration_hours",
                    "peak_time",
                    "peak_score",
                    "area_guess",
                    "top_sensors",
                ]
            )

        rows = []
        sensor_cols = [c for c in results.columns if c.startswith("top_sensor_")]
        for eid, block in active.groupby("event_id"):
            sensor_flat = [s for c in sensor_cols for s in block[c].dropna().tolist()]
            area_vals = block["area_guess"].dropna()
            rows.append(
                {
                    "event_id": int(eid),
                    "start_time": block.index.min(),
                    "end_time": block.index.max(),
                    "duration_hours": (block.index.max() - block.index.min()).total_seconds() / 3600.0,
                    "peak_time": block["score_final"].idxmax(),
                    "peak_score": float(block["score_final"].max()),
                    "area_guess": area_vals.mode().iat[0] if not area_vals.empty else None,
                    "top_sensors": ", ".join(pd.Series(sensor_flat).value_counts().head(self.config.top_sensors_output).index.tolist()),
                }
            )
        return pd.DataFrame(rows)

    def build_stage2_triggers(self, results: pd.DataFrame) -> pd.DataFrame:
        sensor_cols = [c for c in results.columns if c.startswith("top_sensor_")]
        if not sensor_cols:
            return pd.DataFrame(columns=["timestamp", "event_id", "reason", "area_guess", "top_sensors"])

        triggers = []
        last_trigger_time_by_event: Dict[int, pd.Timestamp] = {}
        last_sensor_set_by_event: Dict[int, List[str]] = {}
        min_shift_gap = pd.Timedelta(minutes=self.config.min_sensor_shift_minutes)
        recheck_gap = pd.Timedelta(hours=self.config.stage2_recheck_hours)

        for ts, row in results.iterrows():
            event_id = int(row["event_id"])
            if event_id <= 0:
                continue
            sensors = [row[c] for c in sensor_cols if isinstance(row[c], str) and row[c]]
            reason = None
            if event_id not in last_trigger_time_by_event:
                reason = "event_start"
            else:
                dt = ts - last_trigger_time_by_event[event_id]
                if dt >= recheck_gap:
                    reason = "periodic_recheck"
                else:
                    sim = jaccard_similarity(sensors, last_sensor_set_by_event[event_id])
                    if dt >= min_shift_gap and sim < self.config.sensor_shift_jaccard_threshold:
                        reason = "sensor_shift"

            if reason is not None:
                triggers.append(
                    {
                        "timestamp": ts,
                        "event_id": event_id,
                        "reason": reason,
                        "area_guess": row.get("area_guess"),
                        "top_sensors": ", ".join(sensors),
                    }
                )
                last_trigger_time_by_event[event_id] = ts
                last_sensor_set_by_event[event_id] = sensors

        return pd.DataFrame(triggers)

    # -----------------------------
    # Persistence
    # -----------------------------

    def save_metadata(self, path: str | Path) -> None:
        if not self.fitted_:
            raise RuntimeError("Model must be fit before save_metadata().")
        payload = {
            "config": asdict(self.config),
            "pressure_cols": self.pressure_cols_,
            "flow_cols": self.flow_cols_,
            "flow_score_cols": self.flow_score_cols_,
            "auto_excluded_flow_score_cols": self.auto_excluded_flow_score_cols_,
            "level_cols": self.level_cols_,
            "demand_cols": self.demand_cols_,
            "thresholds": {
                "burst_start_threshold": self.burst_start_threshold_,
                "incipient_start_threshold": self.incipient_start_threshold_,
                "hold_threshold": self.hold_threshold_,
                "flow_hold_threshold": self.flow_hold_threshold_,
            },
        }
        Path(path).write_text(json.dumps(payload, default=str, indent=2), encoding="utf-8")


# -----------------------------------------------------------------------------
# Example CLI
# -----------------------------------------------------------------------------


def _run_cli() -> None:
    parser = argparse.ArgumentParser(description="Recall-first BattLeDIM Stage 1 detector")
    parser.add_argument("--data-root", type=str, default=None, help="Folder with BattLeDIM split CSVs")
    parser.add_argument("--merged-csv", type=str, default=None, help="Single merged BattLeDIM-style CSV")
    parser.add_argument("--timestamp-col", type=str, default=None)
    parser.add_argument("--train-year", type=int, default=2018)
    parser.add_argument("--test-year", type=int, default=2019)
    parser.add_argument("--fixed-leak-report", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="stage1_outputs")
    args = parser.parse_args()

    if not args.data_root and not args.merged_csv:
        parser.error("Pass either --data-root for split BattLeDIM files or --merged-csv for one merged table.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.merged_csv:
        full_df = load_battledim_merged_csv(args.merged_csv, timestamp_col=args.timestamp_col)
        train_df = full_df[full_df.index.year == args.train_year].copy()
        test_df = full_df[full_df.index.year == args.test_year].copy()
        if train_df.empty or test_df.empty:
            raise ValueError("Merged CSV does not contain both requested years.")
        intervals = build_known_intervals_from_leak_columns(train_df, pad_before="30min", pad_after="30min")
        cfg = Stage1Config(flow_score_exclude_cols=["flow_3"] if "flow_3" in full_df.columns else [])
    else:
        train_df = load_battledim_split(args.data_root, args.train_year, timestamp_col=args.timestamp_col)
        test_df = load_battledim_split(args.data_root, args.test_year, timestamp_col=args.timestamp_col)
        intervals = []
        if args.fixed_leak_report:
            intervals = parse_fixed_leak_report(args.fixed_leak_report)
        cfg = Stage1Config()

    detector = BattLeDIMStage1(cfg)
    detector.fit(train_df, known_intervals=intervals)
    results, sensor_detail, events = detector.detect(test_df)
    triggers = detector.build_stage2_triggers(results)

    results.to_csv(output_dir / "stage1_scores.csv")
    sensor_detail.to_csv(output_dir / "stage1_sensor_detail.csv")
    events.to_csv(output_dir / "stage1_events.csv", index=False)
    triggers.to_csv(output_dir / "stage2_triggers.csv", index=False)
    detector.save_metadata(output_dir / "stage1_metadata.json")

    if args.merged_csv and (("leak_count" in test_df.columns) or ("leak_mag_total" in test_df.columns) or ("total_leak_mag" in test_df.columns)):
        truth = pd.Series(False, index=test_df.index)
        if "leak_count" in test_df.columns:
            truth |= test_df["leak_count"].fillna(0.0) > 0.0
        if "leak_mag_total" in test_df.columns:
            truth |= test_df["leak_mag_total"].fillna(0.0) > 0.0
        if "total_leak_mag" in test_df.columns:
            truth |= test_df["total_leak_mag"].fillna(0.0) > 0.0
        metrics = evaluate_stage1_results(results, truth, triggers=triggers)
        metrics.to_csv(output_dir / "stage1_metrics.csv", header=["value"])

    print("Wrote:")
    names = [
        "stage1_scores.csv",
        "stage1_sensor_detail.csv",
        "stage1_events.csv",
        "stage2_triggers.csv",
        "stage1_metadata.json",
    ]
    if (output_dir / "stage1_metrics.csv").exists():
        names.append("stage1_metrics.csv")
    for name in names:
        print(output_dir / name)



if __name__ == "__main__":
    _run_cli()
