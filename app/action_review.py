"""Rhythmic-action segmentation and reviewed segment persistence."""
from __future__ import annotations

import csv
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

_EMG_PYTHON = Path(__file__).resolve().parents[1] / "emg_python"
if _EMG_PYTHON.exists() and str(_EMG_PYTHON) not in sys.path:
    sys.path.insert(0, str(_EMG_PYTHON))

from segmentation.reps import detect_reps


ACTION_FILENAME = "action_segments.csv"
ACTION_FIELDS = [
    "trial",
    "mode",
    "reference_channel",
    "segment_index",
    "start_sample",
    "peak_sample",
    "end_sample_exclusive",
    "start_sec",
    "peak_sec",
    "end_sec",
    "status",
    "updated_at",
]


def action_csv_path(trial_dir: str | Path) -> Path:
    return Path(trial_dir).resolve().parent / ACTION_FILENAME


def _read_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.is_file():
        return []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def detect_rhythmic_segments(
    envelope: np.ndarray,
    sample_rate: float,
    valid_start: int,
    valid_end_exclusive: int,
    min_rep_distance_sec: float = 0.6,
) -> list[dict[str, int]]:
    """Detect valley-peak-valley repetitions inside the reviewed time range."""
    values = np.asarray(envelope, dtype=float).ravel()
    if not (
        math.isfinite(sample_rate)
        and sample_rate > 0
        and 0 <= valid_start < valid_end_exclusive <= len(values)
    ):
        raise ValueError("动作检测参数或时间审核范围无效")

    reviewed = values[valid_start:valid_end_exclusive]
    finite = reviewed[np.isfinite(reviewed)]
    if finite.size < max(16, int(sample_rate * 0.2)):
        return []
    quiet_limit = float(np.percentile(finite, 35))
    quiet = finite[finite <= quiet_limit]
    baseline_median = float(np.median(quiet)) if quiet.size else float(np.median(finite))
    baseline_mad = float(np.median(np.abs(quiet - baseline_median))) if quiet.size else 0.0
    if baseline_mad <= np.finfo(float).eps:
        baseline_mad = float(np.std(quiet)) if quiet.size else 0.0
    baseline_mad = max(baseline_mad, abs(baseline_median) * 0.01, 1e-9)

    detected = detect_reps(
        reviewed,
        sample_rate,
        baseline_median,
        baseline_mad,
        min_rep_dist_s=min_rep_distance_sec,
    )
    segments = []
    for start, peak, end in detected:
        global_start = valid_start + start
        global_peak = valid_start + peak
        global_end = min(valid_end_exclusive, valid_start + end)
        if global_start <= global_peak < global_end:
            segments.append({
                "start_sample": int(global_start),
                "peak_sample": int(global_peak),
                "end_sample_exclusive": int(global_end),
            })
    return segments


def load_action_segments(trial_dir: str | Path) -> list[dict]:
    trial = Path(trial_dir).resolve()
    output = []
    for row in _read_rows(action_csv_path(trial)):
        if row.get("trial") != trial.name:
            continue
        try:
            output.append({
                **row,
                "reference_channel": int(row["reference_channel"]),
                "segment_index": int(row["segment_index"]),
                "start_sample": int(row["start_sample"]),
                "peak_sample": int(row["peak_sample"]),
                "end_sample_exclusive": int(row["end_sample_exclusive"]),
                "start_sec": float(row["start_sec"]),
                "peak_sec": float(row["peak_sec"]),
                "end_sec": float(row["end_sec"]),
            })
        except (KeyError, TypeError, ValueError):
            return []
    return sorted(output, key=lambda item: item["segment_index"])


def save_action_segments(
    trial_dir: str | Path,
    reference_channel: int,
    segments: list[dict],
    n_samples: int,
    sample_rate: float,
    mode: str = "rhythmic",
) -> list[dict]:
    """Replace one Trial's reviewed action rows while preserving other Trials."""
    trial = Path(trial_dir).resolve()
    if not (trial / "emg_data.txt").is_file():
        raise FileNotFoundError(f"Trial TXT 不存在: {trial / 'emg_data.txt'}")
    if not segments:
        raise ValueError("至少保留一个动作分段")
    if reference_channel < 0:
        raise ValueError("参考通道无效")
    if not math.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("采样率必须大于 0")

    from time_review import load_time_review

    time_review = load_time_review(trial)
    if not time_review:
        raise ValueError("请先完成该 Trial 的时间切分审查")
    valid_start = int(time_review["start_sample"])
    valid_end = int(time_review["end_sample_exclusive"])

    ordered = sorted(segments, key=lambda item: int(item["start_sample"]))
    previous_end = -1
    for segment in ordered:
        start = int(segment["start_sample"])
        peak = int(segment["peak_sample"])
        end = int(segment["end_sample_exclusive"])
        if not (0 <= start <= peak < end <= n_samples):
            raise ValueError("动作分段边界或峰值样本无效")
        if start < valid_start or end > valid_end:
            raise ValueError("动作分段必须位于已审核时间范围内")
        if start < previous_end:
            raise ValueError("动作分段不能互相重叠")
        previous_end = end

    csv_path = action_csv_path(trial)
    rows = [row for row in _read_rows(csv_path) if row.get("trial") != trial.name]
    updated_at = datetime.now().isoformat(timespec="seconds")
    for index, segment in enumerate(ordered, start=1):
        start = int(segment["start_sample"])
        peak = int(segment["peak_sample"])
        end = int(segment["end_sample_exclusive"])
        rows.append({
            "trial": trial.name,
            "mode": mode,
            "reference_channel": str(reference_channel),
            "segment_index": str(index),
            "start_sample": str(start),
            "peak_sample": str(peak),
            "end_sample_exclusive": str(end),
            "start_sec": f"{start / sample_rate:.6f}",
            "peak_sec": f"{peak / sample_rate:.6f}",
            "end_sec": f"{end / sample_rate:.6f}",
            "status": "approved",
            "updated_at": updated_at,
        })

    from feature_export import remove_feature_outputs
    remove_feature_outputs(trial)
    temporary = csv_path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=ACTION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(csv_path)
    return load_action_segments(trial)


def remove_action_segments(trial_dir: str | Path) -> None:
    trial = Path(trial_dir).resolve()
    csv_path = action_csv_path(trial)
    if not csv_path.is_file():
        return
    rows = [row for row in _read_rows(csv_path) if row.get("trial") != trial.name]
    temporary = csv_path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=ACTION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(csv_path)
