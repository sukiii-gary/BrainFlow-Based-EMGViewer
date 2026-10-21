"""Trial TXT loading and reviewed time-range persistence."""
from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np


REVIEW_FILENAME = "review_segments.csv"
REVIEW_FIELDS = [
    "trial",
    "txt_file",
    "start_sample",
    "end_sample_exclusive",
    "start_sec",
    "end_sec",
    "sample_rate_hz",
    "status",
    "updated_at",
]


@dataclass(frozen=True)
class TrialData:
    sample_indices: np.ndarray
    emg: np.ndarray
    timestamps: np.ndarray
    sample_rate: float
    channel_names: tuple[str, ...]

    @property
    def n_samples(self) -> int:
        return int(self.emg.shape[0])

    @property
    def duration_sec(self) -> float:
        return self.n_samples / self.sample_rate


def load_trial_txt(txt_path: str | Path) -> TrialData:
    """Load the raw TXT created by Recorder without altering it."""
    path = Path(txt_path)
    if not path.is_file():
        raise FileNotFoundError(f"Trial TXT 不存在: {path}")

    sample_rate = None
    columns = None
    skiprows = 0
    with path.open("r", encoding="utf-8-sig") as source:
        for line_number, line in enumerate(source, start=1):
            stripped = line.strip()
            if stripped.startswith("%"):
                match = re.search(
                    r"Sample\s*Rate\s*=\s*([0-9]+(?:\.[0-9]+)?)",
                    stripped,
                    flags=re.IGNORECASE,
                )
                if match:
                    sample_rate = float(match.group(1))
                continue
            if stripped:
                columns = next(csv.reader([stripped]))
                skiprows = line_number
                break

    if not columns or len(columns) < 3:
        raise ValueError("Trial TXT 缺少有效列头")
    values = np.loadtxt(
        path,
        delimiter=",",
        skiprows=skiprows,
        dtype=np.float64,
        ndmin=2,
        encoding="utf-8-sig",
    )
    if values.shape[0] == 0 or values.shape[1] != len(columns):
        raise ValueError("Trial TXT 没有有效数据，或数据列数与列头不一致")

    timestamps = values[:, -1]
    if sample_rate is None:
        deltas = np.diff(timestamps)
        valid = deltas[np.isfinite(deltas) & (deltas > 0)]
        if valid.size == 0:
            raise ValueError("Trial TXT 缺少采样率，且无法从时间戳推断")
        sample_rate = 1.0 / float(np.median(valid))
    if not math.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError(f"Trial TXT 采样率无效: {sample_rate}")

    return TrialData(
        sample_indices=np.rint(values[:, 0]).astype(np.int64),
        emg=values[:, 1:-1],
        timestamps=timestamps,
        sample_rate=sample_rate,
        channel_names=tuple(columns[1:-1]),
    )


def default_middle_window(n_samples: int) -> tuple[int, int]:
    """Return an inclusive/exclusive sample range retaining middle 70%."""
    if n_samples <= 0:
        raise ValueError("Trial 必须至少包含一个样本")
    keep = max(1, int(round(n_samples * 0.70)))
    start = (n_samples - keep) // 2
    return start, start + keep


def review_csv_path(trial_dir: str | Path) -> Path:
    return Path(trial_dir).resolve().parent / REVIEW_FILENAME


def _read_reviews(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.is_file():
        return []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def load_time_review(trial_dir: str | Path) -> dict | None:
    trial = Path(trial_dir).resolve()
    for row in _read_reviews(review_csv_path(trial)):
        if row.get("trial") == trial.name:
            try:
                return {
                    **row,
                    "start_sample": int(row["start_sample"]),
                    "end_sample_exclusive": int(row["end_sample_exclusive"]),
                    "start_sec": float(row["start_sec"]),
                    "end_sec": float(row["end_sec"]),
                    "sample_rate_hz": float(row["sample_rate_hz"]),
                }
            except (KeyError, TypeError, ValueError):
                return None
    return None


def save_time_review(
    trial_dir: str | Path,
    start_sample: int,
    end_sample_exclusive: int,
    n_samples: int,
    sample_rate: float,
) -> dict:
    """Create or update one reviewed Trial row without touching raw TXT."""
    trial = Path(trial_dir).resolve()
    txt_path = trial / "emg_data.txt"
    if not txt_path.is_file():
        raise FileNotFoundError(f"Trial TXT 不存在: {txt_path}")
    if not (0 <= start_sample < end_sample_exclusive <= n_samples):
        raise ValueError("时间边界必须满足 0 <= 起点 < 终点 <= Trial 样本数")
    if not math.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("采样率必须大于 0")

    previous = load_time_review(trial)
    csv_path = review_csv_path(trial)
    rows = [row for row in _read_reviews(csv_path) if row.get("trial") != trial.name]
    record = {
        "trial": trial.name,
        "txt_file": str(txt_path),
        "start_sample": str(int(start_sample)),
        "end_sample_exclusive": str(int(end_sample_exclusive)),
        "start_sec": f"{start_sample / sample_rate:.6f}",
        "end_sec": f"{end_sample_exclusive / sample_rate:.6f}",
        "sample_rate_hz": f"{sample_rate:.6f}",
        "status": "approved",
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    rows.append(record)

    range_changed = previous and (
        previous["start_sample"] != start_sample
        or previous["end_sample_exclusive"] != end_sample_exclusive
    )
    if range_changed:
        from action_review import remove_action_segments
        from feature_export import remove_feature_outputs
        remove_feature_outputs(trial)
        remove_action_segments(trial)

    temporary = csv_path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(csv_path)
    return load_time_review(trial)


def remove_time_review(trial_dir: str | Path) -> None:
    """Remove a deleted Trial from its Session review CSV."""
    trial = Path(trial_dir).resolve()
    csv_path = review_csv_path(trial)
    if not csv_path.is_file():
        return
    rows = [row for row in _read_reviews(csv_path) if row.get("trial") != trial.name]
    temporary = csv_path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=REVIEW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(csv_path)
