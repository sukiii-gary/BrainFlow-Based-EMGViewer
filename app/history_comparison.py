"""Read-only discovery and signal preparation for historical Trial comparison."""
from __future__ import annotations

import csv
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, Sequence

import numpy as np

_EMG_PYTHON = Path(__file__).resolve().parents[1] / "emg_python"
if _EMG_PYTHON.exists() and str(_EMG_PYTHON) not in sys.path:
    sys.path.insert(0, str(_EMG_PYTHON))

from preprocessing.filters import apply_filters

from action_review import load_action_segments
from time_review import TrialData, load_time_review, load_trial_txt


@dataclass(frozen=True)
class HistoryTrial:
    trial_dir: Path
    txt_path: Path
    session_name: str
    trial_name: str
    group: str
    subject_id: str
    task: str
    started_at: str
    sample_rate: float
    n_channels: int
    n_samples: int
    duration_sec: float
    channel_names: tuple[str, ...]
    time_review: dict | None
    action_count: int
    error: str | None = None

    @property
    def selectable(self) -> bool:
        return self.error is None and self.n_samples > 0


@dataclass(frozen=True)
class SegmentRef:
    trial: HistoryTrial
    source: Literal["full", "review", "action"]
    label: str
    start_sample: int
    end_sample_exclusive: int
    peak_sample: int | None = None

    @property
    def n_samples(self) -> int:
        return self.end_sample_exclusive - self.start_sample


@dataclass(frozen=True)
class LoadedSegment:
    ref: SegmentRef
    emg: np.ndarray
    sample_rate: float
    channel_names: tuple[str, ...]
    sample_indices: np.ndarray
    timestamps: np.ndarray


@dataclass(frozen=True)
class ComparisonSeries:
    ref: SegmentRef
    label: str
    time_sec: np.ndarray
    values: np.ndarray
    sample_rate: float
    channel_names: tuple[str, ...]


@dataclass(frozen=True)
class ComparisonResult:
    mode: Literal["raw", "filtered", "rms"]
    alignment: Literal["start", "peak"]
    common_band_hz: tuple[float, float] | None
    channel_names: tuple[str, ...]
    series: tuple[ComparisonSeries, ...]


def _metadata_value(line: str, key: str, next_key: str | None = None) -> str:
    if next_key:
        pattern = rf"(?:^|;)\s*{re.escape(key)}\s*=\s*(.*?)(?=;\s*{re.escape(next_key)}\s*=)"
    else:
        pattern = rf"(?:^|;)\s*{re.escape(key)}\s*=\s*(.*)$"
    match = re.search(pattern, line, flags=re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _read_trial_summary(txt_path: Path) -> HistoryTrial:
    trial_dir = txt_path.parent.resolve()
    session_name = trial_dir.parent.name
    metadata = {"group": "", "subject": "", "task": "", "start": ""}
    sample_rate = math.nan
    declared_channels = None
    columns: list[str] | None = None
    data_rows = 0

    try:
        with txt_path.open("r", encoding="utf-8-sig", newline="") as source:
            for line in source:
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith("%"):
                    rate = re.search(
                        r"Sample\s*Rate\s*=\s*([0-9]+(?:\.[0-9]+)?)",
                        stripped,
                        flags=re.IGNORECASE,
                    )
                    if rate:
                        sample_rate = float(rate.group(1))
                    count = re.search(
                        r"Number\s+of\s+channels\s*=\s*(\d+)",
                        stripped,
                        flags=re.IGNORECASE,
                    )
                    if count:
                        declared_channels = int(count.group(1))
                    if "Trial" in stripped and "Start" in stripped:
                        metadata["start"] = _metadata_value(stripped, "Start", "Group")
                        metadata["group"] = _metadata_value(stripped, "Group", "Subject")
                        metadata["subject"] = _metadata_value(stripped, "Subject", "Task")
                        metadata["task"] = _metadata_value(stripped, "Task")
                    continue
                if columns is None:
                    columns = next(csv.reader([stripped]))
                else:
                    data_rows += 1

        if not columns or len(columns) < 3:
            raise ValueError("缺少有效列头")
        channel_names = tuple(item.strip() for item in columns[1:-1])
        if columns[0].strip().lower() != "sample index" or columns[-1].strip().lower() != "timestamp":
            raise ValueError("列头不是 Recorder 的 Sample Index/EXG/Timestamp 格式")
        if declared_channels is not None and declared_channels != len(channel_names):
            raise ValueError("声明通道数与列头不一致")
        if not math.isfinite(sample_rate) or sample_rate <= 0:
            raise ValueError("缺少有效采样率")
        if data_rows <= 0:
            raise ValueError("没有有效数据行")

        review = load_time_review(trial_dir)
        actions = load_action_segments(trial_dir)
        return HistoryTrial(
            trial_dir=trial_dir,
            txt_path=txt_path.resolve(),
            session_name=session_name,
            trial_name=trial_dir.name,
            group=metadata["group"],
            subject_id=metadata["subject"],
            task=metadata["task"] or trial_dir.name,
            started_at=metadata["start"],
            sample_rate=sample_rate,
            n_channels=len(channel_names),
            n_samples=data_rows,
            duration_sec=data_rows / sample_rate,
            channel_names=channel_names,
            time_review=review,
            action_count=len(actions),
        )
    except (OSError, UnicodeError, ValueError, csv.Error) as exc:
        return HistoryTrial(
            trial_dir=trial_dir,
            txt_path=txt_path.resolve(),
            session_name=session_name,
            trial_name=trial_dir.name,
            group=metadata["group"],
            subject_id=metadata["subject"],
            task=metadata["task"] or trial_dir.name,
            started_at=metadata["start"],
            sample_rate=sample_rate,
            n_channels=0,
            n_samples=0,
            duration_sec=0.0,
            channel_names=(),
            time_review=None,
            action_count=0,
            error=str(exc),
        )


def discover_history(
    recordings_root: str | Path,
    exclude_trial: str | Path | None = None,
) -> list[HistoryTrial]:
    """Discover this version's Session/Trial hierarchy without loading waveforms."""
    root = Path(recordings_root).expanduser().resolve()
    if not root.is_dir():
        return []
    excluded = Path(exclude_trial).resolve() if exclude_trial is not None else None
    trials = []
    for txt_path in root.glob("*/*/emg_data.txt"):
        resolved_txt = txt_path.resolve()
        if not resolved_txt.is_relative_to(root):
            continue
        if excluded is not None and resolved_txt.parent == excluded:
            continue
        trials.append(_read_trial_summary(resolved_txt))
    return sorted(
        trials,
        key=lambda item: (item.started_at, item.session_name, item.trial_name),
        reverse=True,
    )


def list_segment_candidates(trial: HistoryTrial) -> list[SegmentRef]:
    """Return full-Trial, reviewed-window and reviewed-action candidates."""
    if not trial.selectable:
        return []
    output = [SegmentRef(
        trial=trial,
        source="full",
        label=f"{trial.session_name} / {trial.trial_name} / 完整 Trial",
        start_sample=0,
        end_sample_exclusive=trial.n_samples,
    )]
    review = load_time_review(trial.trial_dir)
    if review:
        start = int(review["start_sample"])
        end = int(review["end_sample_exclusive"])
        if 0 <= start < end <= trial.n_samples:
            output.append(SegmentRef(
                trial=trial,
                source="review",
                label=f"{trial.session_name} / {trial.trial_name} / 审核时间窗",
                start_sample=start,
                end_sample_exclusive=end,
            ))
    for action in load_action_segments(trial.trial_dir):
        start = int(action["start_sample"])
        peak = int(action["peak_sample"])
        end = int(action["end_sample_exclusive"])
        if 0 <= start <= peak < end <= trial.n_samples:
            output.append(SegmentRef(
                trial=trial,
                source="action",
                label=(f"{trial.session_name} / {trial.trial_name} / "
                       f"动作 {int(action['segment_index'])}"),
                start_sample=start,
                end_sample_exclusive=end,
                peak_sample=peak,
            ))
    return output


def _load_trial_data(txt_path: Path) -> TrialData:
    """Load through the shared UTF-8 Trial reader."""
    return load_trial_txt(txt_path)


def load_segment(ref: SegmentRef) -> LoadedSegment:
    """Load only one selected segment and leave every source file untouched."""
    data = _load_trial_data(ref.trial.txt_path)
    start = int(ref.start_sample)
    end = int(ref.end_sample_exclusive)
    if not (0 <= start < end <= data.n_samples):
        raise ValueError(f"历史分段越界: {start}:{end}, Trial 样本数 {data.n_samples}")
    emg = np.asarray(data.emg[start:end], dtype=float)
    if emg.ndim != 2 or emg.shape[1] != len(data.channel_names):
        raise ValueError("历史分段通道结构无效")
    if not np.all(np.isfinite(emg)):
        raise ValueError("历史分段包含 NaN 或无穷值")
    return LoadedSegment(
        ref=ref,
        emg=emg,
        sample_rate=float(data.sample_rate),
        channel_names=data.channel_names,
        sample_indices=data.sample_indices[start:end],
        timestamps=data.timestamps[start:end],
    )


def _common_channels(segments: Sequence[LoadedSegment]) -> tuple[str, ...]:
    first = segments[0].channel_names
    others = [set(item.channel_names) for item in segments[1:]]
    return tuple(name for name in first if all(name in names for names in others))


def _moving_rms(values: np.ndarray, window: int) -> np.ndarray:
    kernel = np.ones(window, dtype=float)
    output = np.empty_like(values, dtype=float)
    normalizer = np.convolve(np.ones(values.shape[0]), kernel, mode="same")
    for channel in range(values.shape[1]):
        squared = np.square(values[:, channel])
        output[:, channel] = np.sqrt(
            np.convolve(squared, kernel, mode="same") / normalizer
        )
    return output


def prepare_comparison(
    refs: Sequence[SegmentRef],
    mode: Literal["raw", "filtered", "rms"] = "raw",
    alignment: Literal["start", "peak"] = "start",
    viewer_config=None,
    rms_window_ms: float = 100.0,
) -> ComparisonResult:
    """Lazily load selected segments and prepare directly comparable series."""
    if not refs:
        raise ValueError("至少选择一个历史分段")
    if mode not in {"raw", "filtered", "rms"}:
        raise ValueError(f"不支持的比较模式: {mode}")
    if alignment not in {"start", "peak"}:
        raise ValueError(f"不支持的对齐方式: {alignment}")
    loaded_trials: dict[Path, TrialData] = {}
    for ref in refs:
        key = ref.trial.txt_path.resolve()
        if key not in loaded_trials:
            data = _load_trial_data(key)
            if data.emg.ndim != 2 or data.emg.shape[1] != len(data.channel_names):
                raise ValueError(f"历史 Trial 通道结构无效: {key}")
            if not np.all(np.isfinite(data.emg)):
                raise ValueError(f"历史 Trial 包含 NaN 或无穷值: {key}")
            loaded_trials[key] = data
        data = loaded_trials[key]
        if not (0 <= ref.start_sample < ref.end_sample_exclusive <= data.n_samples):
            raise ValueError(
                f"历史分段越界: {ref.start_sample}:{ref.end_sample_exclusive}, "
                f"Trial 样本数 {data.n_samples}"
            )

    loaded_for_channels = [SimpleNamespace(channel_names=data.channel_names)
                           for data in loaded_trials.values()]
    channels = _common_channels(loaded_for_channels)
    if not channels:
        raise ValueError("所选历史分段没有共同通道")

    common_band = None
    if mode != "raw":
        low = float(getattr(viewer_config, "bandpass_low", 20.0))
        requested_high = float(getattr(viewer_config, "bandpass_high", 450.0))
        high = min(
            requested_high,
            *(data.sample_rate / 2.0 - 1.0 for data in loaded_trials.values()),
        )
        if not (math.isfinite(low) and math.isfinite(high) and 0 < low < high):
            raise ValueError(f"所选分段没有有效共同频带: {low:g}-{high:g} Hz")
        common_band = (low, high)
    if not math.isfinite(rms_window_ms) or rms_window_ms <= 0:
        raise ValueError("RMS 窗宽必须大于 0")

    processed_trials: dict[Path, np.ndarray] = {}
    for key, data in loaded_trials.items():
        indices = [data.channel_names.index(name) for name in channels]
        raw = np.asarray(data.emg[:, indices], dtype=float)
        if mode == "raw":
            processed_trials[key] = raw.copy()
        else:
            cfg = SimpleNamespace(
                sampling_rate=data.sample_rate,
                preprocess=SimpleNamespace(
                    notch_hz=float(getattr(viewer_config, "notch_hz", 50.0)),
                    notch_bandwidth_hz=float(
                        getattr(viewer_config, "notch_bandwidth_hz", 8.0)
                    ),
                    notch_harmonics=int(
                        getattr(viewer_config, "notch_harmonics", 3)
                    ),
                    bandpass=common_band,
                ),
            )
            values = np.column_stack([
                apply_filters(raw[:, channel], cfg, do_notch=True)
                for channel in range(raw.shape[1])
            ])
            if mode == "rms":
                window = max(1, int(round(data.sample_rate * rms_window_ms / 1000.0)))
                values = _moving_rms(values, min(window, len(values)))
            processed_trials[key] = values

    output = []
    for ref in refs:
        key = ref.trial.txt_path.resolve()
        data = loaded_trials[key]
        values = processed_trials[key][ref.start_sample:ref.end_sample_exclusive]
        if alignment == "start":
            origin = 0
        else:
            if ref.peak_sample is None:
                raise ValueError(f"分段没有已审核峰值，无法峰值对齐: {ref.label}")
            origin = ref.peak_sample - ref.start_sample
            if not 0 <= origin < len(values):
                raise ValueError(f"分段峰值越界: {ref.label}")
        time_sec = (np.arange(len(values), dtype=float) - origin) / data.sample_rate
        output.append(ComparisonSeries(
            ref=ref,
            label=ref.label,
            time_sec=time_sec,
            values=values,
            sample_rate=data.sample_rate,
            channel_names=channels,
        ))

    return ComparisonResult(
        mode=mode,
        alignment=alignment,
        common_band_hz=common_band,
        channel_names=channels,
        series=tuple(output),
    )
