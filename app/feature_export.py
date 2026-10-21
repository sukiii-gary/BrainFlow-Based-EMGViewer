"""Export reviewed EMG features and plots without modifying raw Trial data."""
from __future__ import annotations

import csv
import math
import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import numpy as np

_EMG_PYTHON = Path(__file__).resolve().parents[1] / "emg_python"
if _EMG_PYTHON.exists() and str(_EMG_PYTHON) not in sys.path:
    sys.path.insert(0, str(_EMG_PYTHON))

from preprocessing.envelope import compute_envelope
from preprocessing.filters import apply_filters
from preprocessing.psd import compute_psd
from features.signal_quality import band_power

from action_review import load_action_segments
from time_review import load_time_review, load_trial_txt


RESULTS_DIRNAME = "analysis_results"
FEATURES_FILENAME = "features.csv"
SUPPORTED_METRICS = (
    "raw", "filtered", "envelope", "spectrum", "pli", "mar", "mnf", "mdf",
)
_OUTPUT_FILENAMES = {
    "features": FEATURES_FILENAME,
    "raw": "raw_signal.png",
    "filtered": "filtered_signal.png",
    "envelope": "envelope.png",
    "spectrum": "spectrum_comparison.png",
    "pli": "powerline_interference.png",
    "mar": "motion_artifact.png",
    "mnf": "mnf.png",
    "mdf": "mdf.png",
}

TASK_LABELS = {
    "sit_base": "坐姿静息",
    "std_base": "站姿静息",
    "sit_df": "坐姿背屈",
    "std_df": "站姿背屈",
    "sit_pf": "坐姿跖屈",
    "std_pf": "站姿跖屈",
    "sit_pf_std_pf": "坐姿/站姿跖屈",
    "walk": "平地行走",
    "walk1.5": "跑步机 1.5 km/h 行走",
    "walk3.0": "跑步机 3.0 km/h 行走",
    "walk6.0": "跑步机 6.0 km/h 行走",
    "sit_tf": "坐姿趾屈",
    "std_tf": "站姿趾屈",
    "sit_te": "坐姿趾伸",
    "std_te": "站姿趾伸",
}


@dataclass(frozen=True)
class ExportResult:
    output_dir: Path
    scope: str
    row_count: int
    csv_path: Path | None
    image_paths: dict[str, Path]


def results_dir(trial_dir: str | Path) -> Path:
    return Path(trial_dir).resolve() / RESULTS_DIRNAME


def normalize_task_code(task_name: str) -> str:
    """Normalize a known task code and ignore duplicate-Trial suffixes."""
    code = re.sub(r"\s+", "_", str(task_name or "").strip().lower())
    if code in TASK_LABELS:
        return code
    without_suffix = re.sub(r"_\d+$", "", code)
    return without_suffix if without_suffix in TASK_LABELS else code


def task_display_name(task_name: str) -> str:
    code = normalize_task_code(task_name)
    label = TASK_LABELS.get(code)
    if label:
        return f"{label} ({code})"
    return str(task_name or "").strip() or "未命名任务"


def _read_trial_task(trial_dir: Path) -> str:
    txt_path = trial_dir / "emg_data.txt"
    with txt_path.open("r", encoding="utf-8-sig") as source:
        for line in source:
            if not line.lstrip().startswith("%"):
                break
            match = re.search(r"(?:^|;)\s*Task\s*=\s*([^;\r\n]*)", line)
            if match and match.group(1).strip():
                return match.group(1).strip()
    return trial_dir.name


@lru_cache(maxsize=1)
def _cjk_font_properties():
    """Select an installed CJK font so Chinese task titles render on Windows/macOS."""
    from matplotlib import font_manager
    from matplotlib.font_manager import FontProperties

    for family in ("Microsoft YaHei", "PingFang SC", "SimHei", "Noto Sans CJK SC"):
        try:
            path = font_manager.findfont(family, fallback_to_default=False)
        except ValueError:
            continue
        return FontProperties(fname=path)
    return None


def _task_title_kwargs() -> dict:
    font = _cjk_font_properties()
    return {"fontproperties": font} if font is not None else {}


def remove_feature_outputs(trial_dir: str | Path) -> None:
    """Remove only files owned by this exporter, preserving unrelated files."""
    output_dir = results_dir(trial_dir)
    for filename in _OUTPUT_FILENAMES.values():
        path = output_dir / filename
        if path.is_file():
            path.unlink()
        temporary = Path(f"{path}.tmp")
        if temporary.is_file():
            temporary.unlink()
    if output_dir.is_dir() and not any(output_dir.iterdir()):
        output_dir.rmdir()


def _filter_config(data, viewer_config):
    nyquist = data.sample_rate / 2.0
    low = float(getattr(viewer_config, "bandpass_low", 20.0))
    high = min(float(getattr(viewer_config, "bandpass_high", 450.0)), nyquist - 1.0)
    return SimpleNamespace(
        sampling_rate=data.sample_rate,
        preprocess=SimpleNamespace(
            notch_hz=float(getattr(viewer_config, "notch_hz", 50.0)),
            notch_bandwidth_hz=float(
                getattr(viewer_config, "notch_bandwidth_hz", 8.0)
            ),
            notch_harmonics=int(getattr(viewer_config, "notch_harmonics", 3)),
            bandpass=(low, max(high, low + 1.0)),
        ),
        segmentation=SimpleNamespace(
            use_preprocess_chain=False,
            smooth_window_ms=float(
                getattr(viewer_config, "envelope_window_ms", 100.0)
            ),
        ),
    )


def _segments_for_export(
    trial_dir: Path, time_review: dict | None, n_samples: int,
) -> tuple[str, dict, list[dict]]:
    if time_review:
        actions = load_action_segments(trial_dir)
        if actions:
            return "action", time_review, actions
        return "time_window", time_review, [{
            "segment_index": 1,
            "start_sample": int(time_review["start_sample"]),
            "peak_sample": "",
            "end_sample_exclusive": int(time_review["end_sample_exclusive"]),
        }]
    full_trial = {
        "start_sample": 0,
        "end_sample_exclusive": int(n_samples),
    }
    return "full_trial", full_trial, [{
        "segment_index": 1,
        "start_sample": 0,
        "peak_sample": "",
        "end_sample_exclusive": int(n_samples),
    }]


def _frequency_metrics(signal: np.ndarray, sample_rate: float, selected: set[str]) -> dict:
    frequencies, power = compute_psd(signal, sample_rate, normalize=False)
    total = float(np.sum(power))
    values = {}
    if frequencies.size == 0 or not math.isfinite(total) or total <= 0:
        return {metric: math.nan for metric in selected if metric in {"mnf", "mdf"}}
    if "mnf" in selected:
        values["mnf"] = float(np.sum(frequencies * power) / total)
    if "mdf" in selected:
        cumulative = np.cumsum(power)
        index = min(int(np.searchsorted(cumulative, total / 2.0)), len(frequencies) - 1)
        values["mdf"] = float(frequencies[index])
    return values


def _diagnostic_psd(
    signal: np.ndarray, sample_rate: float, chunk_sec: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Average fixed-size Welch blocks so diagnostic FFT memory stays bounded."""
    signal = np.asarray(signal, dtype=float).ravel()
    chunk_size = max(16, int(round(sample_rate * chunk_sec)))
    if len(signal) <= chunk_size:
        return compute_psd(signal, sample_rate, normalize=False)

    starts = list(range(0, len(signal) - chunk_size + 1, chunk_size))
    final_start = len(signal) - chunk_size
    if starts[-1] != final_start:
        starts.append(final_start)
    spectra = [
        compute_psd(
            signal[start:start + chunk_size], sample_rate, normalize=False
        )
        for start in starts
    ]
    spectra = [(frequencies, power) for frequencies, power in spectra if len(power)]
    if not spectra:
        return np.array([]), np.array([])
    return spectra[0][0], np.mean([power for _, power in spectra], axis=0)


def _quality_values(
    raw_signals: list[np.ndarray], sample_rate: float, notch_hz: float,
    start: int, end: int, selected: set[str],
) -> dict[str, list[float]]:
    """Return selected raw-signal quality percentages by channel."""
    values = {metric: [] for metric in ("pli", "mar") if metric in selected}
    emg_high = min(450.0, sample_rate / 2.0)
    for signal in raw_signals:
        reviewed = np.asarray(signal[start:end], dtype=float)
        reviewed = reviewed - np.mean(reviewed)
        frequencies, power = _diagnostic_psd(reviewed, sample_rate)
        if "pli" in values:
            total = band_power(frequencies, power, 0.0, sample_rate / 2.0)
            line = band_power(
                frequencies, power, notch_hz - 1.5, notch_hz + 1.5
            )
            values["pli"].append(100.0 * line / (total + 1e-12))
        if "mar" in values:
            emg = band_power(frequencies, power, 20.0, emg_high)
            low = band_power(frequencies, power, 0.0, 20.0)
            values["mar"].append(100.0 * low / (emg + 1e-12))
    return values


def _write_feature_csv(path: Path, rows: list[dict], selected: set[str]) -> None:
    fields = [
        "trial", "scope", "segment_index", "channel", "start_sample",
        "peak_sample", "end_sample_exclusive", "start_sec", "peak_sec", "end_sec",
    ]
    if "mnf" in selected:
        fields.append("MNF_Hz")
    if "mdf" in selected:
        fields.append("MDF_Hz")
    temporary = Path(f"{path}.tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _save_figure(figure, path: Path) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    temporary = Path(f"{path}.tmp")
    try:
        FigureCanvasAgg(figure).print_png(str(temporary))
        temporary.replace(path)
    finally:
        figure.clear()


def _plot_signal(
    path: Path,
    data,
    signals: list[np.ndarray],
    time_review: dict,
    segments: list[dict],
    task_name: str,
    title: str,
) -> None:
    from matplotlib.figure import Figure

    start = int(time_review["start_sample"])
    end = int(time_review["end_sample_exclusive"])
    step = max(1, math.ceil((end - start) / 50000))
    positions = np.arange(start, end, step)
    time_axis = positions / data.sample_rate
    figure = Figure(figsize=(12, max(3.2, 2.0 * len(signals))), dpi=150)
    axes = figure.subplots(len(signals), 1, sharex=True, squeeze=False)[:, 0]
    colors = (
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    )
    for channel, axis in enumerate(axes):
        axis.plot(
            time_axis, signals[channel][start:end:step],
            color=colors[channel % len(colors)], linewidth=0.7,
        )
        for segment in segments:
            axis.axvspan(
                int(segment["start_sample"]) / data.sample_rate,
                int(segment["end_sample_exclusive"]) / data.sample_rate,
                color="#2e7d32", alpha=0.08,
            )
        axis.set_ylabel(f"CH{channel + 1} (uV)")
        axis.grid(True, alpha=0.2)
    axes[-1].set_xlabel("Time (s)")
    figure.suptitle(
        f"{task_display_name(task_name)} - {title}", **_task_title_kwargs()
    )
    figure.tight_layout()
    _save_figure(figure, path)


def _plot_spectrum_comparison(
    path: Path,
    data,
    bandpass_only: list[np.ndarray],
    filtered: list[np.ndarray],
    time_review: dict,
    task_name: str,
    notch_hz: float,
) -> None:
    from matplotlib.figure import Figure

    start = int(time_review["start_sample"])
    end = int(time_review["end_sample_exclusive"])
    figure = Figure(figsize=(12, max(3.2, 2.0 * len(filtered))), dpi=150)
    axes = figure.subplots(len(filtered), 1, sharex=True, squeeze=False)[:, 0]
    floor = np.finfo(float).tiny
    for channel, axis in enumerate(axes):
        f_band, p_band = _diagnostic_psd(
            bandpass_only[channel][start:end], data.sample_rate
        )
        f_notch, p_notch = _diagnostic_psd(
            filtered[channel][start:end], data.sample_rate
        )
        if len(f_band):
            axis.plot(
                f_band, 10.0 * np.log10(np.maximum(p_band, floor)),
                color="#6b7280", linewidth=1.0, label="Band-pass only",
            )
        if len(f_notch):
            axis.plot(
                f_notch, 10.0 * np.log10(np.maximum(p_notch, floor)),
                color="#d62728", linewidth=1.0, label="Band-pass + notch",
            )
        axis.axvline(notch_hz, color="#111827", linestyle="--", linewidth=0.8)
        axis.set_ylabel(f"CH{channel + 1}\n(dB/uV^2/Hz)")
        axis.grid(True, alpha=0.2)
    if len(axes):
        axes[0].legend(frameon=False, ncol=2)
    axes[-1].set_xlabel("Frequency (Hz)")
    axes[-1].set_xlim(0.0, min(500.0, data.sample_rate / 2.0))
    figure.suptitle(
        f"{task_display_name(task_name)} - 带通与陷波频谱对照",
        **_task_title_kwargs(),
    )
    figure.tight_layout()
    _save_figure(figure, path)


def _plot_quality_bars(
    path: Path,
    channel_names: tuple[str, ...],
    values: list[float],
    task_name: str,
    title: str,
    ylabel: str,
    color: str,
) -> None:
    from matplotlib.figure import Figure

    figure = Figure(figsize=(9, 5), dpi=150)
    axis = figure.subplots()
    display_channels = [f"CH{index + 1}" for index in range(len(channel_names))]
    bars = axis.bar(display_channels, values, color=color)
    axis.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
    axis.set_ylabel(ylabel)
    axis.set_title(
        f"{task_display_name(task_name)} - {title}", **_task_title_kwargs()
    )
    axis.grid(True, axis="y", alpha=0.25)
    figure.tight_layout()
    _save_figure(figure, path)


def _plot_envelope(
    path: Path,
    data,
    filtered: list[np.ndarray],
    envelopes: list[np.ndarray],
    time_review: dict,
    segments: list[dict],
    task_name: str,
) -> None:
    from matplotlib.figure import Figure

    start = int(time_review["start_sample"])
    end = int(time_review["end_sample_exclusive"])
    step = max(1, math.ceil((end - start) / 50000))
    sample_positions = np.arange(start, end, step)
    time_axis = sample_positions / data.sample_rate
    figure = Figure(figsize=(12, max(3.2, 2.2 * len(filtered))), dpi=150)
    axes = figure.subplots(len(filtered), 1, sharex=True, squeeze=False)[:, 0]
    colors = (
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
        "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    )
    for channel, axis in enumerate(axes):
        color = colors[channel % len(colors)]
        axis.plot(time_axis, filtered[channel][start:end:step], color=color, alpha=0.35, linewidth=0.6)
        axis.plot(time_axis, envelopes[channel][start:end:step], color=color, linewidth=1.4)
        for segment in segments:
            axis.axvspan(
                int(segment["start_sample"]) / data.sample_rate,
                int(segment["end_sample_exclusive"]) / data.sample_rate,
                color="#2e7d32",
                alpha=0.08,
            )
        axis.set_ylabel(f"CH{channel + 1} (uV)")
        axis.grid(True, alpha=0.2)
    axes[-1].set_xlabel("Time (s)")
    figure.suptitle(
        f"{task_display_name(task_name)} - 滤波 EMG 与 RMS 包络",
        **_task_title_kwargs(),
    )
    figure.tight_layout()
    _save_figure(figure, path)


def _plot_frequency_metric(
    path: Path, rows: list[dict], metric: str, task_name: str
) -> None:
    from matplotlib.figure import Figure

    column = "MNF_Hz" if metric == "mnf" else "MDF_Hz"
    title = "Mean Frequency (MNF)" if metric == "mnf" else "Median Frequency (MDF)"
    channels = list(dict.fromkeys(row["channel"] for row in rows))
    display_channels = {
        channel: f"CH{index + 1}" for index, channel in enumerate(channels)
    }
    segment_indices = sorted({int(row["segment_index"]) for row in rows})
    figure = Figure(figsize=(9, 5), dpi=150)
    axis = figure.subplots()
    if len(segment_indices) == 1:
        values = [float(next(row[column] for row in rows if row["channel"] == channel)) for channel in channels]
        axis.bar([display_channels[channel] for channel in channels], values, color="#2878b5")
    else:
        for channel in channels:
            channel_rows = {int(row["segment_index"]): row for row in rows if row["channel"] == channel}
            values = [float(channel_rows[index][column]) for index in segment_indices]
            axis.plot(
                segment_indices, values, marker="o", linewidth=1.5,
                label=display_channels[channel],
            )
        axis.legend(frameon=False)
        axis.set_xticks(segment_indices)
        axis.set_xlabel("Action segment")
    axis.set_ylabel("Frequency (Hz)")
    axis.set_title(
        f"{task_display_name(task_name)} - {title}",
        **_task_title_kwargs(),
    )
    axis.grid(True, axis="y", alpha=0.25)
    figure.tight_layout()
    _save_figure(figure, path)


def export_trial_features(
    trial_dir: str | Path,
    selected_metrics,
    viewer_config=None,
) -> ExportResult:
    """Export selected metrics for actions, a reviewed window, or a full Trial."""
    trial = Path(trial_dir).resolve()
    selected = {str(metric).lower() for metric in selected_metrics}
    if not selected:
        raise ValueError("请至少选择一个导出指标")
    unsupported = selected.difference(SUPPORTED_METRICS)
    if unsupported:
        raise ValueError(f"不支持的指标: {', '.join(sorted(unsupported))}")

    data = load_trial_txt(trial / "emg_data.txt")
    if data.n_samples <= 0:
        raise ValueError("Trial 没有可导出的样本")
    time_review = load_time_review(trial)
    task_name = _read_trial_task(trial)
    config = _filter_config(data, viewer_config)
    notch_hz = float(config.preprocess.notch_hz)
    scope, time_review, segments = _segments_for_export(
        trial, time_review, data.n_samples
    )
    valid_start = int(time_review["start_sample"])
    valid_end = int(time_review["end_sample_exclusive"])
    for segment in segments:
        start = int(segment["start_sample"])
        end = int(segment["end_sample_exclusive"])
        peak = segment.get("peak_sample", "")
        if not (valid_start <= start < end <= valid_end <= data.n_samples):
            raise ValueError("导出分段超出了已审核时间范围")
        if peak != "" and not (start <= int(peak) < end):
            raise ValueError("导出分段的峰值位置无效")

    raw_signals = [
        np.nan_to_num(data.emg[:, channel], nan=0.0, posinf=0.0, neginf=0.0)
        for channel in range(data.emg.shape[1])
    ]
    needs_filtered = bool(selected.intersection({
        "filtered", "envelope", "spectrum", "mnf", "mdf",
    }))
    filtered = [
        apply_filters(signal, config, do_notch=True)
        for signal in raw_signals
    ] if needs_filtered else []
    bandpass_only = [
        apply_filters(signal, config, do_notch=False)
        for signal in raw_signals
    ] if "spectrum" in selected else []
    envelopes = [
        compute_envelope(
            signal, config, config.segmentation.smooth_window_ms
        )
        for signal in filtered
    ] if "envelope" in selected else []
    quality_values = _quality_values(
        raw_signals,
        data.sample_rate,
        notch_hz,
        valid_start,
        valid_end,
        selected,
    ) if selected.intersection({"pli", "mar"}) else {}

    output_dir = results_dir(trial)
    output_dir.mkdir(exist_ok=True)
    rows = []
    scalar_selected = bool(selected.intersection({"mnf", "mdf"}))
    if scalar_selected:
        for segment in segments:
            start = int(segment["start_sample"])
            end = int(segment["end_sample_exclusive"])
            peak = segment.get("peak_sample", "")
            for channel, name in enumerate(data.channel_names):
                values = _frequency_metrics(filtered[channel][start:end], data.sample_rate, selected)
                row = {
                    "trial": trial.name,
                    "scope": scope,
                    "segment_index": int(segment.get("segment_index", 1)),
                    "channel": name,
                    "start_sample": start,
                    "peak_sample": peak,
                    "end_sample_exclusive": end,
                    "start_sec": f"{start / data.sample_rate:.6f}",
                    "peak_sec": f"{int(peak) / data.sample_rate:.6f}" if peak != "" else "",
                    "end_sec": f"{end / data.sample_rate:.6f}",
                }
                if "mnf" in selected:
                    row["MNF_Hz"] = f"{values['mnf']:.6f}"
                if "mdf" in selected:
                    row["MDF_Hz"] = f"{values['mdf']:.6f}"
                rows.append(row)

    desired_files = set()
    csv_path = None
    if scalar_selected:
        csv_path = output_dir / FEATURES_FILENAME
        _write_feature_csv(csv_path, rows, selected)
        desired_files.add(FEATURES_FILENAME)

    image_paths = {}
    if "raw" in selected:
        image_paths["raw"] = output_dir / _OUTPUT_FILENAMES["raw"]
        _plot_signal(
            image_paths["raw"], data, raw_signals, time_review, segments,
            task_name, "原始 EMG 信号",
        )
        desired_files.add(_OUTPUT_FILENAMES["raw"])
    if "filtered" in selected:
        image_paths["filtered"] = output_dir / _OUTPUT_FILENAMES["filtered"]
        _plot_signal(
            image_paths["filtered"], data, filtered, time_review, segments,
            task_name, "带通与陷波后 EMG 信号",
        )
        desired_files.add(_OUTPUT_FILENAMES["filtered"])
    if "envelope" in selected:
        image_paths["envelope"] = output_dir / _OUTPUT_FILENAMES["envelope"]
        _plot_envelope(
            image_paths["envelope"], data, filtered, envelopes,
            time_review, segments, task_name,
        )
        desired_files.add(_OUTPUT_FILENAMES["envelope"])
    if "spectrum" in selected:
        image_paths["spectrum"] = output_dir / _OUTPUT_FILENAMES["spectrum"]
        _plot_spectrum_comparison(
            image_paths["spectrum"], data, bandpass_only, filtered,
            time_review, task_name, notch_hz,
        )
        desired_files.add(_OUTPUT_FILENAMES["spectrum"])
    if "pli" in selected:
        image_paths["pli"] = output_dir / _OUTPUT_FILENAMES["pli"]
        _plot_quality_bars(
            image_paths["pli"], data.channel_names, quality_values["pli"],
            task_name, "工频干扰",
            f"{notch_hz:g} Hz power / total power (%)", "#d95f02",
        )
        desired_files.add(_OUTPUT_FILENAMES["pli"])
    if "mar" in selected:
        image_paths["mar"] = output_dir / _OUTPUT_FILENAMES["mar"]
        _plot_quality_bars(
            image_paths["mar"], data.channel_names, quality_values["mar"],
            task_name, "运动伪迹", "0-20 Hz / 20-450 Hz power (%)", "#7570b3",
        )
        desired_files.add(_OUTPUT_FILENAMES["mar"])
    for metric in ("mnf", "mdf"):
        if metric in selected:
            image_paths[metric] = output_dir / _OUTPUT_FILENAMES[metric]
            _plot_frequency_metric(image_paths[metric], rows, metric, task_name)
            desired_files.add(_OUTPUT_FILENAMES[metric])

    for filename in _OUTPUT_FILENAMES.values():
        stale = output_dir / filename
        if filename not in desired_files and stale.is_file():
            stale.unlink()
    return ExportResult(output_dir, scope, len(rows), csv_path, image_paths)
