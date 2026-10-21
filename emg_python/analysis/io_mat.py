"""读取 MATLAB .mat EMG(肱二头数据)—— 复刻 MATLAB load_task 的 .mat 分支。

支持 v7.3(HDF5,用 h5py)与旧版 v7(scipy.io.loadmat)。
返回 pandas.DataFrame:Time 列 + 各通道列;df.attrs 存 emg_abs_start / fs。

要点(与 MATLAB 对齐):
- data_source 字段(默认 raw_eeg_uv)取原始 EMG;h5py 读回为 (N, nchan),列号=通道 index。
- 时间轴优先 t_seconds / sample_time_s / timestamps(长度==N 才用,并归零),否则 (0:N-1)/fs。
- emg_abs_start 优先 receive_abs_posix_s → sample_abs_posix_s → recording_start_posix_s 的首值。
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd


def _first_val(a) -> float:
    return float(np.asarray(a).ravel()[0])


def load_mat_task(path, cfg, warns: list | None = None) -> pd.DataFrame:
    path = Path(path)
    if warns is None:
        warns = []
    try:
        import h5py
        with h5py.File(path, "r") as f:
            keys = set(f.keys())
            src = cfg.data_source if cfg.data_source in keys else (
                "raw_eeg_uv" if "raw_eeg_uv" in keys else "eeg_uv")
            data = np.array(f[src])                      # (N, nchan) via h5py
            if data.ndim != 2:
                raise ValueError(f"{src} 不是二维: shape={data.shape}")
            # h5py 已把 MATLAB (nchan×N) 读成 (N, nchan);若相反则转置
            if data.shape[0] < data.shape[1]:
                data = data.T
            n = data.shape[0]
            fs = _first_val(f["fs"]) if "fs" in keys else float(cfg.sampling_rate)
            t = _time_vector(f, keys, n, fs)
            abs_start = _abs_start(f, keys)
            return _build_df(data, t, cfg, fs, abs_start, warns)
    except (ImportError, OSError):
        # 回退旧版 v7 .mat
        from scipy.io import loadmat
        m = loadmat(path)
        keys = set(m.keys())
        src = cfg.data_source if cfg.data_source in keys else "raw_eeg_uv"
        data = np.asarray(m[src], dtype=float)
        if data.shape[0] < data.shape[1]:
            data = data.T                                # → (N, nchan)
        n = data.shape[0]
        fs = _first_val(m["fs"]) if "fs" in keys else float(cfg.sampling_rate)
        t = None
        for k in ("t_seconds", "sample_time_s", "timestamps"):
            if k in keys and np.asarray(m[k]).size == n:
                tv = np.asarray(m[k], dtype=float).ravel()
                t = tv - tv[0]
                break
        if t is None:
            t = np.arange(n) / fs
        abs_start = np.nan
        for k in ("receive_abs_posix_s", "sample_abs_posix_s", "recording_start_posix_s"):
            if k in keys:
                abs_start = _first_val(m[k]); break
        return _build_df(data, t, cfg, fs, abs_start, warns)


def _time_vector(f, keys, n, fs):
    for k in ("t_seconds", "sample_time_s", "timestamps"):
        if k in keys:
            tv = np.array(f[k], dtype=float).ravel()
            if tv.size == n:
                return tv - tv[0]
    return np.arange(n) / fs


def _abs_start(f, keys):
    for k in ("receive_abs_posix_s", "sample_abs_posix_s", "recording_start_posix_s"):
        if k in keys:
            return _first_val(f[k])
    return np.nan


def _build_df(data, t, cfg, fs, abs_start, warns) -> pd.DataFrame:
    n = data.shape[0]
    cols = {"Time": np.asarray(t, dtype=float).ravel()[:n]}
    for c in cfg.channel_map:
        if c.index >= data.shape[1]:
            raise IndexError(f"通道 index={c.index} 超出数据列数 {data.shape[1]}")
        x = data[:, c.index].astype(float)
        cols[c.name] = x
        # 质控告警(对应 MATLAB load_task)
        if np.nanstd(x) < np.finfo(float).eps:
            warns.append(f"[std≈0 疑似掉线] {c.name}")
        nnan = int(np.isnan(x).sum())
        if nnan:
            warns.append(f"[含 {nnan} 个 NaN] {c.name}")
    df = pd.DataFrame(cols)
    df.attrs["emg_abs_start"] = abs_start
    df.attrs["fs"] = fs
    return df
