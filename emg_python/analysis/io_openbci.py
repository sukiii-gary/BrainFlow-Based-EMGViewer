"""读取 OpenBCI GUI Raw EXG .txt(TA 数据)—— 复刻 MATLAB load_openbci_txt。

步骤:
  1. 逐行找第一个「strip 后非 % 开头且含 'Sample Index'」的表头行 hdrIdx;
  2. 列名去空格;通道列名 = 'EXG Channel <index>';
  3. Timestamp 列(绝对 POSIX 秒;非 'Timestamp (Formatted)')首值 = emg_abs_start;
  4. 丢弃初始化伪行:所选 EXG 列全为 0 或任一 NaN 的行;
  5. 时间轴 = (0:N-1)/fs(标称 fs,不用文件时间戳做轴)。
返回 pandas.DataFrame:Time + 各通道列;df.attrs['emg_abs_start']。
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd


def load_openbci_task(path, cfg, warns: list | None = None) -> pd.DataFrame:
    path = Path(path)
    if warns is None:
        warns = []
    fs = float(cfg.sampling_rate)

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    hdr_idx = None
    for i, ln in enumerate(lines):
        s = ln.strip()
        if not s.startswith("%") and "Sample Index" in ln:
            hdr_idx = i
            break
    if hdr_idx is None:
        raise ValueError(f"未找到含 'Sample Index' 的表头行: {path.name}")

    cols = [c.strip() for c in lines[hdr_idx].split(",")]
    # 数据从 hdr_idx+1 起
    df_raw = pd.read_csv(path, skiprows=hdr_idx, header=0)
    df_raw.columns = [c.strip() for c in df_raw.columns]

    # 通道列
    chan_cols = {}
    for c in cfg.channel_map:
        col = f"EXG Channel {c.index}"
        if col not in df_raw.columns:
            raise KeyError(f"缺列 '{col}'(可用: {list(df_raw.columns)[:8]}...)")
        chan_cols[c.name] = df_raw[col].to_numpy(dtype=float)

    # Timestamp(绝对 POSIX)首值作对齐锚
    abs_start = np.nan
    if "Timestamp" in df_raw.columns:
        ts = df_raw["Timestamp"].to_numpy(dtype=float)
    else:
        ts = None

    # 丢弃全 0 / NaN 伪行(设备初始化)
    exg = np.column_stack(list(chan_cols.values()))
    bad = np.all(exg == 0, axis=1) | np.any(np.isnan(exg), axis=1)
    good = ~bad
    if not good.any():
        raise ValueError(f"清洗后无有效样本: {path.name}")
    if ts is not None:
        abs_start = float(ts[good][0])

    n = int(good.sum())
    out = {"Time": np.arange(n) / fs}
    for name, arr in chan_cols.items():
        x = arr[good]
        out[name] = x
        if np.nanstd(x) < np.finfo(float).eps:
            warns.append(f"[std≈0 疑似掉线] {name}")
        nnan = int(np.isnan(x).sum())
        if nnan:
            warns.append(f"[含 {nnan} 个 NaN] {name}")

    df = pd.DataFrame(out)
    df.attrs["emg_abs_start"] = abs_start
    df.attrs["fs"] = fs
    return df
