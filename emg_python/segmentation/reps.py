"""节奏性收缩单 rep 检测 —— 供 3.2(节奏背屈 timing/onset)与 3.5(fatigue_5c 疲劳斜率)复用。

在参考通道包络上找收缩峰(find_peaks),每 rep 由峰间局部极小界定 onset/offset;
逐通道各自检测 onset(供电极 onset 差)。两级门槛:baseline med+k·MAD 选候选峰,
再以候选峰高中位(该 trial 收缩幅度稳健估计)剔伪迹(过高)与噪声 blip(过低)。
只做事件检测,不算指标。
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks

from analysis import load_task
from preprocessing import compute_envelope


def baseline_env_stats(cfg, channels=None):
    """各通道 baseline_sitting 包络在 baseline_window_s 窗内的 (median, MAD)。"""
    fs = int(cfg.sampling_rate)
    channels = channels or cfg.channel_names
    dfb = load_task(Path(cfg.emg_dir) / cfg.segmentation.baseline_file, cfg)
    bw = cfg.segmentation.baseline_window_s
    a, b = int(bw[0] * fs), int(bw[1] * fs)
    stats = {}
    for ch in channels:
        env = compute_envelope(dfb[ch].to_numpy(float), cfg)[a:b]
        med = float(np.median(env))
        mad = float(np.median(np.abs(env - med))) or float(np.std(env))
        stats[ch] = (med, mad)
    return stats


def detect_reps(env_ref, fs, base_med, base_mad, k_high=3.0, k_low=1.5,
                min_rep_dist_s=0.6, min_dur_s=0.15, artifact_k=5.0, active_frac=0.3):
    """在参考通道包络上检测收缩 rep,rep 窗 = 相邻峰之间的局部极小(谷-峰-谷)。

    两级门槛:
      1) 候选峰:height = base_med + k_high·mad(仅把明显高于静息的局部峰选出);
      2) 保留:峰高须落在 [active_frac, artifact_k] × 候选峰高中位 之间。
         中位 = 该 trial 真实收缩幅度的稳健估计,据此
           * 剔运动/启动伪迹(> artifact_k×中位,如起点 ~4kµV 尖峰);
           * 剔噪声 blip(< active_frac×中位,贴近静息的小波动)。
    窗边界用峰间局部极小(不依赖绝对基线,对站姿抬高的静息电平稳健);伪迹 rep 连窗一并
    丢弃,故其邻近 rep 窗不含伪迹。首/末 rep 边界外延半个中位峰间隔。
    返回 [(onset, peak, offset)] 样本索引。
    """
    env_ref = np.asarray(env_ref, float)
    high = base_med + k_high * base_mad
    dist = max(1, int(min_rep_dist_s * fs))
    pk, _ = find_peaks(env_ref, height=high, distance=dist)
    n = len(env_ref)
    if pk.size == 0:
        return []
    ref_h = float(np.median(env_ref[pk]))          # 该 trial 收缩幅度稳健估计
    lo_h, hi_h = active_frac * ref_h, artifact_k * ref_h
    bounds = [pk[i] + int(np.argmin(env_ref[pk[i]:pk[i + 1]])) for i in range(len(pk) - 1)]
    half = int(np.median(np.diff(pk)) / 2) if pk.size >= 2 else int(0.5 * fs)
    starts = [max(0, pk[0] - half)] + bounds
    ends = bounds + [min(n - 1, pk[-1] + half)]
    reps = []
    for a, p, b in zip(starts, pk, ends):
        if (b - a) >= min_dur_s * fs and lo_h <= env_ref[p] <= hi_h:
            reps.append((int(a), int(p), int(b)))
    return reps


def channel_onset(env_ch, a, b, peak_frac=0.2):
    """rep 内相对 onset:在样本窗 [a,b) 内,该通道包络首次上穿"谷+20%(峰-谷)"的索引。

    用 rep 自身动态范围(谷→峰)定阈值,对基线漂移/站姿抬高稳健。无则 None。
    """
    seg = np.asarray(env_ch, float)[a:b]
    if seg.size < 3:
        return None
    lo, hi = float(seg.min()), float(seg.max())
    if hi <= lo:
        return None
    thr = lo + peak_frac * (hi - lo)
    idx = np.where(seg >= thr)[0]
    return int(a + idx[0]) if idx.size else None
