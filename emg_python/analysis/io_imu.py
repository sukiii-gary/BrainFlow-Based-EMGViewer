"""读取 Trigno Discover IMU .csv —— 复刻 MATLAB load_imu。

- 前 8 行找含 'Date/Time' 的行 → 起始 datetime(格式 yyyy/M/d HH:mm:ss,时区 +08:00);
- 数据跳过 8 行头;固定列位:
  1=ACCx_t 2=ACCx 3=ACCy_t 4=ACCy 5=ACCz_t 6=ACCz
  7=GYROx_t 8=GYROx 9=GYROy_t 10=GYROy 11=GYROz_t 12=GYROz(1 基;Python 0 基 -1);
- fs = 1/median(diff(t));ok = len(t)>10。
返回 dict。本次 TA use_imu=False,IMU 不参与切分,仅用于读取展示。
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd


def load_imu(csv_path) -> dict:
    csv_path = Path(csv_path)
    imu = {"ok": False, "t": None, "ax": None, "ay": None, "az": None,
           "gx": None, "gy": None, "gz": None, "fs": np.nan,
           "abs_start_posix": np.nan}
    if not csv_path.exists():
        return imu
    try:
        head = csv_path.read_text(encoding="utf-8", errors="replace").splitlines()[:8]
        abs_start = np.nan
        for ln in head:
            if "Date/Time" in ln:
                s = ln.split(",")[1].strip()
                dt = datetime.strptime(s, "%Y/%m/%d %H:%M:%S")
                dt = dt.replace(tzinfo=timezone(timedelta(hours=8)))
                abs_start = dt.timestamp()
                break

        # 只取前 12 列(6 轴×[时间,值]);尾部为 Trigno 统计块(含字符串/空),忽略
        m = pd.read_csv(csv_path, skiprows=8, header=None,
                        usecols=range(12)).to_numpy(dtype=float)
        t = m[:, 0]
        good = np.isfinite(t)
        col = lambda j: m[good, j]
        imu.update(t=col(0), ax=col(1), ay=col(3), az=col(5),
                   gx=col(7), gy=col(9), gz=col(11))
        tt = imu["t"]
        if tt.size > 1:
            imu["fs"] = 1.0 / np.median(np.diff(tt))
        imu["abs_start_posix"] = abs_start
        imu["ok"] = tt.size > 10
    except Exception:
        imu["ok"] = False
    return imu
