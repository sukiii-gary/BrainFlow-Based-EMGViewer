"""统一加载 + 任务发现 + 类别判定 —— 复刻 MATLAB load_task 分派 / discover_tasks / task_category。"""
from __future__ import annotations
from pathlib import Path
import re
import pandas as pd

from .io_mat import load_mat_task
from .io_openbci import load_openbci_task
from .io_imu import load_imu


def task_category(type_name: str) -> str:
    """按文件名前缀判类别(仅 4 类)。对应 MATLAB task_category。"""
    t = type_name.lower()
    if t.startswith("baseline"):
        return "baseline"
    if t.startswith("mvic"):
        return "mvic"
    if t.startswith("fatigue"):
        return "fatigue_bout"
    return "cyclic"     # static_dorsiflexion / walking / stepping / heel / ...


def parse_bpm(type_name: str):
    m = re.search(r"(\d+)\s*bpm", type_name, re.IGNORECASE)
    return int(m.group(1)) if m else None


def discover_tasks(cfg) -> list[dict]:
    """扫描 emg_dir 下所有 cfg.emg_ext 文件,每个=1任务。对应 MATLAB discover_tasks。"""
    emg_dir = Path(cfg.emg_dir)
    files = sorted(emg_dir.glob(f"*{cfg.emg_ext}"))
    cat_order = {"baseline": 0, "mvic": 1, "cyclic": 2, "fatigue_bout": 3}
    tasks = []
    for p in files:
        stem = p.stem
        cat = task_category(stem)
        bpm = parse_bpm(stem)
        exp_reps = None
        if cat == "mvic":
            mm = re.search(r"(\d+)_?rep", stem, re.IGNORECASE)
            exp_reps = int(mm.group(1)) if mm else 3
        tasks.append({
            "file": p.name, "path": p, "type": stem, "category": cat,
            "intensity": bpm, "expected_reps": exp_reps,
            "fatigue": cat == "fatigue_bout",
        })
    tasks.sort(key=lambda d: (cat_order.get(d["category"], 9),
                              d["intensity"] if d["intensity"] else 0,
                              d["file"]))
    return tasks


def load_task(path, cfg, warns: list | None = None) -> pd.DataFrame:
    """按扩展名分派读取。"""
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".txt":
        return load_openbci_task(path, cfg, warns)
    return load_mat_task(path, cfg, warns)


def load_task_imu(task: dict, cfg):
    """按同名文件读取 IMU(存在才读)。返回 imu dict 或 None。"""
    if cfg.imu_dir is None:
        return None
    stem = Path(task["file"]).stem
    csv = Path(cfg.imu_dir) / f"{stem}.csv"
    if not csv.exists():
        return None
    return load_imu(csv)
