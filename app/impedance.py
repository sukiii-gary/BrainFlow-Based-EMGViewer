"""OpenBCI Cyton lead-off impedance calculations."""
from __future__ import annotations

import math

import numpy as np


LEAD_OFF_CURRENT_AMPS = 6.0e-9
SERIES_RESISTOR_OHMS = 2200.0
EMG_CONTACT_TARGET_OHMS = 10_000.0
CYTON_GUI_GREEN_OHMS = 750_000.0
CYTON_GUI_YELLOW_OHMS = 2_500_000.0
LIKELY_OPEN_CIRCUIT_OHMS = 5_000_000.0


def classify_impedance_ohms(impedance_ohms: float) -> dict:
    """Return separate Cyton GUI and surface-EMG interpretations.

    The Cyton GUI colors are retained only as an official device reference.
    They are deliberately separate from the stricter surface-EMG contact goal.
    """
    try:
        impedance = float(impedance_ohms)
    except (TypeError, ValueError):
        impedance = float("nan")
    if not math.isfinite(impedance) or impedance <= 0:
        return {
            "emg_contact": "invalid",
            "emg_contact_label": "无有效读数",
            "emg_contact_color": "#757575",
            "cyton_gui_reference": "invalid",
            "cyton_gui_label": "无有效读数",
            "cyton_gui_color": "#757575",
            "likely_open_circuit": False,
        }

    if impedance < EMG_CONTACT_TARGET_OHMS:
        emg_contact = "target"
        emg_label = "目标 <10 kOhm"
        emg_color = "#2e7d32"
    elif impedance <= 50_000.0:
        emg_contact = "improve"
        emg_label = "可用，建议优化"
        emg_color = "#827717"
    elif impedance <= 100_000.0:
        emg_contact = "high"
        emg_label = "偏高，建议处理"
        emg_color = "#ef6c00"
    else:
        emg_contact = "poor"
        emg_label = "高阻，建议处理"
        emg_color = "#c62828"

    # Use deterministic inclusive boundaries rather than reproducing the
    # OpenBCI GUI's historical strict-comparison boundary color carry-over.
    if impedance < CYTON_GUI_GREEN_OHMS:
        cyton_reference = "green"
        cyton_label = "官方 GUI：绿 (<750 kOhm)"
        cyton_color = "#2e7d32"
    elif impedance < CYTON_GUI_YELLOW_OHMS:
        cyton_reference = "yellow"
        cyton_label = "官方 GUI：黄 (750-<2500 kOhm)"
        cyton_color = "#f9a825"
    else:
        cyton_reference = "red"
        cyton_label = "官方 GUI：红 (>=2500 kOhm)"
        cyton_color = "#c62828"

    return {
        "emg_contact": emg_contact,
        "emg_contact_label": emg_label,
        "emg_contact_color": emg_color,
        "cyton_gui_reference": cyton_reference,
        "cyton_gui_label": cyton_label,
        "cyton_gui_color": cyton_color,
        "likely_open_circuit": impedance >= LIKELY_OPEN_CIRCUIT_OHMS,
    }


def estimate_impedance_ohms(samples_uv) -> dict:
    """Estimate electrode impedance using the formula used by OpenBCI GUI."""
    samples = np.asarray(samples_uv, dtype=np.float64).ravel()
    samples = samples[np.isfinite(samples)]
    if samples.size < 16:
        raise ValueError("有效阻抗测试样本不足")

    std_uv = float(np.std(samples))
    impedance = (
        math.sqrt(2.0) * std_uv * 1.0e-6 / LEAD_OFF_CURRENT_AMPS
        - SERIES_RESISTOR_OHMS
    )
    result = {
        "impedance_ohms": max(0.0, impedance),
        "std_uv": std_uv,
        "sample_count": int(samples.size),
    }
    return {**result, **classify_impedance_ohms(result["impedance_ohms"])}
