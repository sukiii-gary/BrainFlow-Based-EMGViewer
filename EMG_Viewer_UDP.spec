from pathlib import Path
import sys

from PyInstaller.utils.hooks import collect_all


project_dir = Path(SPEC).resolve().parent / "app"
emg_python_dir = project_dir.parent / "emg_python"

brainflow_datas, brainflow_binaries, brainflow_hiddenimports = collect_all("brainflow")
cv2_datas, cv2_binaries, cv2_hiddenimports = collect_all("cv2")

# This conda environment provides PyQt6 bindings in site-packages while the
# matching Qt runtime lives in Library/bin, which PyInstaller cannot infer.
qt_bin_dir = Path(sys.prefix) / "Library" / "bin"
qt_dependency_names = {
    "brotlicommon.dll",
    "brotlidec.dll",
    "double-conversion.dll",
    "freetype.dll",
    "harfbuzz.dll",
    "icudt78.dll",
    "icuin78.dll",
    "icuuc78.dll",
    "libpng16.dll",
    "libwebp.dll",
    "libwebpdemux.dll",
    "libwebpmux.dll",
    "jpeg8.dll",
    "msvcp140.dll",
    "msvcp140_1.dll",
    "msvcp140_2.dll",
    "pcre2-16.dll",
    "openjp2.dll",
    "qhull_r.dll",
    "raqm-0.dll",
    "tiff.dll",
    "vcruntime140.dll",
    "vcruntime140_1.dll",
    "zlib.dll",
    "zlib-ng2.dll",
    "zstd.dll",
}
qt_runtime_binaries = [
    (str(path), "PyQt6/Qt6/bin")
    for path in qt_bin_dir.glob("*.dll")
    if path.name.startswith("Qt6") or path.name.lower() in qt_dependency_names
]

a = Analysis(
    [str(project_dir / "main.py")],
    pathex=[str(project_dir), str(emg_python_dir)],
    binaries=brainflow_binaries + cv2_binaries + qt_runtime_binaries,
    datas=brainflow_datas + cv2_datas,
    hiddenimports=brainflow_hiddenimports + cv2_hiddenimports + [
        "serial",
        "serial.tools.list_ports",
        "serial.tools.list_ports_windows",
        "matplotlib.backends.backend_agg",
        "preprocessing.filters",
        "preprocessing.envelope",
        "preprocessing.psd",
        "features.signal_quality",
        "segmentation.reps",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "PySide6",
        "PySide2",
        "PyQt5",
        "shiboken6",
        "pyarrow",
        "numba",
        "IPython",
        "openpyxl",
        "zmq",
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="EMG_Viewer_UDP_Sync_Camera_Impedance",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
