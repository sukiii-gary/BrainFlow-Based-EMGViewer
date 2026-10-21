"""Offscreen visual smoke test with the synthetic board; no hardware access."""
import json
import os
from pathlib import Path
import sys
import tempfile

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
root = Path(__file__).resolve().parent
sys.path[:0] = [str(root / 'app'), str(root / 'emg_python')]

from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QFont, QFontDatabase
from PyQt6.QtWidgets import QApplication
from board_manager import CytonManager
from config_viewer import ViewerConfig
from metrics import PacketLossTracker
from recorder import Recorder
from signal_chain import SignalChain
from ui.main_window import MainWindow

app = QApplication([])
# The offscreen Qt platform does not discover Windows fallback fonts itself.
QFontDatabase.addApplicationFont('C:/Windows/Fonts/msyh.ttc')
app.setFont(QFont('Microsoft YaHei', 9))
temporary = tempfile.TemporaryDirectory()
config = ViewerConfig(board_id=-1, record_subject='UDP_GUI_TEST', record_dir=temporary.name)
board = CytonManager(board_id=-1)
chain = SignalChain(fs=1000, n_channels=8, bandpass=(20, 450))
window = MainWindow(
    board_manager=board, signal_chain=chain,
    metrics_engine=PacketLossTracker(1000),
    recorder=Recorder(data_dir=temporary.name, n_channels=8), config=config,
)
window.show()
window._on_connect()
window._connect_thread.finished.connect(window._on_start)


def finish():
    report = {
        'title': window.windowTitle(),
        'connected': board.is_connected,
        'streaming': board.is_streaming,
        'data_verified': board.data_stream_verified,
        'channels': board.n_channels,
        'status': window._status_label.text(),
    }
    window.grab().save(str(root / 'gui_smoke.png'))
    (root / 'gui_smoke.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    window._on_stop()
    window.close()
    board.stop()
    app.exit(0 if report['data_verified'] and report['channels'] == 8 else 1)


QTimer.singleShot(3500, finish)
result = app.exec()
temporary.cleanup()
sys.exit(result)
