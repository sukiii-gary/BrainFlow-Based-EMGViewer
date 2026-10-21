import sys
import unittest
from pathlib import Path
from unittest.mock import patch


class PackagedConfigTests(unittest.TestCase):
    def test_frozen_app_records_next_to_executable(self):
        executable = Path("C:/portable/EMG_Viewer_Sync_Impedance.exe")
        with patch.object(sys, "frozen", True, create=True), patch.object(
            sys, "executable", str(executable)
        ):
            namespace = {"__file__": "C:/temp/_MEI123/config_viewer.py"}
            source = Path(__file__).resolve().parents[1] / "config_viewer.py"
            exec(compile(source.read_text(encoding="utf-8"), str(source), "exec"), namespace)

        config = namespace["ViewerConfig"]()
        self.assertEqual(Path(config.record_dir), executable.parent / "recordings")


if __name__ == "__main__":
    unittest.main()
