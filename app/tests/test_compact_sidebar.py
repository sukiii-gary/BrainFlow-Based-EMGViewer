import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from ui.compact_sidebar import CompactSidebar


class CompactSidebarHardwareControlsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.sidebar = CompactSidebar(n_channels=4)

    def test_master_checkboxes_update_all_channels(self):
        self.sidebar._master_bias.click()
        self.sidebar._master_srb2.click()

        settings = self.sidebar.get_hw_settings()["channels"]
        self.assertTrue(all(not item["bias_on"] for item in settings))
        self.assertTrue(all(not item["srb2_on"] for item in settings))
        self.assertEqual(self.sidebar._master_bias.checkState(), Qt.CheckState.Unchecked)
        self.assertEqual(self.sidebar._master_srb2.checkState(), Qt.CheckState.Unchecked)

    def test_individual_channel_changes_update_master_state(self):
        self.sidebar._bias_checkboxes[0].setChecked(False)
        self.assertEqual(
            self.sidebar._master_bias.checkState(), Qt.CheckState.PartiallyChecked
        )

        for checkbox in self.sidebar._bias_checkboxes[1:]:
            checkbox.setChecked(False)
        self.assertEqual(self.sidebar._master_bias.checkState(), Qt.CheckState.Unchecked)

        self.sidebar._bias_checkboxes[0].setChecked(True)
        self.assertEqual(
            self.sidebar._master_bias.checkState(), Qt.CheckState.PartiallyChecked
        )

    def test_clicking_partial_master_selects_all_channels(self):
        self.sidebar._srb2_checkboxes[0].setChecked(False)
        self.assertEqual(
            self.sidebar._master_srb2.checkState(), Qt.CheckState.PartiallyChecked
        )

        self.sidebar._master_srb2.click()

        self.assertTrue(all(cb.isChecked() for cb in self.sidebar._srb2_checkboxes))
        self.assertEqual(self.sidebar._master_srb2.checkState(), Qt.CheckState.Checked)

    def test_default_sidebar_exposes_all_eight_cyton_channels(self):
        sidebar = CompactSidebar()

        self.assertEqual(sidebar.n_channels, 8)
        self.assertEqual(len(sidebar._ch_checkboxes), 8)
        self.assertEqual(len(sidebar.get_hw_settings()["channels"]), 8)


if __name__ == "__main__":
    unittest.main()
