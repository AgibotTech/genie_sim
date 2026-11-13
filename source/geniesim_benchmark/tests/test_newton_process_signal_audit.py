# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""Process disappearance races and actual signal escalation remain distinguishable."""

import signal
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from geniesim_benchmark.newton import layout_batch


class ProcessGroupSignalAuditTests(unittest.TestCase):
    def stop(self, alive, *, timeout=0, kill_effect=None, start=1000):
        with (
            patch.object(layout_batch, "_process_row", return_value=("S", 1, 100, start)),
            patch.object(layout_batch, "_process_rows", return_value={100: ("S", 1, 100, start)}),
            patch.object(layout_batch, "_group_alive", side_effect=alive),
            patch.object(layout_batch.Path, "read_text", return_value="newton-node\n"),
            patch.object(layout_batch.os, "killpg", side_effect=kill_effect) as kill,
        ):
            result = layout_batch._stop_group(100, timeout, lambda: None, expected_start=1000)
        return result, kill

    def test_process_disappearing_before_term_is_not_forced(self):
        result, kill = self.stop([True, True, False], kill_effect=[None, ProcessLookupError()])
        self.assertFalse(result["forced"])
        self.assertEqual([entry["signal"] for entry in result["signals"]], ["SIGINT", "SIGTERM"])
        self.assertEqual([entry["sent"] for entry in result["signals"]], [True, False])
        self.assertEqual(kill.call_count, 2)

    def test_sigint_exit_keeps_member_audit(self):
        result, kill = self.stop([True, False], timeout=0.2)
        self.assertFalse(result["forced"])
        self.assertEqual(result["pgid"], 100)
        self.assertEqual(
            result["signals"],
            [
                {
                    "signal": "SIGINT",
                    "sent": True,
                    "members": [{"pid": 100, "state": "S", "comm": "newton-node", "start_ticks": 1000}],
                }
            ],
        )
        kill.assert_called_once_with(100, signal.SIGINT)

    def test_actual_term_escalation_is_forced(self):
        result, _ = self.stop([True, True, False])
        self.assertTrue(result["forced"])
        self.assertEqual([entry["signal"] for entry in result["signals"]], ["SIGINT", "SIGTERM"])
        self.assertTrue(all(entry["sent"] for entry in result["signals"]))

    def test_group_disappearing_before_kill_returns_actual_signal_audit(self):
        with patch.object(layout_batch.time, "monotonic", side_effect=[0, 1, 1, 7]):
            result, kill = self.stop([True, True, True], kill_effect=[None, None, ProcessLookupError()])
        self.assertTrue(result["forced"])
        self.assertEqual([entry["signal"] for entry in result["signals"]], ["SIGINT", "SIGTERM", "SIGKILL"])
        self.assertEqual([entry["sent"] for entry in result["signals"]], [True, True, False])
        self.assertEqual(kill.call_count, 3)

    def test_reused_group_leader_is_not_signalled(self):
        result, kill = self.stop([], start=9999)
        self.assertFalse(result["forced"])
        self.assertTrue(result["pid_reused"])
        self.assertEqual(result["signals"], [])
        kill.assert_not_called()

    def test_already_dead_group_has_empty_signal_audit(self):
        result, kill = self.stop([False])
        self.assertFalse(result["forced"])
        self.assertEqual(result["signals"], [])
        kill.assert_not_called()

    def test_stop_process_group_records_waited_returncode(self):
        process = SimpleNamespace(pid=100, poll=Mock(), wait=Mock(return_value=-15), returncode=-15)
        with (
            patch.object(layout_batch, "_process_row", return_value=("S", 1, 100, 1000)),
            patch.object(layout_batch, "_stop_group", return_value={"forced": True, "signals": []}),
        ):
            result = layout_batch.stop_process_group(process)
        self.assertEqual(result["returncode"], -15)
        process.wait.assert_called_once_with()
