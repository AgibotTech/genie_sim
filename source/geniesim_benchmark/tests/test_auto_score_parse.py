# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

"""The VLM grader reads the SCORE field as a share of the scoring points, not as its first number."""

import contextlib
import importlib
import io
import types
import unittest
from unittest import mock

import numpy as np

# generators/__init__.py re-exports the auto_score function under the module's name, so import the
# submodule by its dotted name.
auto_score_module = importlib.import_module("geniesim_benchmark.evaluator.generators.auto_score")


def vlm_reply(satisfied: int, score_line: str, counts: bool = True) -> str:
    """A reply in the format PROMPT_AUTOSCORE.txt asks for, with three scoring points."""
    tags = ["[TRUE]" if index < satisfied else "[FALSE]" for index in range(3)]
    lines = ["SCORING_POINTS:", "1. A", "2. B", "3. C", "", "REASONING:"]
    lines += [f"Scoring point {index + 1}: seen in the head view. {tag}" for index, tag in enumerate(tags)]
    lines.append("")
    if counts:
        lines += [f"SATISFIED_COUNT: {satisfied}", "TOTAL_COUNT: 3"]
    lines.append(score_line)
    return "\n".join(lines) + "\n"


def grade_with_log(reply: str) -> tuple:
    """Run auto_score with the model call replaced by a fixed reply; return the score and what it printed."""
    message = types.SimpleNamespace(content=reply)
    response = types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])
    client = mock.MagicMock()
    client.chat.completions.create.return_value = response
    frame = {"head": np.zeros((48, 64, 3), dtype=np.uint8)}
    log = io.StringIO()
    with mock.patch.object(auto_score_module, "OpenAI", return_value=client):
        with contextlib.redirect_stdout(log):
            score, _reasoning = auto_score_module.auto_score("clean up the desktop", [frame])
    return score, log.getvalue()


def grade(reply: str) -> float:
    return grade_with_log(reply)[0]


class AutoScoreParseTests(unittest.TestCase):
    def test_score_line_alone(self):
        """With no counts and no [TRUE]/[FALSE] tags in the reply, the SCORE line is all there is."""
        for score_line, expected in [
            ("SCORE: 0.67", 0.67),
            ("SCORE: 1.0", 1.0),
            ("SCORE: .33", 0.33),
            ("SCORE: 1/3 = 0.33", 1 / 3),
            ("SCORE: 1/3", 1 / 3),
            ("SCORE: 1/3.", 1 / 3),
            ("SCORE: 1.0/3.0 = 0.33", 1 / 3),
            ("SCORE: 2/3 = 0.67", 2 / 3),
            ("SCORE: 2 out of 3", 2 / 3),
            ("SCORE: 1 Out Of 3.", 1 / 3),
            ("SCORE: 1 of 3", 1 / 3),
            ("SCORE: 3/3 = 1.0", 1.0),
            ("SCORE: 67%", 0.67),
            ("SCORE: 67 %", 0.67),
            ("SCORE: 0.33 (1/3)", 1 / 3),
            # Not a score in [0, 1]: ignored, and with nothing to fall back to the score stays 0.0.
            ("SCORE: 5", 0.0),
            ("SCORE: -1", 0.0),
            ("SCORE: 4/3", 0.0),
            ("SCORE: 0/0", 0.0),
        ]:
            with self.subTest(score_line=score_line):
                self.assertAlmostEqual(grade(score_line + "\n"), expected)

    def test_score_matching_the_counts_is_kept(self):
        self.assertAlmostEqual(grade(vlm_reply(2, "SCORE: 0.67")), 0.67)
        self.assertAlmostEqual(grade(vlm_reply(1, "SCORE: 1/3 = 0.33")), 1 / 3)
        self.assertAlmostEqual(grade(vlm_reply(3, "SCORE: 1.0")), 1.0)
        self.assertAlmostEqual(grade(vlm_reply(0, "SCORE: 0.0")), 0.0)

    def test_score_that_does_not_match_the_counts_uses_the_counts(self):
        """PROMPT_AUTOSCORE.txt requires SCORE = SATISFIED_COUNT / TOTAL_COUNT."""
        self.assertAlmostEqual(grade(vlm_reply(1, "SCORE: 1.0")), 1 / 3)
        self.assertAlmostEqual(grade(vlm_reply(1, "SCORE: 1/3\nSCORE: 1.0")), 1 / 3)
        self.assertAlmostEqual(grade(vlm_reply(2, "SCORE: 0.7")), 2 / 3)

    def test_unreadable_score_falls_back_to_the_counts(self):
        self.assertAlmostEqual(grade(vlm_reply(1, "SCORE: 5")), 1 / 3)
        self.assertAlmostEqual(grade(vlm_reply(1, "SCORE: -1")), 1 / 3)
        # No count lines: the counts come from the [TRUE]/[FALSE] tags.
        self.assertAlmostEqual(grade(vlm_reply(2, "SCORE: 5", counts=False)), 2 / 3)


class AutoScoreLogTests(unittest.TestCase):
    def test_api_key_is_not_logged(self):
        with mock.patch.object(auto_score_module, "API_KEY", "sk-test-value-not-a-real-key"):
            _score, log = grade_with_log(vlm_reply(1, "SCORE: 0.33"))
        self.assertNotIn("sk-test-value-not-a-real-key", log)
        self.assertIn("API_KEY: set", log)


if __name__ == "__main__":
    unittest.main()
