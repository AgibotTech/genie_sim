from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from geniesim_benchmark.dataset.convert import agibot_to_lerobot


class EncodeVideosTest(TestCase):
    def setUp(self):
        self._temporary_directory = TemporaryDirectory()
        self.addCleanup(self._temporary_directory.cleanup)
        self.output_dir = Path(self._temporary_directory.name)

    def test_rgb_directories_match_lerobot_feature_keys(self):
        with patch.object(
            agibot_to_lerobot.subprocess,
            "run",
            return_value=SimpleNamespace(returncode=0, stderr=""),
        ) as run:
            agibot_to_lerobot.encode_videos(Path("/input"), self.output_dir, 7)

        output_paths = [Path(call.args[0][-1]) for call in run.call_args_list]
        relative_paths = {path.relative_to(self.output_dir) for path in output_paths}

        self.assertEqual(len(relative_paths), 6)
        self.assertIn(
            Path("videos/chunk-000/observation.images.top_head/episode_000007.mp4"),
            relative_paths,
        )
        self.assertIn(
            Path("videos/chunk-000/observation.images.hand_left/episode_000007.mp4"),
            relative_paths,
        )
        self.assertIn(
            Path("videos/chunk-000/observation.images.hand_right/episode_000007.mp4"),
            relative_paths,
        )
        self.assertIn(
            Path("videos/chunk-000/top_head_depth/episode_000007.mp4"),
            relative_paths,
        )
