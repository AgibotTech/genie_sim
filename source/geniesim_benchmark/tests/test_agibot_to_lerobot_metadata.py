from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import call, patch

import numpy as np
from PIL import Image

from geniesim_benchmark.dataset.convert import agibot_to_lerobot


class ImageStatsTest(TestCase):
    def test_image_stats_are_normalized_and_have_nonzero_count(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            image_paths = [root / "red.png", root / "blue.png"]
            Image.new("RGB", (2, 2), (255, 0, 0)).save(image_paths[0])
            Image.new("RGB", (2, 2), (0, 0, 255)).save(image_paths[1])

            stats = agibot_to_lerobot._compute_image_stats(image_paths)

        self.assertEqual(stats["count"], [2])
        np.testing.assert_allclose(stats["min"], np.zeros((3, 1, 1)))
        np.testing.assert_allclose(stats["max"], np.array([1.0, 0.0, 1.0]).reshape(3, 1, 1))
        np.testing.assert_allclose(stats["mean"], np.array([0.5, 0.0, 0.5]).reshape(3, 1, 1))
        np.testing.assert_allclose(stats["std"], np.array([0.5, 0.0, 0.5]).reshape(3, 1, 1))


class GlobalIndexTest(TestCase):
    def test_batch_conversion_passes_cumulative_global_index_offsets(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "input"
            output_dir = root / "output"
            episodes = [input_dir / "episode_000", input_dir / "episode_001"]
            input_dir.mkdir()

            with (
                patch.object(agibot_to_lerobot, "_require_ffmpeg"),
                patch.object(agibot_to_lerobot, "_require_heavy_deps"),
                patch.object(agibot_to_lerobot, "detect_episodes", return_value=episodes),
                patch.object(
                    agibot_to_lerobot,
                    "convert_episode",
                    side_effect=[
                        {"episode_index": 0, "length": 3},
                        {"episode_index": 1, "length": 2},
                    ],
                ) as convert_episode,
                patch.object(agibot_to_lerobot, "generate_meta"),
            ):
                agibot_to_lerobot.convert_agibot_to_lerobot(input_dir, output_dir)

        self.assertEqual(
            convert_episode.call_args_list,
            [
                call(
                    episodes[0],
                    output_dir.resolve(),
                    0,
                    None,
                    fps=30.0,
                    global_index_start=0,
                ),
                call(
                    episodes[1],
                    output_dir.resolve(),
                    1,
                    None,
                    fps=30.0,
                    global_index_start=3,
                ),
            ],
        )
