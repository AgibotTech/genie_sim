# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# Author: Genie Sim Team
# License: Mozilla Public License Version 2.0

import unittest

from geniesim_benchmark.dataset.convert.agibot_to_lerobot import _read_extrinsic


class ReadExtrinsicTest(unittest.TestCase):
    def test_reads_flat_legacy_layout(self):
        rotation = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        translation = [1, 2, 3]

        self.assertEqual(
            _read_extrinsic({"rotation": rotation, "translation": translation}),
            (rotation, translation),
        )

    def test_reads_nested_data_collection_layout(self):
        rotation = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        translation = [1, 2, 3]

        self.assertEqual(
            _read_extrinsic(
                {
                    "extrinsic": {
                        "rotation_matrix": rotation,
                        "translation_vector": translation,
                    }
                }
            ),
            (rotation, translation),
        )

    def test_rejects_incomplete_layout(self):
        with self.assertRaisesRegex(KeyError, "extrinsic rotation/translation"):
            _read_extrinsic({"extrinsic": {"rotation_matrix": [[1, 0, 0]]}})


if __name__ == "__main__":
    unittest.main()
