"""Presentation labels must follow the real episode boundary and preserve source video."""
import hashlib
import importlib.util
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

MODULE = Path(__file__).parents[1] / 'src/geniesim_benchmark/utils/video_annotations.py'
spec = importlib.util.spec_from_file_location('video_annotations', MODULE)
labels = importlib.util.module_from_spec(spec)
spec.loader.exec_module(labels)


@unittest.skipUnless(shutil.which('ffmpeg'), 'ffmpeg is required for video integration')
class VideoLabelTests(unittest.TestCase):
    def test_episode_boundary_and_source_preservation(self):
        with tempfile.TemporaryDirectory() as directory:
            source, dest = Path(directory)/'source.mp4', Path(directory)/'labelled.mp4'
            subprocess.run(['ffmpeg','-v','error','-y','-f','lavfi','-i','color=c=gray:s=640x400:r=5',
                            '-t','1','-c:v','libx264',str(source)],check=True)
            before = hashlib.sha256(source.read_bytes()).hexdigest()
            segments = [dict(start_frame=0,end_frame=2,instruction='Fold the first cloth.',episode_idx=0,scene_id='A'),
                        dict(start_frame=2,end_frame=None,instruction='Fold the next cloth.',episode_idx=1,scene_id='B')]
            receipt = labels.label_video(source,dest,segments)
            self.assertEqual(before,hashlib.sha256(source.read_bytes()).hexdigest())
            frames = subprocess.check_output(['ffmpeg','-v','error','-i',str(dest),'-f','rawvideo','-pix_fmt','rgb24','-'])
            frame_size = 640*400*3
            self.assertEqual(len(frames),5*frame_size)
            first, last = frames[:frame_size], frames[-frame_size:]
            self.assertNotEqual(first[:640*50*3],last[:640*50*3])
            self.assertGreater(max(first[640*340*3:]),200)
            self.assertGreater(max(last[640*340*3:]),200)
            self.assertEqual(receipt,labels.label_video(source,dest,segments))
            with self.assertRaises(ValueError):
                labels.label_video(source,source,segments)


if __name__ == '__main__':
    unittest.main()
