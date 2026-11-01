#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# License: Mozilla Public License Version 2.0
"""Run the existing replay entry and verify newly rendered/converted output."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


def verify(output, expected, log, config):
    import h5py
    import numpy as np

    report = {'success': False, 'errors': [], 'episodes': []}
    errors = report['errors']
    episodes = sorted(p for p in output.iterdir() if (p / 'state.json').is_file()) if output.exists() else []
    if len(episodes) != expected:
        errors.append(f'Expected {expected} episodes, found {len(episodes)}')
    if log.count('SimDataConverter done:') != expected:
        errors.append('Conversion success count does not match expected episodes')
    preset = json.loads((Path('replay/preset_configs') / f'{config["preset"]}.json').read_text())
    cfg = {**preset.get('replay_config', {}), **config.get('replay_config', {})}
    for episode in episodes:
        result = {'path': str(episode), 'errors': [], 'videos': []}
        try:
            state = json.loads((episode / 'state.json').read_text())
            frames = len(state['frames'])
            for filename in ('recording_info.json', 'metadata.yaml', 'replay_done'):
                if not (episode / filename).is_file():
                    raise ValueError(f'Missing {filename}')
            if not config.get('sub_mode') and not (episode / 'data_info.json').is_file():
                raise ValueError('Missing collection data_info.json')
            with h5py.File(episode / 'aligned_joints.h5') as data:
                if len(data) != frames or frames == 0:
                    raise ValueError(f'H5 groups {len(data)} != source frames {frames}')
                timestamps = np.array([data[str(index)]['main_timestamp'][()] for index in range(frames)], dtype=float)
                joints = np.stack([data[str(index)]['state/joint/position'][()] for index in range(frames)])
                if not np.isfinite(joints).all() or not np.isfinite(timestamps).all() or not (np.diff(timestamps) > 0).all():
                    raise ValueError('H5 joint values/timestamps are invalid')
                result.update({'frames': frames, 'joint_shape': list(joints.shape)})
            camera_info = json.loads((episode / 'recording_info.json').read_text()).get('camera_info', {})
            expected_videos = set()
            for camera in camera_info.values():
                outputs = camera.get('output', {})
                if cfg.get('image_mode'):
                    for key in ('rgb', 'depth'):
                        if key in outputs:
                            for index in range(frames):
                                image = episode / outputs[key].replace('{frame_num}', str(index))
                                if not image.is_file() or image.stat().st_size == 0:
                                    raise ValueError(f'Missing camera frame: {image}')
                else:
                    if outputs.get('video'):
                        expected_videos.add(Path(outputs['video']).name)
                    if outputs.get('depth'):
                        expected_videos.add(Path(outputs['depth']).stem + '.mp4')
            videos = sorted((episode / 'observations/videos').glob('*.mp4'))
            missing = expected_videos - {video.name for video in videos}
            if missing:
                raise ValueError(f'Missing camera videos: {sorted(missing)}')
            if not cfg.get('image_mode') and state.get('cameras') and not videos:
                raise ValueError('No camera videos were written')
            for video in videos:
                command = ['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
                           '-show_entries', 'stream=nb_read_frames,avg_frame_rate,width,height', '-of', 'json', str(video)]
                info = json.loads(subprocess.check_output(command, text=True))['streams'][0]
                if int(info['nb_read_frames']) != frames:
                    raise ValueError(f'{video.name}: video frame count differs from source')
                result['videos'].append({'name': video.name, **info})
        except Exception as exc:
            result['errors'].append(str(exc))
        report['episodes'].append(result)
    report['success'] = not errors and all(not ep['errors'] for ep in report['episodes'])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--expected-episodes', type=int, required=True)
    parser.add_argument('--headless', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    run = Path(args.config).parent
    log_path = run / 'replay.log'
    command = [sys.executable, '-m', 'replay', '--input_dir', args.input_dir,
               '--config_json', args.config, '--recording_root', args.output_dir]
    if not args.headless:
        command.append('--no-headless')
    with log_path.open('wb') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        report = {'success': False, 'errors': [f'Replay exited with {result.returncode}; see {log_path}']}
    else:
        report = verify(Path(args.output_dir), args.expected_episodes, log_path.read_text(errors='replace'),
                        json.loads(Path(args.config).read_text()))
    (run / 'output_check.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report['success'] else result.returncode or 1


if __name__ == '__main__':
    raise SystemExit(main())
