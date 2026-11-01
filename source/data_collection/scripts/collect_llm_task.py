#!/usr/bin/env python3
# Copyright (c) 2023-2026, AgiBot Inc. All Rights Reserved.
# License: Mozilla Public License Version 2.0
"""Collect one LLM task serially, reusing the existing server/episode pipeline."""

import argparse
from dataclasses import replace
import json
import random
from pathlib import Path

from resolve_llm_task_root import resolve_llm_task_root
from llm_task_policy import validate_run_times
from collection_runtime import (
    load_json, now_iso, run_episode_subprocess, scene_dirs,
    server_config_from_task_info, start_server, stop_server, write_json,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True)
    parser.add_argument('--scene', default='random')
    parser.add_argument('--run-times', type=int, default=1)
    parser.add_argument('--report-dir', required=True)
    parser.add_argument('--record', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--headless', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--server-start-timeout', type=int, default=420)
    parser.add_argument('--episode-timeout', type=int, default=600)
    parser.add_argument('--task-result-timeout', type=int, default=600)
    args = parser.parse_args()
    if args.run_times < 1 or Path(args.task).name != args.task or args.task in ('.', '..'):
        parser.error('Use a task directory name and positive run-times')
    root = resolve_llm_task_root()
    if root is None:
        parser.error('No LLM task library configured')
    task = root / args.task
    scenes = scene_dirs(task, 2**31)
    if not scenes or (args.scene != 'random' and args.scene not in scenes):
        parser.error(f'Scene not found for {args.task}: {args.scene}')
    try:
        validate_run_times(task, args.scene, args.run_times)
    except ValueError as exc:
        parser.error(str(exc))
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report = {'task': args.task, 'task_root': str(root), 'started_at': now_iso(), 'record': args.record, 'episodes': []}
    for index in range(args.run_times):
        scene = random.choice(scenes) if args.scene == 'random' else args.scene
        metadata = task / scene / 'task_info.json'
        if not metadata.is_file():
            metadata = task / 'task_info.json'
        cfg = replace(server_config_from_task_info(load_json(metadata), publish_ros=args.record),
                      enable_render=args.render, headless=args.headless)
        server = None
        server_settings = {'task_info_path': str(metadata), 'server_config': cfg.as_dict(),
                           'server_argv': cfg.argv()}
        write_json(report_dir / f'server_config_{index + 1}.json', server_settings)
        print(f'[server] task_info={metadata}; backend={cfg.backend}; '
              f'physics_frequency={cfg.physics_step}; enable_deformable={cfg.enable_deformable}', flush=True)
        try:
            print(f'[collection] {args.task}/{scene}, episode {index + 1}/{args.run_times}', flush=True)
            server = start_server(cfg, report_dir / f'server_{index + 1}.log', args.server_start_timeout)
            episode = run_episode_subprocess(
                root, args.task, scene, report_dir / f'client_{index + 1}.log',
                report_dir / f'episode_{index + 1}.json', 'localhost:50051', args.record,
                args.episode_timeout, args.task_result_timeout,
            )
            if args.record and episode.get('success'):
                recording = Path(episode['recording_dir'])
                for filename in ('state.json', 'metadata.yaml', 'aligned_joints.h5'):
                    if not (recording / filename).is_file():
                        raise ValueError(f'Missing recorded artifact: {filename}')
                frames = load_json(recording / 'state.json').get('frames', [])
                if not frames:
                    raise ValueError('Recording contains no frames')
                episode['frames'] = len(frames)
        except Exception as exc:
            episode = {'task': args.task, 'scene': scene, 'success': False, 'error': str(exc)}
        finally:
            stop_server(server)
        episode.update(server_settings)
        report['episodes'].append(episode)
        write_json(report_dir / 'collection_result.json', report)
        if not episode.get('success'):
            break
    report['finished_at'] = now_iso()
    report['success'] = len(report['episodes']) == args.run_times and all(ep.get('success') for ep in report['episodes'])
    write_json(report_dir / 'collection_result.json', report)
    print(f'[result] success={report["success"]}; {report_dir / "collection_result.json"}', flush=True)
    return 0 if report['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
