# Copyright (c) 2026, AgiBot Inc. All Rights Reserved.
# License: Mozilla Public License Version 2.0
"""Create labelled presentation videos without changing policy images or native scores."""
from pathlib import Path
import hashlib
import json
import subprocess
import tempfile
import textwrap

LABEL_SCHEMA = 'geniesim.video-labels.v1'


def label_video(source, destination, segments):
    """Bake instruction (top) and zero-based episode/scene (bottom-right).

    Segment boundaries use decoded video frame indices, including VFR input.
    ``end_frame=None`` labels through the end. Original video is retained.
    """
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError('labelled video must use a separate destination')
    if not segments or segments[0]['start_frame'] != 0:
        raise ValueError('labels must cover the video from its first frame')
    for i, segment in enumerate(segments):
        if not str(segment['instruction']).strip():
            raise ValueError('instruction must be nonempty')
        if type(segment['episode_idx']) is not int or segment['episode_idx'] < 0:
            raise ValueError('episode_idx must be a nonnegative integer')
        end = segment.get('end_frame')
        if end is not None and end <= segment['start_frame']:
            raise ValueError('invalid frame interval')
        if i and segments[i-1].get('end_frame') != segment['start_frame']:
            raise ValueError('segment coverage must be contiguous')
    receipt = {'schema': LABEL_SCHEMA, 'source': str(source),
               'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'segments': segments}
    metadata = destination.with_suffix('.labels.json')
    if destination.exists() and metadata.exists():
        previous = json.loads(metadata.read_text())
        if all(previous.get(k) == v for k, v in receipt.items()):
            if previous.get('video_sha256') == hashlib.sha256(destination.read_bytes()).hexdigest():
                return previous
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='geniesim-labels-') as temporary:
        filters = ['scale=640:-2']
        for i, segment in enumerate(segments):
            instruction = Path(temporary) / f'instruction{i}.txt'
            text = '\n'.join(textwrap.wrap(str(segment['instruction']), width=64))
            instruction.write_text(text)
            label = Path(temporary) / f'episode{i}.txt'
            label.write_text(f"Episode: {segment['episode_idx']}\nScene: {segment['scene_id']}")
            start, end = segment['start_frame'], segment.get('end_frame')
            enable = f'gte(n,{start})' if end is None else f'between(n,{start},{end-1})'
            height = 20 * len(text.splitlines()) + 16
            filters += [f"drawbox=x=0:y=0:w=iw:h={height}:color=black@0.65:t=fill:enable='{enable}'",
                        f"drawtext=textfile={instruction}:expansion=none:fontsize=18:fontcolor=white:x=10:y=8:line_spacing=2:enable='{enable}'",
                        f"drawtext=textfile={label}:expansion=none:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.65:boxborderw=6:x=w-tw-12:y=h-th-12:line_spacing=3:enable='{enable}'"]
        temporary_video = destination.with_suffix('.tmp.mp4')
        result = subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(source), '-an', '-vf', ','.join(filters),
                                 '-c:v', 'libx264', '-threads', '2', '-preset', 'veryfast', '-crf', '26',
                                 '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(temporary_video)],
                                capture_output=True, text=True, timeout=900)
        if result.returncode:
            temporary_video.unlink(missing_ok=True)
            raise RuntimeError(result.stderr[-2000:])
        subprocess.run(['ffmpeg', '-v', 'error', '-i', str(temporary_video), '-frames:v', '1', '-f', 'null', '-'],
                       check=True, capture_output=True, timeout=30)
        temporary_video.replace(destination)
    receipt['video_sha256'] = hashlib.sha256(destination.read_bytes()).hexdigest()
    metadata.write_text(json.dumps(receipt, ensure_ascii=False, indent=2))
    return receipt
