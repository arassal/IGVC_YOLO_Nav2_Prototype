import argparse
import json
from pathlib import Path

import cv2


def parse_args():
    parser = argparse.ArgumentParser(
        description='Extract evenly spaced validation frames from a video file.')
    parser.add_argument('--input-video', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--max-frames', type=int, default=200)
    parser.add_argument('--step', type=int, default=30)
    parser.add_argument('--prefix', default='frame')
    parser.add_argument('--extension', default='jpg', choices=['jpg', 'jpeg', 'png'])
    parser.add_argument('--jpeg-quality', type=int, default=95)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.input_video)
    if not cap.isOpened():
        raise RuntimeError(f'Could not open video: {args.input_video}')

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    saved = []
    frame_idx = 0

    write_params = []
    if args.extension in {'jpg', 'jpeg'}:
        write_params = [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]

    while len(saved) < args.max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % args.step == 0:
            name = f'{args.prefix}_{len(saved) + 1:06d}.{args.extension}'
            path = output_dir / name
            if not cv2.imwrite(str(path), frame, write_params):
                raise RuntimeError(f'Failed to write frame: {path}')
            saved.append({
                'file': name,
                'source_frame_index': frame_idx,
                'source_time_sec': frame_idx / fps if fps else None,
            })
        frame_idx += 1

    cap.release()
    manifest = {
        'input_video': str(Path(args.input_video).resolve()),
        'output_dir': str(output_dir.resolve()),
        'fps': fps,
        'total_source_frames': total_frames,
        'step': args.step,
        'saved_frames': len(saved),
        'frames': saved,
    }
    (output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({
        'input_video': manifest['input_video'],
        'saved_frames': len(saved),
        'output_dir': manifest['output_dir'],
    }, indent=2))


if __name__ == '__main__':
    main()
