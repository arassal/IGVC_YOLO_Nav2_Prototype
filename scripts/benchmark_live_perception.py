import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description='Benchmark the current road/lane plus object perception stack.')
    parser.add_argument('--project-root', default='/home/alexander/Desktop/seg')
    parser.add_argument(
        '--segmentation-weights',
        default='/home/alexander/Desktop/seg/data/weights/yolopv2.pt')
    parser.add_argument('--object-model', default='models/roboflow_logistics_yolov8.pt')
    parser.add_argument('--image-dir', default='proof/source_images')
    parser.add_argument('--output-json', default='validation/benchmark_report.json')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--limit', type=int, default=20)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--object-conf', type=float, default=0.35)
    parser.add_argument('--object-imgsz', type=int, default=640)
    return parser.parse_args()


def collect_images(image_dir, limit):
    root = Path(image_dir)
    images = []
    for pattern in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
        images.extend(root.glob(pattern))
    images = sorted(images)
    return images[:limit]


def summarize(values):
    if not values:
        return {'count': 0}
    return {
        'count': len(values),
        'mean_ms': statistics.mean(values),
        'median_ms': statistics.median(values),
        'min_ms': min(values),
        'max_ms': max(values),
        'fps_mean': 1000.0 / statistics.mean(values) if statistics.mean(values) else 0.0,
    }


def main():
    args = parse_args()
    repo_root = Path.cwd()
    project_root = Path(args.project_root)
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from utils.utils import driving_area_mask, lane_line_mask, letterbox

    images = collect_images(args.image_dir, args.limit)
    if not images:
        raise RuntimeError(f'No images found in {args.image_dir}')

    device = torch.device('cpu' if args.device == 'cpu' else args.device)
    seg_model = torch.jit.load(args.segmentation_weights, map_location=device).to(device)
    seg_model.eval()
    object_model = YOLO(str(repo_root / args.object_model))

    timings = []
    per_image = []

    with torch.no_grad():
        for index, image_path in enumerate(images):
            frame = cv2.imread(str(image_path))
            if frame is None:
                continue

            start = time.perf_counter()
            seg_input = cv2.resize(frame.copy(), (1280, 720), interpolation=cv2.INTER_LINEAR)
            img = letterbox(seg_input, 640, stride=32)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)
            img = np.ascontiguousarray(img)
            tensor = torch.from_numpy(img).to(device).float() / 255.0
            if tensor.ndimension() == 3:
                tensor = tensor.unsqueeze(0)
            _pred, seg, ll = seg_model(tensor)
            da_mask = driving_area_mask(seg)
            ll_mask = lane_line_mask(ll)

            object_result = object_model.predict(
                str(image_path),
                conf=args.object_conf,
                imgsz=args.object_imgsz,
                verbose=False,
                device=args.device,
            )[0]
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            detections = []
            for box in object_result.boxes:
                class_name = object_result.names[int(box.cls[0])]
                detections.append({
                    'class_name': class_name,
                    'confidence': float(box.conf[0]),
                })

            record = {
                'image': str(image_path),
                'elapsed_ms': elapsed_ms,
                'drivable_pixels': int(np.count_nonzero(da_mask)),
                'lane_pixels': int(np.count_nonzero(ll_mask)),
                'object_count': len(detections),
                'object_classes': sorted({det['class_name'] for det in detections}),
            }
            per_image.append(record)
            if index >= args.warmup:
                timings.append(elapsed_ms)

    report = {
        'device': args.device,
        'image_dir': str(Path(args.image_dir).resolve()),
        'image_count': len(per_image),
        'warmup_frames': args.warmup,
        'timing_summary': summarize(timings),
        'per_image': per_image,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report['timing_summary'], indent=2))
    print(f'Wrote benchmark report to {output_path}')


if __name__ == '__main__':
    main()
