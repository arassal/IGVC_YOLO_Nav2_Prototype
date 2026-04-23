import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate combined road/lane segmentation plus traffic cone proof images.')
    parser.add_argument(
        '--project-root',
        default='/home/alexander/Desktop/seg',
        help='YOLOPv2 project root containing utils/.')
    parser.add_argument(
        '--segmentation-weights',
        default='/home/alexander/Desktop/seg/data/weights/yolopv2.pt',
        help='YOLOPv2 TorchScript weights for drivable area and lane masks.')
    parser.add_argument(
        '--object-model',
        default='models/roboflow_logistics_yolov8.pt',
        help='YOLOv8 object model with traffic cone class.')
    parser.add_argument(
        '--input-dir',
        default='proof/source_images',
        help='Road images containing traffic cones.')
    parser.add_argument(
        '--output-dir',
        default='proof/combined',
        help='Output directory for combined proof images.')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--object-conf', type=float, default=0.35)
    parser.add_argument('--object-imgsz', type=int, default=1280)
    parser.add_argument('--limit', type=int, default=6)
    parser.add_argument(
        '--main-image-stem',
        default='road_cars_cones_input',
        help='Input image stem to use for the main single-image proof.')
    return parser.parse_args()


def collect_images(path):
    root = Path(path)
    return sorted(root.glob('*.jpg'), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)


def draw_cones(image, result):
    cones = []
    for box in result.boxes:
        class_name = result.names[int(box.cls[0])]
        if class_name != 'traffic cone':
            continue
        xyxy = [float(v) for v in box.xyxy[0]]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = [int(v) for v in xyxy]
        thickness = max(2, round(min(image.shape[:2]) / 220))
        text_scale = max(0.55, min(image.shape[:2]) / 1000)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 140, 255), thickness)
        cv2.putText(
            image, f'cone {conf:.2f}', (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, text_scale, (0, 140, 255),
            thickness, cv2.LINE_AA)
        cones.append({'confidence': conf, 'xyxy': xyxy})
    return cones


def make_contact_sheet(rows, output_path):
    if not rows:
        return
    rendered = []
    for original, overlay, drivable, lane in rows:
        w, h = 320, 240
        original = cv2.resize(original, (w, h), interpolation=cv2.INTER_AREA)
        overlay = cv2.resize(overlay, (w, h), interpolation=cv2.INTER_AREA)
        drivable = cv2.cvtColor(
            cv2.resize(drivable, (w, h), interpolation=cv2.INTER_NEAREST),
            cv2.COLOR_GRAY2BGR,
        )
        lane = cv2.cvtColor(
            cv2.resize(lane, (w, h), interpolation=cv2.INTER_NEAREST),
            cv2.COLOR_GRAY2BGR,
        )
        rendered.append(np.hstack([original, overlay, drivable, lane]))
    cv2.imwrite(str(output_path), np.vstack(rendered))


def add_badge(image, text):
    pad = 10
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 2
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x1, y1 = 10, 10
    x2, y2 = x1 + tw + pad * 2, y1 + th + pad * 2
    overlay = image.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 30, 30), -1)
    cv2.addWeighted(overlay, 0.55, image, 0.45, 0, image)
    cv2.putText(
        image, text, (x1 + pad, y2 - pad),
        font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def main():
    args = parse_args()
    repo_root = Path.cwd()
    project_root = Path(args.project_root)
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from utils.utils import driving_area_mask, lane_line_mask, letterbox, show_seg_result

    output_dir = repo_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cpu' if args.device == 'cpu' else args.device)
    seg_model = torch.jit.load(args.segmentation_weights, map_location=device).to(device)
    seg_model.eval()

    object_model = YOLO(str(repo_root / args.object_model))
    images = collect_images(repo_root / args.input_dir)[:args.limit]
    if not images:
        raise RuntimeError(f'No images found in {args.input_dir}')

    contact_rows = []
    main_image = None
    with torch.no_grad():
        for image_path in images:
            frame = cv2.imread(str(image_path))
            if frame is None:
                continue
            original_size = (frame.shape[1], frame.shape[0])

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
            drivable_mask = cv2.resize(
                (da_mask * 255).astype(np.uint8),
                original_size,
                interpolation=cv2.INTER_NEAREST,
            )
            lane_mask = cv2.resize(
                (ll_mask * 255).astype(np.uint8),
                original_size,
                interpolation=cv2.INTER_NEAREST,
            )

            semantic_overlay = seg_input.copy()
            show_seg_result(semantic_overlay, (da_mask, ll_mask), is_demo=True)
            semantic_overlay = cv2.resize(
                semantic_overlay, original_size, interpolation=cv2.INTER_LINEAR)

            object_result = object_model.predict(
                str(image_path),
                conf=args.object_conf,
                imgsz=args.object_imgsz,
                verbose=False,
                device=args.device,
            )[0]
            cones = draw_cones(semantic_overlay, object_result)

            add_badge(semantic_overlay, f'road/lane segmentation + {len(cones)} traffic cones')

            output_path = output_dir / f'{image_path.stem}_semantic_cones.jpg'
            cv2.imwrite(str(output_path), semantic_overlay)
            cv2.imwrite(str(output_dir / f'{image_path.stem}_drivable_mask.png'), drivable_mask)
            cv2.imwrite(str(output_dir / f'{image_path.stem}_lane_mask.png'), lane_mask)
            contact_rows.append((frame, semantic_overlay, drivable_mask, lane_mask))
            if image_path.stem == args.main_image_stem:
                main_image = semantic_overlay

    if contact_rows:
        cv2.imwrite(
            str(output_dir / 'semantic_segmentation_plus_cones_road.jpg'),
            main_image if main_image is not None else contact_rows[0][1],
        )
        make_contact_sheet(contact_rows, output_dir / 'semantic_segmentation_plus_cones_contact_sheet.jpg')
    print(f'Wrote combined proof images to {output_dir}')


if __name__ == '__main__':
    main()
