import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image as PilImage
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor


DEFAULT_MODEL_ID = 'nvidia/segformer-b0-finetuned-cityscapes-512-1024'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate SegFormer-only dashcam semantic segmentation proofs.')
    parser.add_argument(
        '--input-dir',
        default='proof/segformer_raw_dashcam_inputs',
        help='Directory of dashcam/car-view images.')
    parser.add_argument(
        '--output-dir',
        default='proof/segformer_dashcam',
        help='Output directory for SegFormer-only proof images.')
    parser.add_argument('--model-id', default=DEFAULT_MODEL_ID)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--limit', type=int, default=8)
    return parser.parse_args()


def collect_images(input_dir, limit):
    root = Path(input_dir)
    images = []
    for pattern in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
        images.extend(root.glob(pattern))
    images = sorted(images)
    return images[:limit]


def make_overlay(frame, class_mask, label2id):
    colors = {
        'road': (70, 70, 70),
        'sidewalk': (120, 120, 120),
        'person': (0, 220, 255),
        'car': (255, 130, 0),
        'truck': (255, 80, 0),
        'bus': (255, 80, 80),
        'traffic light': (60, 220, 60),
        'traffic sign': (30, 180, 30),
        'building': (180, 180, 180),
        'vegetation': (60, 160, 60),
        'fence': (120, 90, 80),
    }
    color_mask = np.zeros_like(frame)
    for label, color in colors.items():
        class_id = label2id.get(label)
        if class_id is not None:
            color_mask[class_mask == class_id] = color
    active = np.any(color_mask != 0, axis=2)
    overlay = frame.copy()
    blended = cv2.addWeighted(frame, 0.55, color_mask, 0.45, 0)
    overlay[active] = blended[active]
    return overlay, color_mask


def add_label(image, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2
    pad = 10
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (tw + pad * 2, th + pad * 2), (15, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, image, 0.35, 0, image)
    cv2.putText(image, text, (pad, th + pad), font, scale, (255, 255, 255), thickness)


def make_contact_sheet(rows, output_path):
    rendered_rows = []
    for original, overlay, road, sidewalk in rows:
        tiles = []
        for image, label in (
            (original, 'RAW INPUT - unmodified'),
            (overlay, 'segformer semantic overlay'),
            (cv2.cvtColor(road, cv2.COLOR_GRAY2BGR), 'road mask'),
            (cv2.cvtColor(sidewalk, cv2.COLOR_GRAY2BGR), 'sidewalk mask'),
        ):
            tile = cv2.resize(image, (320, 240), interpolation=cv2.INTER_AREA)
            add_label(tile, label)
            tiles.append(tile)
        rendered_rows.append(np.hstack(tiles))
    if rendered_rows:
        cv2.imwrite(str(output_path), np.vstack(rendered_rows))


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = collect_images(args.input_dir, args.limit)
    if not images:
        raise RuntimeError(f'No images found in {args.input_dir}')

    device = torch.device(args.device if args.device != 'cpu' else 'cpu')
    processor = SegformerImageProcessor.from_pretrained(args.model_id)
    model = SegformerForSemanticSegmentation.from_pretrained(args.model_id).to(device)
    model.eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    label2id = {label.lower(): idx for idx, label in id2label.items()}

    rows = []
    records = []
    with torch.no_grad():
        for image_path in images:
            frame = cv2.imread(str(image_path))
            if frame is None:
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            inputs = processor(images=PilImage.fromarray(frame_rgb), return_tensors='pt')
            inputs = {key: value.to(device) for key, value in inputs.items()}
            logits = model(**inputs).logits
            logits = torch.nn.functional.interpolate(
                logits,
                size=frame.shape[:2],
                mode='bilinear',
                align_corners=False,
            )
            class_mask = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

            road = ((class_mask == label2id.get('road', -1)).astype(np.uint8) * 255)
            sidewalk = ((class_mask == label2id.get('sidewalk', -1)).astype(np.uint8) * 255)
            overlay, class_overlay = make_overlay(frame, class_mask, label2id)

            stem = image_path.stem
            cv2.imwrite(str(output_dir / f'{stem}_segformer_overlay.jpg'), overlay)
            cv2.imwrite(str(output_dir / f'{stem}_road_mask.png'), road)
            cv2.imwrite(str(output_dir / f'{stem}_sidewalk_mask.png'), sidewalk)
            cv2.imwrite(str(output_dir / f'{stem}_class_overlay.png'), class_overlay)
            rows.append((frame, overlay, road, sidewalk))

            unique, counts = np.unique(class_mask, return_counts=True)
            class_counts = {
                id2label.get(int(class_id), str(class_id)): int(count)
                for class_id, count in zip(unique.tolist(), counts.tolist())
            }
            records.append({
                'image': str(image_path),
                'road_pixels': int(np.count_nonzero(road)),
                'sidewalk_pixels': int(np.count_nonzero(sidewalk)),
                'top_classes': sorted(
                    class_counts.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )[:8],
            })

    make_contact_sheet(rows, output_dir / 'segformer_dashcam_contact_sheet.jpg')
    (output_dir / 'segformer_dashcam_summary.json').write_text(
        json.dumps({
            'model_id': args.model_id,
            'input_dir': str(Path(args.input_dir).resolve()),
            'image_count': len(records),
            'records': records,
        }, indent=2))
    print(json.dumps({
        'model_id': args.model_id,
        'output_dir': str(output_dir),
        'image_count': len(records),
    }, indent=2))


if __name__ == '__main__':
    main()
