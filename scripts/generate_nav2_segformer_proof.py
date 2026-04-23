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
        description='Generate SegFormer + Nav2 proof assets from dashcam images.')
    parser.add_argument('--input-dir', default='/home/alexander/Desktop/img')
    parser.add_argument('--output-dir', default='proof/segformer_nav2_igvc')
    parser.add_argument('--model-id', default=DEFAULT_MODEL_ID)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--limit', type=int, default=6)
    parser.add_argument('--grid-resolution', type=float, default=0.05)
    parser.add_argument('--grid-width-m', type=float, default=6.0)
    parser.add_argument('--grid-length-m', type=float, default=8.0)
    return parser.parse_args()


def collect_images(input_dir, limit):
    root = Path(input_dir)
    images = []
    for pattern in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
        images.extend(sorted(root.glob(pattern)))
    return images[:limit]


def binary_mask(class_mask, label2id, class_name):
    class_id = label2id.get(class_name)
    if class_id is None:
        return np.zeros(class_mask.shape, dtype=np.uint8)
    return (class_mask == class_id).astype(np.uint8) * 255


def refine_masks(frame, road_mask_raw, sidewalk_mask):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    height, width = road_mask_raw.shape
    roi = np.zeros((height, width), dtype=np.uint8)
    roi[int(height * 0.35):, :] = 255

    asphalt_mask = cv2.inRange(hsv, (0, 0, 35), (179, 80, 185))
    white_mask = cv2.inRange(hsv, (0, 0, 180), (179, 55, 255))
    yellow_mask = cv2.inRange(hsv, (12, 55, 110), (42, 255, 255))
    lane_hint_mask = cv2.bitwise_or(white_mask, yellow_mask)

    kernel_large = np.ones((21, 21), np.uint8)
    kernel_small = np.ones((5, 5), np.uint8)
    road_neighborhood = cv2.dilate(road_mask_raw, kernel_large, iterations=1)
    asphalt_support = cv2.bitwise_and(asphalt_mask, road_neighborhood)
    asphalt_support = cv2.bitwise_and(asphalt_support, roi)
    asphalt_support = cv2.bitwise_and(asphalt_support, cv2.bitwise_not(sidewalk_mask))

    road_mask = cv2.bitwise_or(road_mask_raw, asphalt_support)
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel_large)
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, kernel_small)

    lane_hint_mask = cv2.bitwise_and(lane_hint_mask, roi)
    lane_hint_mask = cv2.bitwise_and(
        lane_hint_mask, cv2.dilate(road_mask, kernel_large, iterations=1))
    lane_hint_mask = cv2.morphologyEx(
        lane_hint_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    lane_hint_mask = cv2.dilate(lane_hint_mask, kernel_small, iterations=1)
    return road_mask, lane_hint_mask


def make_overlay(frame, class_mask, label2id, road_mask, lane_hint_mask):
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
    }
    color_mask = np.zeros_like(frame)
    for label, color in colors.items():
        class_id = label2id.get(label)
        if class_id is not None:
            color_mask[class_mask == class_id] = color
    color_mask[road_mask > 0] = (80, 80, 80)
    color_mask[lane_hint_mask > 0] = (0, 255, 255)
    active = np.any(color_mask != 0, axis=2)
    overlay = frame.copy()
    blended = cv2.addWeighted(frame, 0.55, color_mask, 0.45, 0)
    overlay[active] = blended[active]
    return overlay


def select_largest_lane_component(mask):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(mask)
    best_index = -1
    best_score = 0
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        width = stats[label, cv2.CC_STAT_WIDTH]
        height = stats[label, cv2.CC_STAT_HEIGHT]
        if area < 8 or height < 6:
            continue
        score = area + (height * 2) - width
        if score > best_score:
            best_score = score
            best_index = label
    if best_index == -1:
        return np.zeros_like(mask)
    return np.where(labels == best_index, 255, 0).astype(np.uint8)


def build_lane_corridor(left_boundary, right_boundary):
    corridor = np.zeros_like(left_boundary)
    for row in range(corridor.shape[0]):
        left_cols = np.flatnonzero(left_boundary[row] > 0)
        right_cols = np.flatnonzero(right_boundary[row] > 0)
        if left_cols.size == 0 or right_cols.size == 0:
            continue
        left_x = int(np.max(left_cols))
        right_x = int(np.min(right_cols))
        if right_x <= left_x:
            continue
        corridor[row, left_x:right_x + 1] = 255
    corridor = cv2.morphologyEx(corridor, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return corridor


def extract_igvc_lane_features(frame, road_mask, lane_hint_mask, grid_width_cells, grid_height_cells):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    height, width = road_mask.shape
    roi = np.zeros((height, width), dtype=np.uint8)
    roi[int(height * 0.45):, :] = 255
    white_mask = cv2.inRange(hsv, (0, 0, 165), (179, 70, 255))
    white_mask = cv2.bitwise_and(white_mask, roi)
    road_support = cv2.dilate(road_mask, np.ones((25, 25), np.uint8), iterations=1)
    white_mask = cv2.bitwise_and(white_mask, road_support)
    white_mask = cv2.bitwise_or(white_mask, lane_hint_mask)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    white_mask = cv2.dilate(white_mask, np.ones((3, 3), np.uint8), iterations=1)

    src = np.float32([
        [width * 0.05, height * 0.98],
        [width * 0.95, height * 0.98],
        [width * 0.65, height * 0.62],
        [width * 0.35, height * 0.62],
    ])
    dst = np.float32([
        [0, grid_height_cells - 1],
        [grid_width_cells - 1, grid_height_cells - 1],
        [grid_width_cells - 1, 0],
        [0, 0],
    ])
    transform = cv2.getPerspectiveTransform(src, dst)
    white_bev = cv2.warpPerspective(
        white_mask, transform, (grid_width_cells, grid_height_cells), flags=cv2.INTER_NEAREST)
    white_bev = cv2.morphologyEx(white_bev, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    white_bev = cv2.dilate(white_bev, np.ones((3, 3), np.uint8), iterations=1)

    half = grid_width_cells // 2
    left = select_largest_lane_component(white_bev[:, :half])
    right = select_largest_lane_component(white_bev[:, half:])
    left_full = np.zeros_like(white_bev)
    right_full = np.zeros_like(white_bev)
    left_full[:, :half] = left
    right_full[:, half:] = right
    corridor = build_lane_corridor(left_full, right_full)
    if np.count_nonzero(corridor) == 0:
        corridor = white_bev.copy()
    return white_mask, white_bev, corridor


def project_nav2(road_mask, lane_hint_mask, lane_corridor, grid_width_cells, grid_height_cells):
    height, width = road_mask.shape
    src = np.float32([
        [width * 0.05, height * 0.98],
        [width * 0.95, height * 0.98],
        [width * 0.65, height * 0.62],
        [width * 0.35, height * 0.62],
    ])
    dst = np.float32([
        [0, grid_height_cells - 1],
        [grid_width_cells - 1, grid_height_cells - 1],
        [grid_width_cells - 1, 0],
        [0, 0],
    ])
    transform = cv2.getPerspectiveTransform(src, dst)
    road_bev = cv2.warpPerspective(
        road_mask, transform, (grid_width_cells, grid_height_cells), flags=cv2.INTER_NEAREST)
    lane_bev = cv2.warpPerspective(
        lane_hint_mask, transform, (grid_width_cells, grid_height_cells), flags=cv2.INTER_NEAREST)
    road_bev = cv2.morphologyEx(road_bev, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    drivable = np.where(road_bev > 0, 255, 0).astype(np.uint8)
    if np.count_nonzero(lane_corridor) > 0:
        drivable = cv2.bitwise_and(drivable, cv2.dilate(
            lane_corridor, np.ones((9, 9), np.uint8), iterations=1))
    drivable = cv2.bitwise_or(drivable, lane_bev)
    drivable = cv2.morphologyEx(drivable, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    keepout = np.where(drivable > 0, 0, 255).astype(np.uint8)
    return keepout, drivable


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
    for original, overlay, road, lane_hint, igvc_lane_bev, corridor, keepout in rows:
        tiles = []
        for image, label in (
            (original, 'raw input'),
            (overlay, 'segformer + hsv'),
            (cv2.cvtColor(road, cv2.COLOR_GRAY2BGR), 'refined road mask'),
            (cv2.cvtColor(lane_hint, cv2.COLOR_GRAY2BGR), 'lane hint mask'),
            (cv2.cvtColor(igvc_lane_bev, cv2.COLOR_GRAY2BGR), 'igvc lane bev'),
            (cv2.cvtColor(corridor, cv2.COLOR_GRAY2BGR), 'igvc corridor'),
            (cv2.cvtColor(keepout, cv2.COLOR_GRAY2BGR), 'nav2 keepout bev'),
        ):
            tile = cv2.resize(image, (280, 200), interpolation=cv2.INTER_AREA)
            add_label(tile, label)
            tiles.append(tile)
        rendered_rows.append(np.hstack(tiles))
    if rendered_rows:
        cv2.imwrite(str(output_path), np.vstack(rendered_rows))


def main():
    args = parse_args()
    input_images = collect_images(args.input_dir, args.limit)
    if not input_images:
        raise RuntimeError(f'No images found in {args.input_dir}')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grid_width_cells = max(1, int(round(args.grid_width_m / args.grid_resolution)))
    grid_height_cells = max(1, int(round(args.grid_length_m / args.grid_resolution)))
    device = torch.device(args.device if args.device != 'cpu' else 'cpu')
    processor = SegformerImageProcessor.from_pretrained(args.model_id)
    model = SegformerForSemanticSegmentation.from_pretrained(args.model_id).to(device)
    model.eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    label2id = {label.lower(): idx for idx, label in id2label.items()}

    rows = []
    records = []
    with torch.no_grad():
        for image_path in input_images:
            frame = cv2.imread(str(image_path))
            if frame is None:
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            inputs = processor(images=PilImage.fromarray(frame_rgb), return_tensors='pt')
            inputs = {key: value.to(device) for key, value in inputs.items()}
            logits = model(**inputs).logits
            logits = torch.nn.functional.interpolate(
                logits, size=frame.shape[:2], mode='bilinear', align_corners=False)
            class_mask = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

            road_raw = binary_mask(class_mask, label2id, 'road')
            sidewalk = binary_mask(class_mask, label2id, 'sidewalk')
            road, lane_hint = refine_masks(frame, road_raw, sidewalk)
            overlay = make_overlay(frame, class_mask, label2id, road, lane_hint)
            igvc_white, igvc_lane_bev, corridor = extract_igvc_lane_features(
                frame, road, lane_hint, grid_width_cells, grid_height_cells)
            keepout, drivable = project_nav2(
                road, lane_hint, corridor, grid_width_cells, grid_height_cells)

            stem = image_path.stem
            cv2.imwrite(str(output_dir / f'{stem}_overlay.jpg'), overlay)
            cv2.imwrite(str(output_dir / f'{stem}_road_refined.png'), road)
            cv2.imwrite(str(output_dir / f'{stem}_lane_hint.png'), lane_hint)
            cv2.imwrite(str(output_dir / f'{stem}_igvc_white.png'), igvc_white)
            cv2.imwrite(str(output_dir / f'{stem}_igvc_lane_bev.png'), igvc_lane_bev)
            cv2.imwrite(str(output_dir / f'{stem}_igvc_corridor.png'), corridor)
            cv2.imwrite(str(output_dir / f'{stem}_nav2_keepout.png'), keepout)
            cv2.imwrite(str(output_dir / f'{stem}_nav2_drivable.png'), drivable)
            rows.append((frame, overlay, road, lane_hint, igvc_lane_bev, corridor, keepout))

            unique, counts = np.unique(class_mask, return_counts=True)
            class_counts = {
                id2label.get(int(class_id), str(class_id)): int(count)
                for class_id, count in zip(unique.tolist(), counts.tolist())
            }
            records.append({
                'image': str(image_path),
                'road_pixels_raw': int(np.count_nonzero(road_raw)),
                'road_pixels_refined': int(np.count_nonzero(road)),
                'lane_hint_pixels': int(np.count_nonzero(lane_hint)),
                'igvc_white_pixels': int(np.count_nonzero(igvc_white)),
                'igvc_lane_bev_pixels': int(np.count_nonzero(igvc_lane_bev)),
                'igvc_corridor_cells': int(np.count_nonzero(corridor)),
                'nav2_keepout_cells': int(np.count_nonzero(keepout)),
                'top_classes': sorted(
                    class_counts.items(), key=lambda item: item[1], reverse=True)[:8],
            })

    make_contact_sheet(rows, output_dir / 'segformer_nav2_contact_sheet.jpg')
    (output_dir / 'segformer_nav2_summary.json').write_text(
        json.dumps({
            'model_id': args.model_id,
            'input_dir': str(Path(args.input_dir).resolve()),
            'grid_resolution': args.grid_resolution,
            'grid_width_m': args.grid_width_m,
            'grid_length_m': args.grid_length_m,
            'image_count': len(records),
            'records': records,
        }, indent=2)
    )
    print(json.dumps({
        'output_dir': str(output_dir),
        'image_count': len(records),
    }, indent=2))


if __name__ == '__main__':
    main()
