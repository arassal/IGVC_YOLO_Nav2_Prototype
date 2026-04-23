import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate proof assets for the IGVC BEV lane pipeline.')
    parser.add_argument('--input-dir', default='/home/alexander/Desktop/img')
    parser.add_argument('--output-dir', default='proof/igvc_bev')
    parser.add_argument('--limit', type=int, default=6)
    parser.add_argument('--grid-resolution', type=float, default=0.05)
    parser.add_argument('--x-range', nargs=2, type=float, default=[0.0, 15.0])
    parser.add_argument('--y-range', nargs=2, type=float, default=[-10.0, 10.0])
    return parser.parse_args()


def collect_images(input_dir, limit):
    root = Path(input_dir)
    images = []
    for pattern in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
        images.extend(sorted(root.glob(pattern)))
    return images[:limit]


def perspective_transform(width, height, grid_width_cells, grid_height_cells):
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
    return cv2.getPerspectiveTransform(src, dst)


def largest_lane_component(mask):
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
    return cv2.morphologyEx(corridor, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))


def extract_products(frame, grid_width_cells, grid_height_cells):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    height, width = frame.shape[:2]
    roi = np.zeros((height, width), dtype=np.uint8)
    roi[int(height * 0.45):, :] = 255

    asphalt_mask = cv2.inRange(hsv, (0, 0, 30), (179, 90, 190))
    white_mask = cv2.inRange(hsv, (0, 0, 165), (179, 70, 255))
    yellow_mask = cv2.inRange(hsv, (12, 50, 105), (42, 255, 255))

    asphalt_mask = cv2.bitwise_and(asphalt_mask, roi)
    white_mask = cv2.bitwise_and(white_mask, roi)
    yellow_mask = cv2.bitwise_and(yellow_mask, roi)

    kernel_large = np.ones((21, 21), np.uint8)
    kernel_medium = np.ones((7, 7), np.uint8)
    kernel_small = np.ones((3, 3), np.uint8)

    asphalt_mask = cv2.morphologyEx(asphalt_mask, cv2.MORPH_CLOSE, kernel_large)
    asphalt_mask = cv2.morphologyEx(asphalt_mask, cv2.MORPH_OPEN, kernel_medium)

    lane_hint_mask = cv2.bitwise_or(white_mask, yellow_mask)
    lane_hint_mask = cv2.bitwise_and(
        lane_hint_mask, cv2.dilate(asphalt_mask, kernel_large, iterations=1))
    lane_hint_mask = cv2.morphologyEx(lane_hint_mask, cv2.MORPH_OPEN, kernel_small)
    lane_hint_mask = cv2.dilate(lane_hint_mask, kernel_small, iterations=1)

    road_mask = cv2.bitwise_or(asphalt_mask, cv2.dilate(lane_hint_mask, kernel_medium, iterations=1))
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel_large)
    road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, kernel_medium)

    transform = perspective_transform(width, height, grid_width_cells, grid_height_cells)
    lane_bev = cv2.warpPerspective(
        lane_hint_mask,
        transform,
        (grid_width_cells, grid_height_cells),
        flags=cv2.INTER_NEAREST,
    )
    lane_bev = cv2.morphologyEx(lane_bev, cv2.MORPH_OPEN, kernel_small)
    lane_bev = cv2.dilate(lane_bev, kernel_small, iterations=1)
    road_bev = cv2.warpPerspective(
        road_mask,
        transform,
        (grid_width_cells, grid_height_cells),
        flags=cv2.INTER_NEAREST,
    )
    road_bev = cv2.morphologyEx(road_bev, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    half = grid_width_cells // 2
    left_boundary = largest_lane_component(lane_bev[:, :half])
    right_boundary = largest_lane_component(lane_bev[:, half:])
    left_full = np.zeros_like(lane_bev)
    right_full = np.zeros_like(lane_bev)
    left_full[:, :half] = left_boundary
    right_full[:, half:] = right_boundary
    lane_corridor = build_lane_corridor(left_full, right_full)
    if np.count_nonzero(lane_corridor) == 0:
        lane_corridor = lane_bev.copy()

    drivable = np.where(road_bev > 0, 255, 0).astype(np.uint8)
    if np.count_nonzero(lane_corridor) > 0:
        drivable = cv2.bitwise_and(
            drivable,
            cv2.dilate(lane_corridor, np.ones((9, 9), np.uint8), iterations=1),
        )
    drivable = cv2.bitwise_or(drivable, lane_bev)
    drivable = cv2.morphologyEx(drivable, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    keepout = np.where(drivable > 0, 0, 255).astype(np.uint8)

    inverse = np.linalg.inv(transform)
    corridor_image = cv2.warpPerspective(
        lane_corridor, inverse, (width, height), flags=cv2.INTER_NEAREST)
    overlay = frame.copy()
    color_mask = np.zeros_like(frame)
    color_mask[road_mask > 0] = (70, 70, 70)
    color_mask[white_mask > 0] = (255, 255, 255)
    color_mask[yellow_mask > 0] = (0, 215, 255)
    color_mask[corridor_image > 0] = (255, 255, 0)
    active = np.any(color_mask != 0, axis=2)
    overlay[active] = cv2.addWeighted(frame, 0.55, color_mask, 0.45, 0)[active]

    return {
        'asphalt_mask': asphalt_mask,
        'white_mask': white_mask,
        'lane_hint_mask': lane_hint_mask,
        'road_mask': road_mask,
        'road_bev': road_bev,
        'lane_bev': lane_bev,
        'lane_corridor': lane_corridor,
        'keepout': keepout,
        'overlay': overlay,
    }


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
    for row in rows:
        tiles = []
        for image, label in (
            (row['raw'], 'raw input'),
            (cv2.cvtColor(row['asphalt_mask'], cv2.COLOR_GRAY2BGR), 'asphalt mask'),
            (cv2.cvtColor(row['white_mask'], cv2.COLOR_GRAY2BGR), 'white mask'),
            (cv2.cvtColor(row['lane_hint_mask'], cv2.COLOR_GRAY2BGR), 'lane hint'),
            (cv2.cvtColor(row['lane_bev'], cv2.COLOR_GRAY2BGR), 'lane bev'),
            (cv2.cvtColor(row['lane_corridor'], cv2.COLOR_GRAY2BGR), 'lane corridor'),
            (cv2.cvtColor(row['keepout'], cv2.COLOR_GRAY2BGR), 'nav2 keepout'),
            (row['overlay'], 'overlay'),
        ):
            tile = cv2.resize(image, (250, 180), interpolation=cv2.INTER_AREA)
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

    grid_width_cells = max(
        1, int(round((args.y_range[1] - args.y_range[0]) / args.grid_resolution)))
    grid_height_cells = max(
        1, int(round((args.x_range[1] - args.x_range[0]) / args.grid_resolution)))

    summary = []
    rows = []
    for image_path in input_images:
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue
        outputs = extract_products(frame, grid_width_cells, grid_height_cells)
        stem = image_path.stem
        cv2.imwrite(str(output_dir / f'{stem}_overlay.jpg'), outputs['overlay'])
        cv2.imwrite(str(output_dir / f'{stem}_asphalt_mask.png'), outputs['asphalt_mask'])
        cv2.imwrite(str(output_dir / f'{stem}_white_mask.png'), outputs['white_mask'])
        cv2.imwrite(str(output_dir / f'{stem}_lane_hint_mask.png'), outputs['lane_hint_mask'])
        cv2.imwrite(str(output_dir / f'{stem}_lane_bev.png'), outputs['lane_bev'])
        cv2.imwrite(str(output_dir / f'{stem}_lane_corridor.png'), outputs['lane_corridor'])
        cv2.imwrite(str(output_dir / f'{stem}_nav2_keepout.png'), outputs['keepout'])
        rows.append({'raw': frame, **outputs})
        summary.append({
            'image': image_path.name,
            'asphalt_pixels': int(np.count_nonzero(outputs['asphalt_mask'])),
            'white_pixels': int(np.count_nonzero(outputs['white_mask'])),
            'lane_hint_pixels': int(np.count_nonzero(outputs['lane_hint_mask'])),
            'lane_bev_pixels': int(np.count_nonzero(outputs['lane_bev'])),
            'lane_corridor_cells': int(np.count_nonzero(outputs['lane_corridor'])),
            'nav2_keepout_cells': int(np.count_nonzero(outputs['keepout'] == 255)),
        })

    make_contact_sheet(rows, output_dir / 'igvc_bev_contact_sheet.jpg')
    (output_dir / 'igvc_bev_summary.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
