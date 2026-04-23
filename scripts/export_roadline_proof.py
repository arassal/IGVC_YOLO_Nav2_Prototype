import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export YOLOPv2 drivable-area and lane-line proof images.')
    parser.add_argument(
        '--project-root',
        default='/home/alexander/Desktop/seg',
        help='YOLOPv2 project root containing utils/ and data/.')
    parser.add_argument(
        '--weights',
        default='/home/alexander/Desktop/seg/data/weights/yolopv2.pt',
        help='TorchScript YOLOPv2 weights.')
    parser.add_argument(
        '--source-dir',
        default='/home/alexander/Desktop/seg/data/demo',
        help='Folder of input images.')
    parser.add_argument(
        '--output-dir',
        default='/home/alexander/Desktop/roadline_demo_proof',
        help='Folder where proof images are written.')
    parser.add_argument(
        '--device',
        default='cpu',
        help='Torch device string, for example cpu or 0.')
    parser.add_argument(
        '--limit',
        type=int,
        default=6,
        help='Maximum number of images to export.')
    return parser.parse_args()


def collect_images(source_dir):
    paths = []
    for pattern in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
        paths.extend(Path(source_dir).glob(pattern))
    return sorted(paths)


def resize_mask(mask, size):
    return cv2.resize(
        (mask * 255).astype(np.uint8),
        size,
        interpolation=cv2.INTER_NEAREST,
    )


def make_contact_sheet(rows, output_path):
    if not rows:
        return
    thumbs = []
    for original, overlay, drivable, lane in rows:
        h, w = 180, 320
        original = cv2.resize(original, (w, h))
        overlay = cv2.resize(overlay, (w, h))
        drivable_bgr = cv2.cvtColor(cv2.resize(drivable, (w, h)), cv2.COLOR_GRAY2BGR)
        lane_bgr = cv2.cvtColor(cv2.resize(lane, (w, h)), cv2.COLOR_GRAY2BGR)
        row = np.hstack([original, overlay, drivable_bgr, lane_bgr])
        thumbs.append(row)
    sheet = np.vstack(thumbs)
    cv2.imwrite(str(output_path), sheet)


def main():
    args = parse_args()
    project_root = Path(args.project_root)
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from utils.utils import (
        driving_area_mask,
        lane_line_mask,
        letterbox,
        select_device,
        show_seg_result,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = collect_images(args.source_dir)[:args.limit]
    if not image_paths:
        raise RuntimeError(f'No images found in {args.source_dir}')

    device = select_device(args.device)
    model = torch.jit.load(args.weights, map_location=device).to(device)
    half = device.type != 'cpu'
    if half:
        model.half()
    model.eval()

    contact_rows = []
    with torch.no_grad():
        for image_path in image_paths:
            frame = cv2.imread(str(image_path))
            if frame is None:
                print(f'skip unreadable image: {image_path}')
                continue

            im0 = cv2.resize(frame.copy(), (1280, 720), interpolation=cv2.INTER_LINEAR)
            img = letterbox(im0, 640, stride=32)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)
            img = np.ascontiguousarray(img)

            tensor = torch.from_numpy(img).to(device)
            tensor = tensor.half() if half else tensor.float()
            tensor /= 255.0
            if tensor.ndimension() == 3:
                tensor = tensor.unsqueeze(0)

            _pred, seg, ll = model(tensor)
            da_mask = driving_area_mask(seg)
            ll_mask = lane_line_mask(ll)

            drivable = resize_mask(da_mask, (im0.shape[1], im0.shape[0]))
            lane = resize_mask(ll_mask, (im0.shape[1], im0.shape[0]))

            overlay = im0.copy()
            show_seg_result(overlay, (da_mask, ll_mask), is_demo=True)

            stem = image_path.stem
            cv2.imwrite(str(output_dir / f'{stem}_input.jpg'), im0)
            cv2.imwrite(str(output_dir / f'{stem}_overlay.jpg'), overlay)
            cv2.imwrite(str(output_dir / f'{stem}_drivable_mask.png'), drivable)
            cv2.imwrite(str(output_dir / f'{stem}_lane_mask.png'), lane)
            contact_rows.append((im0, overlay, drivable, lane))

            lane_pixels = int((lane > 0).sum())
            drivable_pixels = int((drivable > 0).sum())
            print(
                f'{image_path.name}: lane_pixels={lane_pixels} '
                f'drivable_pixels={drivable_pixels}')

    make_contact_sheet(contact_rows, output_dir / 'contact_sheet.jpg')
    print(f'Wrote proof outputs to {output_dir}')


if __name__ == '__main__':
    main()
