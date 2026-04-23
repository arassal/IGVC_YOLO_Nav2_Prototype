import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate traffic cone detections.')
    parser.add_argument(
        '--model',
        default='/home/alexander/Desktop/HERE_Object_Anomaly/models/roboflow_logistics_yolov8.pt')
    parser.add_argument(
        '--dataset',
        default='/home/alexander/Desktop/HERE_Object_Anomaly/cone_test/cone_dataset')
    parser.add_argument(
        '--output-dir',
        default='/home/alexander/Desktop/HERE_Object_Anomaly/cone_test/proof')
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--iou', type=float, default=0.50)
    parser.add_argument('--device', default='cpu')
    return parser.parse_args()


def load_boxes(xml_path):
    root = ET.parse(xml_path).getroot()
    boxes = []
    for obj in root.findall('object'):
        name = obj.findtext('name', '').lower()
        if name not in {'trafficcone', 'traffic cone', 'cone'}:
            continue
        b = obj.find('bndbox')
        boxes.append([
            float(b.findtext('xmin')),
            float(b.findtext('ymin')),
            float(b.findtext('xmax')),
            float(b.findtext('ymax')),
        ])
    return boxes


def iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    denom = area_a + area_b - inter
    return inter / denom if denom else 0.0


def match_predictions(gt_boxes, pred_boxes, threshold):
    matched_gt = set()
    true_pos = 0
    for pred in sorted(pred_boxes, key=lambda x: x['confidence'], reverse=True):
        best_idx = None
        best_iou = 0.0
        for idx, gt in enumerate(gt_boxes):
            if idx in matched_gt:
                continue
            score = iou(gt, pred['xyxy'])
            if score > best_iou:
                best_iou = score
                best_idx = idx
        if best_idx is not None and best_iou >= threshold:
            matched_gt.add(best_idx)
            true_pos += 1
    false_pos = len(pred_boxes) - true_pos
    false_neg = len(gt_boxes) - true_pos
    return true_pos, false_pos, false_neg


def draw_boxes(image, gt_boxes, pred_boxes):
    for box in gt_boxes:
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(image, (x1, y1), (x2, y2), (30, 200, 30), 2)
        cv2.putText(image, 'gt', (x1, max(18, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 200, 30), 2)
    for pred in pred_boxes:
        x1, y1, x2, y2 = [int(v) for v in pred['xyxy']]
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 140, 255), 2)
        cv2.putText(image, f"cone {pred['confidence']:.2f}", (x1, max(36, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2)


def make_contact_sheet(images, output_path):
    thumbs = [cv2.resize(img, (320, 240), interpolation=cv2.INTER_AREA) for img in images[:8]]
    if not thumbs:
        return
    rows = []
    for idx in range(0, len(thumbs), 4):
        row = thumbs[idx:idx + 4]
        while len(row) < 4:
            row.append(np.zeros_like(thumbs[0]))
        rows.append(np.hstack(row))
    cv2.imwrite(str(output_path), np.vstack(rows))


def main():
    args = parse_args()
    dataset = Path(args.dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    totals = {'tp': 0, 'fp': 0, 'fn': 0, 'gt': 0, 'pred': 0, 'images': 0}
    image_results = []
    annotated = []

    for xml_path in sorted((dataset / 'annotations').glob('*.xml'),
                           key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem):
        image_path = dataset / f'{xml_path.stem}.jpg'
        if not image_path.exists():
            continue
        gt_boxes = load_boxes(xml_path)
        result = model.predict(str(image_path), conf=args.conf, imgsz=640,
                               verbose=False, device=args.device)[0]
        pred_boxes = []
        for box in result.boxes:
            class_name = result.names[int(box.cls[0])]
            if class_name != 'traffic cone':
                continue
            pred_boxes.append({
                'confidence': float(box.conf[0]),
                'xyxy': [float(v) for v in box.xyxy[0]],
            })
        tp, fp, fn = match_predictions(gt_boxes, pred_boxes, args.iou)
        totals['tp'] += tp
        totals['fp'] += fp
        totals['fn'] += fn
        totals['gt'] += len(gt_boxes)
        totals['pred'] += len(pred_boxes)
        totals['images'] += 1
        image_results.append({
            'image': image_path.name,
            'ground_truth': len(gt_boxes),
            'predicted': len(pred_boxes),
            'tp': tp,
            'fp': fp,
            'fn': fn,
        })

        image = cv2.imread(str(image_path))
        draw_boxes(image, gt_boxes, pred_boxes)
        if pred_boxes or len(annotated) < 8:
            annotated.append(image)

    precision = totals['tp'] / (totals['tp'] + totals['fp']) if totals['tp'] + totals['fp'] else 0.0
    recall = totals['tp'] / (totals['tp'] + totals['fn']) if totals['tp'] + totals['fn'] else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    summary = {
        'model': args.model,
        'dataset': str(dataset),
        'confidence_threshold': args.conf,
        'iou_threshold': args.iou,
        'totals': totals,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'images': image_results,
    }
    (output_dir / 'traffic_cone_eval.json').write_text(json.dumps(summary, indent=2))
    make_contact_sheet(annotated, output_dir / 'traffic_cone_eval_contact_sheet.jpg')
    print(json.dumps({
        'images': totals['images'],
        'ground_truth_cones': totals['gt'],
        'predicted_cones': totals['pred'],
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4),
    }, indent=2))


if __name__ == '__main__':
    main()
