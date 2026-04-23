import argparse
import base64
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np
import torch
from PIL import Image as PilImage
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor


DEFAULT_MODEL_ID = 'nvidia/segformer-b0-finetuned-cityscapes-512-1024'


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>IGVC Lane Viewer</title>
  <style>
    :root {
      color-scheme: light;
      font-family: Arial, Helvetica, sans-serif;
      background: #f6f7f8;
      color: #161a1d;
    }
    body {
      margin: 0;
    }
    header {
      padding: 18px 24px;
      background: #ffffff;
      border-bottom: 1px solid #d6d9dc;
    }
    h1 {
      margin: 0 0 6px;
      font-size: 24px;
      line-height: 1.2;
    }
    p {
      margin: 0;
      color: #4a535b;
      line-height: 1.45;
    }
    main {
      padding: 18px 24px 28px;
    }
    .toolbar {
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
      margin-bottom: 16px;
    }
    button, select {
      border: 1px solid #aeb6bf;
      border-radius: 6px;
      background: #ffffff;
      color: #161a1d;
      padding: 9px 12px;
      font-size: 15px;
    }
    button {
      cursor: pointer;
    }
    button:disabled {
      opacity: 0.55;
      cursor: default;
    }
    .status {
      min-height: 22px;
      color: #4a535b;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
    }
    figure {
      margin: 0;
      background: #ffffff;
      border: 1px solid #d6d9dc;
      border-radius: 6px;
      overflow: hidden;
    }
    figcaption {
      padding: 10px 12px;
      font-weight: 700;
      border-bottom: 1px solid #e4e6e8;
    }
    img {
      display: block;
      width: 100%;
      height: auto;
      background: #111;
    }
    pre {
      white-space: pre-wrap;
      background: #ffffff;
      border: 1px solid #d6d9dc;
      border-radius: 6px;
      padding: 12px;
      margin: 14px 0 0;
      color: #20262b;
    }
    @media (max-width: 1200px) {
      .grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }
    }
    @media (max-width: 900px) {
      .grid {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <header>
    <h1>IGVC Lane Viewer</h1>
    <p>Raw image, SegFormer road understanding, IGVC white-line extraction, BEV lane products, and Nav2 keepout preview. No YOLO is used in this viewer.</p>
  </header>
  <main>
    <div class="toolbar">
      <button id="prev">Previous</button>
      <button id="next">Next</button>
      <select id="imageSelect"></select>
      <button id="run">Run Pipeline</button>
      <span class="status" id="status">Loading image list...</span>
    </div>
    <div class="grid">
      <figure>
        <figcaption>RAW INPUT - unmodified</figcaption>
        <img id="raw" alt="Raw input">
      </figure>
      <figure>
        <figcaption>SegFormer + HSV overlay</figcaption>
        <img id="overlay" alt="SegFormer overlay">
      </figure>
      <figure>
        <figcaption>Road mask raw</figcaption>
        <img id="roadRaw" alt="Raw road mask">
      </figure>
      <figure>
        <figcaption>Road mask refined</figcaption>
        <img id="road" alt="Road mask">
      </figure>
      <figure>
        <figcaption>Sidewalk mask</figcaption>
        <img id="sidewalk" alt="Sidewalk mask">
      </figure>
      <figure>
        <figcaption>Lane hint mask</figcaption>
        <img id="laneHint" alt="Lane hint mask">
      </figure>
      <figure>
        <figcaption>IGVC white-line mask</figcaption>
        <img id="igvcWhite" alt="IGVC white-line mask">
      </figure>
      <figure>
        <figcaption>IGVC lane BEV</figcaption>
        <img id="igvcBev" alt="IGVC lane BEV">
      </figure>
      <figure>
        <figcaption>IGVC lane corridor</figcaption>
        <img id="igvcCorridor" alt="IGVC lane corridor">
      </figure>
      <figure>
        <figcaption>Nav2 keepout preview</figcaption>
        <img id="nav2Keepout" alt="Nav2 keepout preview">
      </figure>
    </div>
    <pre id="metadata">Select an image and run the pipeline. Use Left and Right arrow keys to move between frames.</pre>
  </main>
  <script>
    let images = [];
    let index = 0;
    const select = document.getElementById('imageSelect');
    const statusEl = document.getElementById('status');
    const rawEl = document.getElementById('raw');
    const overlayEl = document.getElementById('overlay');
    const roadRawEl = document.getElementById('roadRaw');
    const roadEl = document.getElementById('road');
    const sidewalkEl = document.getElementById('sidewalk');
    const laneHintEl = document.getElementById('laneHint');
    const igvcWhiteEl = document.getElementById('igvcWhite');
    const igvcBevEl = document.getElementById('igvcBev');
    const igvcCorridorEl = document.getElementById('igvcCorridor');
    const nav2KeepoutEl = document.getElementById('nav2Keepout');
    const metadataEl = document.getElementById('metadata');
    let latestRunToken = 0;

    function setStatus(text) {
      statusEl.textContent = text;
    }

    function current() {
      return images[index];
    }

    function refreshRaw() {
      if (!current()) return;
      select.value = String(index);
      rawEl.src = `/image?index=${index}&t=${Date.now()}`;
      overlayEl.removeAttribute('src');
      roadRawEl.removeAttribute('src');
      roadEl.removeAttribute('src');
      sidewalkEl.removeAttribute('src');
      laneHintEl.removeAttribute('src');
      igvcWhiteEl.removeAttribute('src');
      igvcBevEl.removeAttribute('src');
      igvcCorridorEl.removeAttribute('src');
      nav2KeepoutEl.removeAttribute('src');
      metadataEl.textContent = 'Segmentation not run for this selection yet.';
      setStatus(`Selected ${current().name}`);
    }

    async function loadImages() {
      const res = await fetch('/api/images');
      const data = await res.json();
      images = data.images;
      select.innerHTML = '';
      images.forEach((img, i) => {
        const option = document.createElement('option');
        option.value = String(i);
        option.textContent = img.name;
        select.appendChild(option);
      });
      if (images.length === 0) {
        setStatus('No images found.');
        return;
      }
      refreshRaw();
    }

    async function runSegmentation() {
      if (!current()) return;
      const runToken = ++latestRunToken;
      setStatus('Running SegFormer...');
      document.getElementById('run').disabled = true;
      try {
        const res = await fetch(`/api/segment?index=${index}`);
        const data = await res.json();
        if (!res.ok) {
          throw new Error(data.error || 'Segmentation failed');
        }
        if (runToken !== latestRunToken) {
          return;
        }
        overlayEl.src = data.overlay;
        roadRawEl.src = data.road_mask_raw;
        roadEl.src = data.road_mask;
        sidewalkEl.src = data.sidewalk_mask;
        laneHintEl.src = data.lane_hint_mask;
        igvcWhiteEl.src = data.igvc_white_mask;
        igvcBevEl.src = data.igvc_lane_bev;
        igvcCorridorEl.src = data.igvc_lane_corridor_mask;
        nav2KeepoutEl.src = data.nav2_keepout_mask;
        metadataEl.textContent = JSON.stringify(data.metadata, null, 2);
        setStatus(`Done in ${data.metadata.timing_ms.toFixed(1)} ms`);
      } catch (err) {
        if (runToken === latestRunToken) {
          setStatus(err.message);
        }
      } finally {
        if (runToken === latestRunToken) {
          document.getElementById('run').disabled = false;
        }
      }
    }

    function move(delta) {
      if (!images.length) return;
      index = (index + delta + images.length) % images.length;
      refreshRaw();
      runSegmentation();
    }

    document.getElementById('prev').addEventListener('click', () => {
      move(-1);
    });
    document.getElementById('next').addEventListener('click', () => {
      move(1);
    });
    select.addEventListener('change', () => {
      index = Number(select.value);
      refreshRaw();
      runSegmentation();
    });
    document.getElementById('run').addEventListener('click', runSegmentation);
    document.addEventListener('keydown', (event) => {
      if (event.target && ['SELECT', 'INPUT', 'TEXTAREA'].includes(event.target.tagName)) {
        return;
      }
      if (event.key === 'ArrowLeft') {
        event.preventDefault();
        move(-1);
      } else if (event.key === 'ArrowRight') {
        event.preventDefault();
        move(1);
      }
    });

    loadImages().then(runSegmentation).catch(err => setStatus(err.message));
  </script>
</body>
</html>
"""


def parse_args():
    parser = argparse.ArgumentParser(description='Local SegFormer image viewer.')
    parser.add_argument('--image-dir', default='/home/alexander/Desktop/img')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=7861)
    parser.add_argument('--model-id', default=DEFAULT_MODEL_ID)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--enable-hsv-refinement', action='store_true', default=True)
    parser.add_argument('--disable-hsv-refinement', action='store_false', dest='enable_hsv_refinement')
    parser.add_argument('--nav2-grid-resolution', type=float, default=0.05)
    parser.add_argument('--nav2-x-range', nargs=2, type=float, default=[0.0, 15.0])
    parser.add_argument('--nav2-y-range', nargs=2, type=float, default=[-10.0, 10.0])
    parser.add_argument('--nav2-src-bottom-y', type=float, default=0.98)
    parser.add_argument('--nav2-src-top-y', type=float, default=0.62)
    parser.add_argument('--nav2-src-bottom-left-x', type=float, default=0.05)
    parser.add_argument('--nav2-src-bottom-right-x', type=float, default=0.95)
    parser.add_argument('--nav2-src-top-left-x', type=float, default=0.35)
    parser.add_argument('--nav2-src-top-right-x', type=float, default=0.65)
    parser.add_argument('--lane-detect-on-threshold', type=float, default=0.30)
    parser.add_argument('--lane-detect-off-threshold', type=float, default=0.18)
    parser.add_argument('--lane-corridor-target-cells', type=float, default=220.0)
    parser.add_argument('--lane-bev-target-pixels', type=float, default=140.0)
    return parser.parse_args()


def collect_images(image_dir):
    root = Path(image_dir)
    images = []
    for pattern in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
        images.extend(root.glob(pattern))
    return sorted(images)


def encode_image(image, extension='.jpg'):
    ok, buffer = cv2.imencode(extension, image)
    if not ok:
        raise RuntimeError('Failed to encode image')
    encoded = base64.b64encode(buffer).decode('ascii')
    mime = 'image/png' if extension == '.png' else 'image/jpeg'
    return f'data:{mime};base64,{encoded}'


class SegFormerRunner:
    def __init__(self, args):
        self.model_id = args.model_id
        self.device = torch.device(args.device if args.device != 'cpu' else 'cpu')
        self.enable_hsv_refinement = args.enable_hsv_refinement
        self.nav2_grid_resolution = float(args.nav2_grid_resolution)
        self.nav2_x_range = [float(v) for v in args.nav2_x_range]
        self.nav2_y_range = [float(v) for v in args.nav2_y_range]
        self.nav2_src_bottom_y = float(args.nav2_src_bottom_y)
        self.nav2_src_top_y = float(args.nav2_src_top_y)
        self.nav2_src_bottom_left_x = float(args.nav2_src_bottom_left_x)
        self.nav2_src_bottom_right_x = float(args.nav2_src_bottom_right_x)
        self.nav2_src_top_left_x = float(args.nav2_src_top_left_x)
        self.nav2_src_top_right_x = float(args.nav2_src_top_right_x)
        self.lane_detect_on_threshold = float(args.lane_detect_on_threshold)
        self.lane_detect_off_threshold = float(args.lane_detect_off_threshold)
        self.lane_corridor_target_cells = float(args.lane_corridor_target_cells)
        self.lane_bev_target_pixels = float(args.lane_bev_target_pixels)
        self.nav2_x_min, self.nav2_x_max = self.nav2_x_range
        self.nav2_y_min, self.nav2_y_max = self.nav2_y_range
        self.nav2_grid_width_m = self.nav2_y_max - self.nav2_y_min
        self.nav2_grid_length_m = self.nav2_x_max - self.nav2_x_min
        self.nav2_grid_width_cells = max(
            1, int(round(self.nav2_grid_width_m / self.nav2_grid_resolution)))
        self.nav2_grid_height_cells = max(
            1, int(round(self.nav2_grid_length_m / self.nav2_grid_resolution)))
        self.processor = SegformerImageProcessor.from_pretrained(self.model_id)
        self.model = SegformerForSemanticSegmentation.from_pretrained(self.model_id).to(self.device)
        self.model.eval()
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        self.label2id = {label.lower(): idx for idx, label in self.id2label.items()}
        self.lock = threading.Lock()

    def segment(self, image_path):
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise RuntimeError(f'Failed to read {image_path}')
        start = time.perf_counter()
        with self.lock, torch.no_grad():
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            inputs = self.processor(images=PilImage.fromarray(frame_rgb), return_tensors='pt')
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            logits = self.model(**inputs).logits
            logits = torch.nn.functional.interpolate(
                logits,
                size=frame.shape[:2],
                mode='bilinear',
                align_corners=False,
            )
            class_mask = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        road_raw = self._binary_mask(class_mask, 'road')
        sidewalk = self._binary_mask(class_mask, 'sidewalk')
        road, lane_hint = self._refine_masks(frame, road_raw, sidewalk)
        igvc_white, igvc_lane_bev, igvc_lane_corridor = self._extract_igvc_lane_features(
            frame,
            road,
            lane_hint,
            frame.shape[1],
            frame.shape[0],
        )
        lane_confidence, lane_detected = self._estimate_lane_confidence(
            igvc_lane_bev, igvc_lane_corridor)
        igvc_lane_corridor_image = self._project_bev_to_image(
            igvc_lane_corridor, frame.shape[1], frame.shape[0])
        overlay = self._overlay(frame, class_mask, road, lane_hint, igvc_lane_corridor_image)
        nav2_keepout_mask, nav2_drivable_mask = self._project_nav2_grids(
            road,
            lane_hint,
            igvc_lane_corridor,
            lane_detected,
            frame.shape[1],
            frame.shape[0],
        )
        unique, counts = np.unique(class_mask, return_counts=True)
        class_counts = {
            self.id2label.get(int(class_id), str(class_id)): int(count)
            for class_id, count in zip(unique.tolist(), counts.tolist())
        }
        top_classes = sorted(class_counts.items(), key=lambda item: item[1], reverse=True)[:8]
        return {
            'overlay': encode_image(overlay),
            'road_mask_raw': encode_image(road_raw, '.png'),
            'road_mask': encode_image(road, '.png'),
            'sidewalk_mask': encode_image(sidewalk, '.png'),
            'lane_hint_mask': encode_image(lane_hint, '.png'),
            'igvc_white_mask': encode_image(igvc_white, '.png'),
            'igvc_lane_bev': encode_image(igvc_lane_bev, '.png'),
            'igvc_lane_corridor_mask': encode_image(igvc_lane_corridor, '.png'),
            'nav2_keepout_mask': encode_image(nav2_keepout_mask, '.png'),
            'metadata': {
                'image': str(image_path),
                'model_id': self.model_id,
                'hsv_refinement_enabled': self.enable_hsv_refinement,
                'road_pixels_raw': int(np.count_nonzero(road_raw)),
                'road_pixels': int(np.count_nonzero(road)),
                'sidewalk_pixels': int(np.count_nonzero(sidewalk)),
                'lane_hint_pixels': int(np.count_nonzero(lane_hint)),
                'igvc_white_pixels': int(np.count_nonzero(igvc_white)),
                'igvc_lane_bev_pixels': int(np.count_nonzero(igvc_lane_bev)),
                'igvc_lane_corridor_pixels': int(np.count_nonzero(igvc_lane_corridor)),
                'nav2_keepout_cells': int(np.count_nonzero(nav2_keepout_mask)),
                'lane_confidence': lane_confidence,
                'lane_detected': lane_detected,
                'planner_mode_hint': 'lane_following' if lane_detected else 'obstacle_avoidance',
                'top_classes': top_classes,
                'timing_ms': elapsed_ms,
                'yolo_used': False,
            },
        }

    def _binary_mask(self, class_mask, label):
        class_id = self.label2id.get(label)
        if class_id is None:
            return np.zeros(class_mask.shape, dtype=np.uint8)
        return (class_mask == class_id).astype(np.uint8) * 255

    def _refine_masks(self, frame, road_raw, sidewalk):
        if not self.enable_hsv_refinement:
            return road_raw, np.zeros_like(road_raw)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        height, width = road_raw.shape
        roi = np.zeros((height, width), dtype=np.uint8)
        roi[int(height * 0.35):, :] = 255

        asphalt = cv2.inRange(hsv, (0, 0, 35), (179, 80, 185))
        white = cv2.inRange(hsv, (0, 0, 180), (179, 55, 255))
        yellow = cv2.inRange(hsv, (12, 55, 110), (42, 255, 255))
        lane_hint = cv2.bitwise_or(white, yellow)

        kernel_large = np.ones((21, 21), np.uint8)
        kernel_small = np.ones((5, 5), np.uint8)
        road_neighborhood = cv2.dilate(road_raw, kernel_large, iterations=1)
        asphalt_support = cv2.bitwise_and(asphalt, road_neighborhood)
        asphalt_support = cv2.bitwise_and(asphalt_support, roi)
        asphalt_support = cv2.bitwise_and(asphalt_support, cv2.bitwise_not(sidewalk))

        road = cv2.bitwise_or(road_raw, asphalt_support)
        road = cv2.morphologyEx(road, cv2.MORPH_CLOSE, kernel_large)
        road = cv2.morphologyEx(road, cv2.MORPH_OPEN, kernel_small)

        lane_hint = cv2.bitwise_and(lane_hint, roi)
        lane_hint = cv2.bitwise_and(lane_hint, cv2.dilate(road, kernel_large, iterations=1))
        lane_hint = cv2.morphologyEx(lane_hint, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        lane_hint = cv2.dilate(lane_hint, kernel_small, iterations=1)
        return road, lane_hint

    def _overlay(self, frame, class_mask, road_mask, lane_hint, igvc_lane_corridor_image):
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
            class_id = self.label2id.get(label)
            if class_id is not None:
                color_mask[class_mask == class_id] = color
        color_mask[road_mask > 0] = (80, 80, 80)
        color_mask[lane_hint > 0] = (0, 255, 255)
        color_mask[igvc_lane_corridor_image > 0] = (255, 255, 0)
        active = np.any(color_mask != 0, axis=2)
        overlay = frame.copy()
        blended = cv2.addWeighted(frame, 0.55, color_mask, 0.45, 0)
        overlay[active] = blended[active]
        return overlay

    def _extract_igvc_lane_features(self, frame, road_mask, lane_hint_mask, width, height):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        roi = np.zeros((height, width), dtype=np.uint8)
        roi[int(height * 0.45):, :] = 255
        white_mask = cv2.inRange(hsv, (0, 0, 165), (179, 70, 255))
        white_mask = cv2.bitwise_and(white_mask, roi)
        road_support = cv2.dilate(road_mask, np.ones((25, 25), np.uint8), iterations=1)
        white_mask = cv2.bitwise_and(white_mask, road_support)
        white_mask = cv2.bitwise_or(white_mask, lane_hint_mask)
        white_mask = cv2.morphologyEx(
            white_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        white_mask = cv2.dilate(white_mask, np.ones((3, 3), np.uint8), iterations=1)

        transform = self._nav2_perspective_transform(width, height)
        white_bev = cv2.warpPerspective(
            white_mask,
            transform,
            (self.nav2_grid_width_cells, self.nav2_grid_height_cells),
            flags=cv2.INTER_NEAREST,
        )
        white_bev = cv2.morphologyEx(
            white_bev, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        white_bev = cv2.dilate(white_bev, np.ones((3, 3), np.uint8), iterations=1)
        left_boundary, right_boundary = self._select_lane_boundaries(white_bev)
        corridor = self._build_lane_corridor(left_boundary, right_boundary)
        if np.count_nonzero(corridor) == 0:
            corridor = white_bev.copy()
        return white_mask, white_bev, corridor

    def _nav2_perspective_transform(self, width, height):
        src = np.float32([
            [width * self.nav2_src_bottom_left_x, height * self.nav2_src_bottom_y],
            [width * self.nav2_src_bottom_right_x, height * self.nav2_src_bottom_y],
            [width * self.nav2_src_top_right_x, height * self.nav2_src_top_y],
            [width * self.nav2_src_top_left_x, height * self.nav2_src_top_y],
        ])
        dst = np.float32([
            [0, self.nav2_grid_height_cells - 1],
            [self.nav2_grid_width_cells - 1, self.nav2_grid_height_cells - 1],
            [self.nav2_grid_width_cells - 1, 0],
            [0, 0],
        ])
        return cv2.getPerspectiveTransform(src, dst)

    def _project_bev_to_image(self, bev_mask, width, height):
        transform = self._nav2_perspective_transform(width, height)
        inverse = np.linalg.inv(transform)
        return cv2.warpPerspective(
            bev_mask,
            inverse,
            (width, height),
            flags=cv2.INTER_NEAREST,
        )

    def _select_lane_boundaries(self, white_bev):
        left_half = white_bev[:, : self.nav2_grid_width_cells // 2]
        right_half = white_bev[:, self.nav2_grid_width_cells // 2 :]
        left_boundary = self._largest_lane_component(left_half)
        right_boundary = self._largest_lane_component(right_half)
        right_full = np.zeros_like(white_bev)
        left_full = np.zeros_like(white_bev)
        left_full[:, : self.nav2_grid_width_cells // 2] = left_boundary
        right_full[:, self.nav2_grid_width_cells // 2 :] = right_boundary
        return left_full, right_full

    def _largest_lane_component(self, mask):
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

    def _build_lane_corridor(self, left_boundary, right_boundary):
        corridor = np.zeros_like(left_boundary)
        for row in range(self.nav2_grid_height_cells):
            left_cols = np.flatnonzero(left_boundary[row] > 0)
            right_cols = np.flatnonzero(right_boundary[row] > 0)
            if left_cols.size == 0 or right_cols.size == 0:
                continue
            left_x = int(np.max(left_cols))
            right_x = int(np.min(right_cols))
            if right_x <= left_x:
                continue
            corridor[row, left_x:right_x + 1] = 255
        corridor = cv2.morphologyEx(
            corridor, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        return corridor

    def _estimate_lane_confidence(self, igvc_lane_bev, igvc_lane_corridor):
        corridor_cells = float(np.count_nonzero(igvc_lane_corridor))
        bev_pixels = float(np.count_nonzero(igvc_lane_bev))
        corridor_score = min(1.0, corridor_cells / max(1.0, self.lane_corridor_target_cells))
        bev_score = min(1.0, bev_pixels / max(1.0, self.lane_bev_target_pixels))
        confidence = (0.7 * corridor_score) + (0.3 * bev_score)
        lane_detected = confidence >= self.lane_detect_on_threshold
        if confidence < self.lane_detect_off_threshold:
            lane_detected = False
        return confidence, lane_detected

    def _project_nav2_grids(
        self,
        road_mask,
        lane_hint_mask,
        igvc_lane_corridor,
        lane_detected,
        width,
        height,
    ):
        transform = self._nav2_perspective_transform(width, height)
        road_bev = cv2.warpPerspective(
            road_mask,
            transform,
            (self.nav2_grid_width_cells, self.nav2_grid_height_cells),
            flags=cv2.INTER_NEAREST,
        )
        lane_bev = cv2.warpPerspective(
            lane_hint_mask,
            transform,
            (self.nav2_grid_width_cells, self.nav2_grid_height_cells),
            flags=cv2.INTER_NEAREST,
        )
        road_bev = cv2.morphologyEx(
            road_bev,
            cv2.MORPH_CLOSE,
            np.ones((5, 5), np.uint8),
        )
        drivable = np.where(road_bev > 0, 255, 0).astype(np.uint8)
        if lane_detected and np.count_nonzero(igvc_lane_corridor) > 0:
            drivable = cv2.bitwise_and(
                drivable,
                cv2.dilate(igvc_lane_corridor, np.ones((9, 9), np.uint8), iterations=1),
            )
        drivable = cv2.bitwise_or(drivable, lane_bev)
        drivable = cv2.morphologyEx(
            drivable, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        keepout = np.where(drivable > 0, 0, 255).astype(np.uint8)
        return keepout, drivable


def make_handler(images, runner):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, status, content_type, body):
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, status, payload):
            self._send(status, 'application/json', json.dumps(payload).encode('utf-8'))

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == '/':
                self._send(200, 'text/html; charset=utf-8', HTML.encode('utf-8'))
                return
            if parsed.path == '/api/images':
                payload = {
                    'images': [
                        {'index': idx, 'name': path.name, 'path': str(path)}
                        for idx, path in enumerate(images)
                    ]
                }
                self._send_json(200, payload)
                return
            if parsed.path == '/image':
                try:
                    index = int(parse_qs(parsed.query).get('index', ['0'])[0])
                    path = images[index]
                    data = path.read_bytes()
                    content_type = 'image/png' if path.suffix.lower() == '.png' else 'image/jpeg'
                    self._send(200, content_type, data)
                except Exception as exc:
                    self._send_json(400, {'error': str(exc)})
                return
            if parsed.path == '/api/segment':
                try:
                    index = int(parse_qs(parsed.query).get('index', ['0'])[0])
                    self._send_json(200, runner.segment(images[index]))
                except Exception as exc:
                    self._send_json(500, {'error': str(exc)})
                return
            self._send_json(404, {'error': 'not found'})

        def log_message(self, fmt, *args):
            return

    return Handler


def main():
    args = parse_args()
    images = collect_images(args.image_dir)
    if not images:
        raise RuntimeError(f'No images found in {args.image_dir}')
    print(f'Loading SegFormer model: {args.model_id}')
    runner = SegFormerRunner(args)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(images, runner))
    print(f'Serving {len(images)} images from {args.image_dir}')
    print(f'Open http://{args.host}:{args.port}')
    server.serve_forever()


if __name__ == '__main__':
    main()
