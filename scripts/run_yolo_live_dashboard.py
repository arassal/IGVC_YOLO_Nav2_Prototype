#!/usr/bin/env python3
import argparse
import json
import sys
import threading
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import requests
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_PROJECT_ROOT = Path('/home/alexander/github/av-perception')
DEFAULT_WEIGHTS = DEFAULT_PROJECT_ROOT / 'data/weights/yolopv2.pt'
DEFAULT_OUTPUT_DIR = REPO_ROOT / '.tmp' / 'yolo_live_dashboard'
SNAPSHOT_NAMES = ('left_raw', 'front_raw', 'right_raw')


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>IGVC YOLO Live Dashboard</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; background: #0b0d10; color: #e8edf2; }}
    .wrap {{ max-width: 1700px; margin: 0 auto; padding: 16px; }}
    h1 {{ margin: 0 0 6px; font-size: 28px; }}
    p {{ margin: 0 0 16px; color: #aab4bf; }}
    .grid3 {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin-bottom: 14px; }}
    .grid1 {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 14px; }}
    .panel {{ background: #14181d; border: 1px solid #2d333b; border-radius: 8px; overflow: hidden; }}
    .panel header {{ padding: 10px 12px; background: #1b2128; border-bottom: 1px solid #2d333b; font-size: 14px; font-weight: 600; }}
    .panel img {{ display: block; width: 100%; height: auto; background: #000; }}
    .center {{ grid-column: 2 / span 1; }}
    .status {{ margin-top: 12px; font-size: 13px; color: #99a4af; }}
    .live {{ margin: 0 0 14px; font-size: 13px; color: #c7d0da; }}
    code {{ color: #d7e3f0; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>IGVC YOLOPv2 Live Dashboard</h1>
    <p>Raw camera row, segmented row, then BEV.</p>
    <div class="live" id="live-status">Checking stream state...</div>
    <div class="grid3">
      <section class="panel"><header>Left Raw</header><img data-img="left_raw.jpg" alt="Left raw"></section>
      <section class="panel"><header>Front Raw</header><img data-img="front_raw.jpg" alt="Front raw"></section>
      <section class="panel"><header>Right Raw</header><img data-img="right_raw.jpg" alt="Right raw"></section>
    </div>
    <div class="grid3">
      <section class="panel"><header>Left Segmented</header><img data-img="left_segmented.jpg" alt="Left segmented"></section>
      <section class="panel"><header>Front Segmented</header><img data-img="front_mask.jpg" alt="Front segmented"></section>
      <section class="panel"><header>Right Segmented</header><img data-img="right_segmented.jpg" alt="Right segmented"></section>
    </div>
    <div class="grid1">
      <section class="panel center"><header>BEV</header><img data-img="combined_bev.jpg" alt="BEV"></section>
    </div>
    <div class="status">
      Source: <code>{snapshot_base}</code><br>
      White = lane, gray = drivable area.
    </div>
  </div>
  <script>
    const refreshMs = {refresh_ms};
    const imgs = Array.from(document.querySelectorAll("img[data-img]"));
    const statusEl = document.getElementById("live-status");
    function tick() {{
      const stamp = Date.now();
      for (const img of imgs) {{
        img.src = `${{img.dataset.img}}?t=${{stamp}}`;
      }}
    }}
    async function refreshStatus() {{
      try {{
        const res = await fetch(`status.json?t=${{Date.now()}}`, {{ cache: "no-store" }});
        const data = await res.json();
        const parts = [
          `left: ${{data.left_raw ? "live" : "unavailable"}}`,
          `front: ${{data.front_raw ? "live" : "unavailable"}}`,
          `right: ${{data.right_raw ? "live" : "unavailable"}}`,
          `mask source: front`,
          `bev: combined`
        ];
        statusEl.textContent = parts.join(" | ");
      }} catch (err) {{
        statusEl.textContent = "status unavailable";
      }}
    }}
    tick();
    setInterval(tick, refreshMs);
    refreshStatus();
    setInterval(refreshStatus, 1000);
  </script>
</body>
</html>
"""


class YoloPv2Runner:
    def __init__(self, project_root: Path, weights_path: Path, device_name: str, img_size: int, conf: float, iou: float):
        self.project_root = project_root
        self.weights_path = weights_path
        self.device_name = device_name
        self.img_size = img_size
        self.conf = conf
        self.iou = iou

        if not self.project_root.is_dir():
            raise RuntimeError(f'project_root does not exist: {self.project_root}')
        if not self.weights_path.is_file():
            raise RuntimeError(f'weights do not exist: {self.weights_path}')
        if str(self.project_root) not in sys.path:
            sys.path.insert(0, str(self.project_root))

        from utils.utils import (
            driving_area_mask,
            lane_line_mask,
            letterbox,
            non_max_suppression,
            select_device,
            show_seg_result,
            split_for_trace_model,
        )

        self.driving_area_mask = driving_area_mask
        self.lane_line_mask = lane_line_mask
        self.letterbox = letterbox
        self.non_max_suppression = non_max_suppression
        self.select_device = select_device
        self.show_seg_result = show_seg_result
        self.split_for_trace_model = split_for_trace_model

        self.device = self.select_device(self.device_name)
        self.model = torch.jit.load(str(self.weights_path), map_location=self.device).to(self.device)
        self.half = self.device.type != 'cpu'
        if self.half:
            self.model.half()
        self.model.eval()

    def infer(self, frame: np.ndarray):
        orig_h, orig_w = frame.shape[:2]
        model_frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)
        img = self.letterbox(model_frame, self.img_size, stride=32)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        tensor = torch.from_numpy(img).to(self.device)
        tensor = tensor.half() if self.half else tensor.float()
        tensor /= 255.0
        if tensor.ndimension() == 3:
            tensor = tensor.unsqueeze(0)

        with torch.no_grad():
            pred_raw, seg, ll = self.model(tensor)
        if isinstance(pred_raw, (list, tuple)) and len(pred_raw) == 2:
            pred = self.split_for_trace_model(pred_raw[0], pred_raw[1])
        else:
            pred = self.split_for_trace_model(pred_raw, None)
        _ = self.non_max_suppression(pred, self.conf, self.iou)

        da_mask = self.driving_area_mask(seg).astype(np.uint8)
        ll_mask = self.lane_line_mask(ll).astype(np.uint8)

        overlay = model_frame.copy()
        self.show_seg_result(overlay, (da_mask, ll_mask), is_demo=True)

        if (orig_w, orig_h) != (1280, 720):
            overlay = cv2.resize(overlay, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            da_mask = cv2.resize(da_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            ll_mask = cv2.resize(ll_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        return overlay, da_mask, ll_mask


def fetch_image(session: requests.Session, url: str, timeout: float) -> np.ndarray:
    resp = session.get(url, timeout=timeout, verify=False)
    resp.raise_for_status()
    arr = np.frombuffer(resp.content, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f'Failed to decode image from {url}')
    return image


def make_unavailable_image(width: int, height: int, label: str) -> np.ndarray:
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:] = (18, 22, 28)
    cv2.rectangle(img, (0, 0), (width - 1, height - 1), (70, 75, 82), 2)
    cv2.putText(img, label, (24, height // 2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220, 225, 230), 2, cv2.LINE_AA)
    cv2.putText(img, "stream unavailable", (24, height // 2 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (180, 186, 193), 2, cv2.LINE_AA)
    return img


def color_mask(da_mask: np.ndarray, ll_mask: np.ndarray) -> np.ndarray:
    viz = np.zeros((da_mask.shape[0], da_mask.shape[1], 3), dtype=np.uint8)
    viz[da_mask > 0] = (96, 96, 96)
    viz[ll_mask > 0] = (255, 255, 255)
    return viz


def make_bev(da_mask: np.ndarray, ll_mask: np.ndarray) -> np.ndarray:
    height, width = da_mask.shape[:2]
    src = np.float32([
        [width * 0.05, height * 0.98],
        [width * 0.95, height * 0.98],
        [width * 0.65, height * 0.62],
        [width * 0.35, height * 0.62],
    ])
    out_w = 480
    out_h = 480
    dst = np.float32([
        [0, out_h - 1],
        [out_w - 1, out_h - 1],
        [out_w - 1, 0],
        [0, 0],
    ])
    transform = cv2.getPerspectiveTransform(src, dst)
    road = cv2.warpPerspective((da_mask * 255).astype(np.uint8), transform, (out_w, out_h), flags=cv2.INTER_NEAREST)
    lane = cv2.warpPerspective((ll_mask * 255).astype(np.uint8), transform, (out_w, out_h), flags=cv2.INTER_NEAREST)
    bev = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    bev[road > 0] = (80, 80, 80)
    bev[lane > 0] = (255, 255, 255)
    return bev


def render_mask(valid: bool, label: str, da_mask: np.ndarray, ll_mask: np.ndarray) -> np.ndarray:
    if not valid:
        return make_unavailable_image(960, 540, f"{label} MASK")
    return color_mask(da_mask, ll_mask)


def compose_combined_bev(results: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], valid: dict[str, bool]) -> np.ndarray:
    canvas_h = 720
    canvas_w = 960
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    # Front camera occupies the forward center of the vehicle.
    if valid.get('front_raw'):
        front = make_bev(results['front_raw'][1], results['front_raw'][2])
        front = cv2.resize(front, (360, 360), interpolation=cv2.INTER_NEAREST)
        y0, x0 = 300, 300
        canvas[y0:y0 + 360, x0:x0 + 360] = np.maximum(canvas[y0:y0 + 360, x0:x0 + 360], front)

    # Side cameras are rotated into a common top-down frame and placed to the left/right of the front sector.
    for key, x0 in [('left_raw', 80), ('right_raw', 520)]:
        if not valid.get(key):
            continue
        side = make_bev(results[key][1], results[key][2])
        side = cv2.resize(side, (280, 280), interpolation=cv2.INTER_NEAREST)
        rotate_code = cv2.ROTATE_90_COUNTERCLOCKWISE if key == 'left_raw' else cv2.ROTATE_90_CLOCKWISE
        side = cv2.rotate(side, rotate_code)
        y0 = 360
        canvas[y0:y0 + 280, x0:x0 + 280] = np.maximum(canvas[y0:y0 + 280, x0:x0 + 280], side)

    cv2.rectangle(canvas, (458, 620), (502, 670), (0, 180, 255), -1)
    cv2.putText(canvas, 'vehicle', (430, 705), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 225, 230), 2, cv2.LINE_AA)
    return canvas


def save_image(path: Path, image: np.ndarray) -> None:
    tmp = path.parent / f'{path.stem}.tmp{path.suffix}'
    cv2.imwrite(str(tmp), image)
    tmp.replace(path)


def processing_loop(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_name in ('front_bev.jpg',):
        stale_path = output_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()
    (output_dir / 'index.html').write_text(
        HTML.format(snapshot_base=args.snapshot_base, refresh_ms=int(args.refresh_interval * 1000)),
        encoding='utf-8',
    )

    runner = YoloPv2Runner(
        project_root=Path(args.project_root),
        weights_path=Path(args.weights_path),
        device_name=args.device,
        img_size=args.img_size,
        conf=args.conf_thres,
        iou=args.iou_thres,
    )

    session = requests.Session()
    session.verify = False
    urls = {name: f'{args.snapshot_base.rstrip("/")}/{name}' for name in SNAPSHOT_NAMES}

    while True:
        started = time.perf_counter()
        frames = {}
        valid = {}
        for name, url in urls.items():
            try:
                frames[name] = fetch_image(session, url, args.timeout)
                valid[name] = True
            except Exception:
                frames[name] = make_unavailable_image(960, 540, name.replace('_raw', '').upper())
                valid[name] = False

        results = {}
        for name in SNAPSHOT_NAMES:
            try:
                if valid[name]:
                    overlay, da_mask, ll_mask = runner.infer(frames[name])
                else:
                    overlay = frames[name]
                    da_mask = np.zeros(frames[name].shape[:2], dtype=np.uint8)
                    ll_mask = np.zeros(frames[name].shape[:2], dtype=np.uint8)
            except Exception:
                overlay = frames[name]
                da_mask = np.zeros(frames[name].shape[:2], dtype=np.uint8)
                ll_mask = np.zeros(frames[name].shape[:2], dtype=np.uint8)
            results[name] = (overlay, da_mask, ll_mask)

        save_image(output_dir / 'left_raw.jpg', frames['left_raw'])
        save_image(output_dir / 'front_raw.jpg', frames['front_raw'])
        save_image(output_dir / 'right_raw.jpg', frames['right_raw'])
        save_image(output_dir / 'left_segmented.jpg', render_mask(valid['left_raw'], 'LEFT', results['left_raw'][1], results['left_raw'][2]))
        save_image(output_dir / 'front_mask.jpg', render_mask(valid['front_raw'], 'FRONT', results['front_raw'][1], results['front_raw'][2]))
        save_image(output_dir / 'right_segmented.jpg', render_mask(valid['right_raw'], 'RIGHT', results['right_raw'][1], results['right_raw'][2]))
        save_image(output_dir / 'combined_bev.jpg', compose_combined_bev(results, valid))
        (output_dir / 'status.json').write_text(json.dumps({
            'left_raw': valid['left_raw'],
            'front_raw': valid['front_raw'],
            'right_raw': valid['right_raw'],
            'timestamp': time.time(),
        }), encoding='utf-8')

        elapsed = time.perf_counter() - started
        sleep_for = max(0.01, args.refresh_interval - elapsed)
        time.sleep(sleep_for)


def serve_forever(directory: Path, host: str, port: int):
    handler = partial(SimpleHTTPRequestHandler, directory=str(directory))
    server = ThreadingHTTPServer((host, port), handler)
    server.serve_forever()


def main():
    parser = argparse.ArgumentParser(description='Run a live local YOLOPv2 dashboard from dinosaur snapshots.')
    parser.add_argument('--snapshot-base', default='https://dinosaur.tail808f0a.ts.net/snapshot')
    parser.add_argument('--project-root', default=str(DEFAULT_PROJECT_ROOT))
    parser.add_argument('--weights-path', default=str(DEFAULT_WEIGHTS))
    parser.add_argument('--output-dir', default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8022)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--conf-thres', type=float, default=0.30)
    parser.add_argument('--iou-thres', type=float, default=0.45)
    parser.add_argument('--refresh-interval', type=float, default=0.35)
    parser.add_argument('--timeout', type=float, default=4.0)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    worker = threading.Thread(target=processing_loop, args=(args,), daemon=True)
    worker.start()
    serve_forever(out_dir, args.host, args.port)


if __name__ == '__main__':
    main()
