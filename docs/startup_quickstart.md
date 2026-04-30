# Startup Quickstart

This is the shortest path to get the current testing stack back up without re-deriving the commands.

## Scope

This startup path does:

- reconnect to `dinosaur`
- make sure the three ZED raw camera topics are alive
- make sure the Jetson camera dashboard is serving snapshots on port `8090`
- start the local YOLO dashboard on port `8034`

This startup path does **not**:

- touch motors
- launch actuator control
- launch Nav2 motion
- launch the phone joystick UI

## One command

From the repo root:

```bash
bash scripts/start_live_video_stack.sh
```

When it succeeds, use:

```text
http://127.0.0.1:8034/index.html
```

## What the local page shows

- row 1: left raw | front raw | right raw
- row 2: left segmented | front segmented | right segmented
- row 3: combined BEV

## Jetson endpoints

Raw snapshot source:

```text
http://100.93.121.3:8090/snapshot/front_raw
http://100.93.121.3:8090/snapshot/left_raw
http://100.93.121.3:8090/snapshot/right_raw
```

Health endpoint:

```text
http://100.93.121.3:8090/health
```

Expected healthy state:

```text
left_raw=True
front_raw=True
right_raw=True
```

## What the script actually starts

On `dinosaur`:

- `camera_dashboard.py` via `uvicorn` on port `8090`
- `zed_front` with serial `42569280`
- `zed_left` with serial `43779087`
- `zed_right` with serial `49910017`

Locally:

- `scripts/run_yolo_live_dashboard.py` on `127.0.0.1:8034`

## If the page is up but empty

Check:

```bash
curl http://100.93.121.3:8090/health
```

If all three are `False`, the Jetson cameras are not publishing.

If the Jetson health is good but the local page is stale, restart the local viewer only:

```bash
pkill -f 'run_yolo_live_dashboard.py --host 127.0.0.1 --port 8034' || true
bash scripts/start_live_video_stack.sh
```

## If you want only status

```bash
bash scripts/start_live_video_stack.sh status
```
