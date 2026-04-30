#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
JETSON_HOST="dinosaur@100.93.121.3"
JETSON_PASS="ruasonid"
LOCAL_DASH_PORT="8034"
JETSON_DASH_PORT="8090"
LOCAL_PYTHON="/home/alexander/github/av-perception/.venv/bin/python"
LOCAL_DASH_SCRIPT="$REPO_ROOT/scripts/run_yolo_live_dashboard.py"

remote() {
  sshpass -p "$JETSON_PASS" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 "$JETSON_HOST" "$@"
}

remote_bash() {
  remote "/bin/bash -lc '$1'"
}

print_status() {
  echo "Jetson camera dashboard:"
  curl -sS -m 3 "http://100.93.121.3:${JETSON_DASH_PORT}/health" || true
  echo
  echo "Local viewer:"
  curl -sS -I -m 3 "http://127.0.0.1:${LOCAL_DASH_PORT}/index.html" || true
}

start_jetson_dashboard() {
  remote_bash "source /opt/ros/humble/setup.bash && source ~/IGVC/install/setup.bash && nohup python3 -m uvicorn camera_dashboard:app --host 0.0.0.0 --port ${JETSON_DASH_PORT} --app-dir /home/dinosaur </dev/null > /tmp/camera_dashboard.log 2>&1 &"
}

start_zed_if_missing() {
  local name="$1"
  local serial="$2"
  local cfg="/home/dinosaur/zed_${name}_runtime.yaml"
  local topic="/zed_${name}/zed_node/rgb/color/rect/image"
  if remote_bash "source /opt/ros/humble/setup.bash && source ~/IGVC/install/setup.bash && ros2 topic list | grep -qx '${topic}'"; then
    return 0
  fi
  remote_bash "source /opt/ros/humble/setup.bash && source ~/IGVC/install/setup.bash && nohup /usr/bin/python3 /opt/ros/humble/bin/ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zedx camera_name:=zed_${name} serial_number:=${serial} publish_tf:=false publish_urdf:=false use_sim_time:=false ros_params_override_path:=${cfg} </dev/null > /tmp/zed_${name}_runtime.log 2>&1 &"
}

start_local_dashboard() {
  pkill -f "run_yolo_live_dashboard.py --host 127.0.0.1 --port ${LOCAL_DASH_PORT}" || true
  mkdir -p "$REPO_ROOT/.tmp"
  setsid -f "$LOCAL_PYTHON" -u "$LOCAL_DASH_SCRIPT" \
    --host 127.0.0.1 \
    --port "$LOCAL_DASH_PORT" \
    --snapshot-base "http://100.93.121.3:${JETSON_DASH_PORT}/snapshot" \
    --refresh-interval 0.5 \
    < /dev/null > "$REPO_ROOT/.tmp/yolo_live_dashboard.log" 2>&1
}

main() {
  local mode="${1:-start}"
  case "$mode" in
    status)
      print_status
      ;;
    start)
      start_jetson_dashboard
      sleep 2
      start_zed_if_missing "front" "42569280"
      start_zed_if_missing "left" "43779087"
      start_zed_if_missing "right" "49910017"
      sleep 6
      start_local_dashboard
      sleep 2
      print_status
      echo
      echo "Open: http://127.0.0.1:${LOCAL_DASH_PORT}/index.html"
      ;;
    *)
      echo "Usage: bash scripts/start_live_video_stack.sh [start|status]"
      exit 1
      ;;
  esac
}

main "$@"
