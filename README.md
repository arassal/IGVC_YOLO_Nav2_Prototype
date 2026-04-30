# IGVC YOLO Nav2 Prototype

Public dashboard:

- https://arassal.github.io/IGVC_YOLO_Nav2_Prototype/

Working branch for this prototype:

- `codex/yolo-nav2-rebuild`

Rebuild contract for another agent:

- [docs/rebuild_contract.md](/home/alexander/Desktop/IGVC_Nav2_SegFormer/docs/rebuild_contract.md)

Startup quickstart:

- [docs/startup_quickstart.md](/home/alexander/Desktop/IGVC_Nav2_SegFormer/docs/startup_quickstart.md)

This repository is the current ROS 2 / Nav2 prototype for IGVC-style lane following with:

- `YOLOPv2` for drivable area and lane segmentation
- `YOLOv8` for obstacle blocking
- a Nav2-facing BEV keepout/drivable grid

The target path is:

```text
front ZED image
-> YOLOPv2 drivable + lane masks
-> YOLOv8 obstacle removal
-> BEV projection
-> OccupancyGrid + CostmapFilterInfo
-> Nav2 local costmap
```

The repo also includes a local testing dashboard for:

- left raw
- front raw
- right raw
- left/front/right semantic mask panels
- combined BEV

## What matters

The primary deliverable is not the UI. It is the ROS 2 / Nav2 contract.

This branch is considered usable when it does these things cleanly:

- builds in ROS 2 Jazzy
- subscribes to a live ZED image topic
- publishes drivable and lane masks
- publishes Nav2-compatible `OccupancyGrid`
- publishes Nav2-compatible `CostmapFilterInfo`
- keeps the output frame in `base_link`

## Main ROS node

Primary node:

- `live_perception_node`

Source:

- [live_perception_node.py](/home/alexander/Desktop/IGVC_Nav2_SegFormer/ros2_ws/src/seg_ros_bridge/seg_ros_bridge/live_perception_node.py)

Launch:

- [live_perception.launch.py](/home/alexander/Desktop/IGVC_Nav2_SegFormer/ros2_ws/src/seg_ros_bridge/launch/live_perception.launch.py)

## ROS 2 interface

Image/debug topics:

- `/seg_ros/live/input_image`
- `/seg_ros/live/overlay_image`
- `/seg_ros/live/drivable_mask`
- `/seg_ros/live/lane_mask`
- `/seg_ros/live/lane_confidence`
- `/seg_ros/live/nav2/bev_debug`

Metadata topics:

- `/seg_ros/live/detections`
- `/seg_ros/live/timing`
- `/seg_ros/live/label_info`

Nav2 topics:

- `/seg_ros/live/nav2/filter_mask`
- `/seg_ros/live/nav2/drivable_grid`
- `/seg_ros/live/nav2/costmap_filter_info`

Nav2 message types:

- `/seg_ros/live/nav2/filter_mask` -> `nav_msgs/msg/OccupancyGrid`
- `/seg_ros/live/nav2/drivable_grid` -> `nav_msgs/msg/OccupancyGrid`
- `/seg_ros/live/nav2/costmap_filter_info` -> `nav2_msgs/msg/CostmapFilterInfo`

Frame assumptions:

- published grid frame: `base_link`
- grid origin:
  - `x = nav2_x_range[0]`
  - `y = nav2_y_range[0]`

## Build

```bash
cd /home/alexander/Desktop/IGVC_Nav2_SegFormer/ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select seg_ros_bridge
source install/setup.bash
```

## Live launch

```bash
cd /home/alexander/Desktop/IGVC_Nav2_SegFormer/ros2_ws
source /opt/ros/jazzy/setup.bash
source install/setup.bash

ros2 launch seg_ros_bridge live_perception.launch.py \
  image_topic:=/zed_front/zed_node/rgb/color/rect/image \
  device:=cuda:0
```

If GPU is not available:

```bash
ros2 launch seg_ros_bridge live_perception.launch.py \
  image_topic:=/zed_front/zed_node/rgb/color/rect/image \
  device:=cpu
```

## Required external weights

YOLOPv2 weights:

- `/home/alexander/github/av-perception/data/weights/yolopv2.pt`

YOLOv8 object weights:

- [models/roboflow_logistics_yolov8.pt](/home/alexander/Desktop/IGVC_Nav2_SegFormer/models/roboflow_logistics_yolov8.pt)

Weight notes:

- [models/README.md](/home/alexander/Desktop/IGVC_Nav2_SegFormer/models/README.md)

## Dashboard

Local dashboard runner:

- [run_yolo_live_dashboard.py](/home/alexander/Desktop/IGVC_Nav2_SegFormer/scripts/run_yolo_live_dashboard.py)

Current layout:

- row 1: left raw | front raw | right raw
- row 2: left semantic | front semantic | right semantic
- row 3: combined BEV

This is for testing only. Nav2 compatibility is determined by the ROS topics above, not by the browser output.

## What another agent must preserve

If another agent rebuilds this, the following must remain true:

1. `live_perception_node` stays the ROS entry point
2. Nav2 topics keep the same names
3. Nav2 topics keep the same message types
4. `CostmapFilterInfo.filter_mask_topic` points to `/seg_ros/live/nav2/filter_mask`
5. grid frame remains `base_link`
6. drivable space is reduced by detected obstacles before grid publication
7. launch file exposes the BEV/grid parameters

## Verification status

Verified locally on this branch:

- `python3 -m py_compile` passes for the live node
- `colcon build --packages-select seg_ros_bridge` passes
- the node now contains a Nav2-facing occupancy-grid path

Not yet honestly field-verified:

- final on-car latency
- final BEV trapezoid tuning
- full Nav2 behavior on the vehicle

So the right claim is:

```text
ROS 2 compatible: yes
Nav2 interface compatible: yes
field-proven on the car: not yet
```
