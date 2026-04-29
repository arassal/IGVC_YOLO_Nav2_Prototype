# Rebuild Contract

This file exists so another agent can recreate the current prototype without guessing which code path matters.

## Goal

Recreate a ROS 2 Jazzy perception node that:

1. subscribes to one live ZED image topic
2. runs `YOLOPv2` for drivable area and lane masks
3. runs `YOLOv8` for obstacle detection
4. removes detected obstacles from drivable space
5. projects the result into a BEV occupancy grid
6. publishes Nav2-compatible keepout/drivable topics

## Canonical entry point

Use this node:

- `ros2_ws/src/seg_ros_bridge/seg_ros_bridge/live_perception_node.py`

Use this launch file:

- `ros2_ws/src/seg_ros_bridge/launch/live_perception.launch.py`

Do not replace the entry point with the older BEV-only node for this branch.

## Canonical topic contract

### Inputs

- `image_topic` parameter
  - expected default: `/zed/zed_node/rgb/color/rect/image`

### Debug / visualization outputs

- `/seg_ros/live/input_image`
- `/seg_ros/live/overlay_image`
- `/seg_ros/live/drivable_mask`
- `/seg_ros/live/lane_mask`
- `/seg_ros/live/lane_confidence`
- `/seg_ros/live/nav2/bev_debug`

### Metadata outputs

- `/seg_ros/live/detections`
- `/seg_ros/live/timing`
- `/seg_ros/live/label_info`

### Nav2 outputs

- `/seg_ros/live/nav2/filter_mask`
- `/seg_ros/live/nav2/drivable_grid`
- `/seg_ros/live/nav2/costmap_filter_info`

## Required message types

- `/seg_ros/live/nav2/filter_mask` -> `nav_msgs/msg/OccupancyGrid`
- `/seg_ros/live/nav2/drivable_grid` -> `nav_msgs/msg/OccupancyGrid`
- `/seg_ros/live/nav2/costmap_filter_info` -> `nav2_msgs/msg/CostmapFilterInfo`

If those types change, Nav2 compatibility is broken.

## Required grid semantics

- grid frame must be `base_link`
- grid origin must be:
  - `x = nav2_x_range[0]`
  - `y = nav2_y_range[0]`
- keepout mask must use:
  - `100` for keepout / blocked
  - `0` for free
- `CostmapFilterInfo.filter_mask_topic` must equal:
  - `/seg_ros/live/nav2/filter_mask`

## Required logic

The live node must preserve this order:

1. infer drivable mask from YOLOPv2
2. infer lane mask from YOLOPv2
3. infer object boxes from YOLOv8
4. carve object boxes out of drivable mask
5. project drivable and lane masks into BEV
6. publish keepout and drivable grids

If an agent skips step 4, the prototype no longer matches the current intent.

## Parameters that matter

- `nav2_publish_grid`
- `nav2_grid_resolution`
- `nav2_x_range`
- `nav2_y_range`
- `src_bottom_y`
- `src_top_y`
- `src_bottom_left_x`
- `src_bottom_right_x`
- `src_top_left_x`
- `src_top_right_x`

These must remain launch-configurable.

## External dependencies

YOLOPv2 source tree:

- `/home/alexander/github/av-perception`

YOLOPv2 weights:

- `/home/alexander/github/av-perception/data/weights/yolopv2.pt`

YOLOv8 weights:

- `models/roboflow_logistics_yolov8.pt`

## Build check

Minimum build check:

```bash
cd ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select seg_ros_bridge
```

Minimum syntax check:

```bash
python3 -m py_compile ros2_ws/src/seg_ros_bridge/seg_ros_bridge/live_perception_node.py
```

## What is not part of the contract

These are useful, but not the primary compatibility contract:

- the browser dashboard
- GitHub Pages
- old SegFormer path
- old BEV-only path

If an agent has to choose between preserving the browser page and preserving the ROS/Nav2 topic contract, preserve the ROS/Nav2 topic contract.
