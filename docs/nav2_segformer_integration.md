# Nav2 SegFormer Integration

This branch publishes a **local keepout mask** for Nav2, not a global map.

## Published Nav2 Topics

| Topic | Type | Notes |
|---|---|---|
| `/seg_ros/segformer/nav2/filter_mask` | `nav_msgs/msg/OccupancyGrid` | keepout mask for Nav2 filter |
| `/seg_ros/segformer/nav2/drivable_grid` | `nav_msgs/msg/OccupancyGrid` | debug drivable grid |
| `/seg_ros/segformer/nav2/costmap_filter_info` | `nav2_msgs/msg/CostmapFilterInfo` | filter metadata |

## Lane State Topics

| Topic | Type | Notes |
|---|---|---|
| `/seg_ros/segformer/lane_detected` | `std_msgs/msg/Bool` | true when lane corridor confidence is high enough |
| `/seg_ros/segformer/planner_mode_hint` | `std_msgs/msg/String` | `lane_following` or `obstacle_avoidance` |

## Grid Convention

- frame: `base_link`
- `x_range`: configurable, default `[0.0, 15.0]`
- `y_range`: configurable, default `[-10.0, 10.0]`
- resolution: configurable, default `0.05 m`
- origin:
  - `x = x_range[0]`
  - `y = y_range[0]`

This means the grid is defined directly in the vehicle frame instead of
being inferred from width/length only.

## Camera Prior

The branch also carries a front-camera mount prior:

- `camera_mount_x = 0.35`
- `camera_mount_y = 0.0`
- `camera_mount_z = 0.75`
- `camera_mount_yaw = 0.0`

This is treated as a practical initialization for the front ZED X mounting,
not as a final measured calibration.

## Projection Method

The current implementation uses a lower-image trapezoid and projects it into a local bird's-eye grid.

This is a pragmatic local planner representation, not a calibrated metric perception stack.

## Recommended Use

Use this branch with a **local Nav2 costmap filter**, not as a substitute for a proper global map.

Example config:

- [nav2_keepout_example.yaml](../config/nav2_keepout_example.yaml)

## Limits

- no depth fusion
- no LiDAR fusion
- no object-aware keepout beyond what the semantic road mask rejects
- projection is heuristic until calibrated against the actual camera geometry
