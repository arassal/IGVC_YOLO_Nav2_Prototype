# Vehicle Geometry And Bring-Up

This document is the source of truth for:

- sensor geometry
- camera serial mapping
- ZED bring-up
- perception validation
- Nav2 semantic costmap validation

The goal is to remove guesswork. Fill this out once, then use it every time the vehicle comes up.

## Known Now

These are the only facts that should be assumed without re-measuring:

- active vehicle workspace on `dinosaur`: `/home/dinosaur/IGVC`
- do not mix it with `/home/dinosaur/IGVC_ROS2`
- front camera is the primary required camera
- back camera is out of scope
- current physical sketch measurements are in inches
- updated front camera measurement from the user: `(23 7/8, 0, -3)` using the same sketch convention

## Step 1: Define The Robot Frame

Before touching TF, URDF, BEV, or costmaps, define `base_link`.

Record exactly one answer:

- rear axle center
- chassis center
- XSENS location
- another fixed point

Use this template:

```text
base_link definition:
coordinate convention:
units:
```

Required convention confirmation:

```text
+x =
+y =
+z =
```

Recommended convention:

```text
+x = forward
+y = left
+z = up
```

## Step 2: Sensor Geometry Sheet

Record every sensor pose relative to `base_link`.

Use one row per sensor:

```text
sensor_name, x, y, z, yaw, pitch, roll, units, notes
```

Required sensors:

- `zed_front`
- `zed_left`
- `zed_right`
- `lidar`
- `xsens`

Copy and fill this:

```text
base_link:
units:
convention:

zed_front: x=?, y=?, z=?, yaw=?, pitch=?, roll=?
zed_left:  x=?, y=?, z=?, yaw=?, pitch=?, roll=?
zed_right: x=?, y=?, z=?, yaw=?, pitch=?, roll=?
lidar:     x=?, y=?, z=?, yaw=?, pitch=?, roll=?
xsens:     x=?, y=?, z=?, yaw=?, pitch=?, roll=?
```

### What counts as a good measurement

- Use one unit system only.
- State whether the measurement is to the sensor center, lens center, or housing reference point.
- If pitch and roll are assumed zero, write `0` explicitly.
- If a value is estimated, mark it as estimated.

## Step 3: Vehicle Dimensions

These are needed for Nav2 tuning and BEV sanity checks.

Fill this in:

```text
wheelbase:
track_width:
overall_width:
overall_length:
front_overhang:
rear_overhang:
min_turn_radius:
```

If `min_turn_radius` is unknown, compute it later from steering geometry.

## Step 4: Camera Serial Map

Do not trust memory here. Write down the actual mapping once.

Use this table:

```text
physical_position, serial_number, ros_namespace, status, verified_by
```

Template:

```text
front, ?, /zed_front, ?, ?
left,  ?, /zed_left,  ?, ?
right, ?, /zed_right, ?, ?
```

### Verification rule

A camera is only "verified" when all of these are true:

1. the ZED SDK detects it
2. the ROS node launches
3. the RGB topic publishes
4. the point cloud topic publishes
5. you visually confirmed the physical direction

## Step 5: Minimal Bring-Up Order

Bring the system up in this order.

### 5.1 Power and machine check

Run:

```bash
hostname
date
df -h
free -h
```

Confirm:

- correct machine
- enough disk
- enough memory

### 5.2 Check Tailscale / SSH access

Run:

```bash
tailscale status
hostname -I
```

Record:

- tailnet name
- current IP

### 5.3 Check ZED hardware visibility

Run:

```bash
/usr/local/zed/tools/ZED_Explorer -a
```

Record:

- detected serials
- any missing serial

### 5.4 Launch one camera only

Start with front camera first.

Verify:

```bash
ros2 topic list | grep zed_front
ros2 topic hz /zed_front/zed_node/rgb/color/rect/image
ros2 topic hz /zed_front/zed_node/point_cloud/cloud_registered
```

Do not move on until front is confirmed.

### 5.5 Launch left camera only

Repeat the same validation for left.

### 5.6 Launch right camera only

Repeat the same validation for right.

### 5.7 Launch combined perception

Only after single-camera validation passes.

Verify:

```bash
ros2 topic list | grep /perception/
ros2 topic hz /perception/front/semantic_mask
ros2 topic hz /perception/front/semantic_points
```

Repeat for left and right when enabled.

## Step 6: Perception Validation Checklist

For each active camera, verify:

- raw RGB image is visible
- point cloud is publishing
- segmentation mask is publishing
- confidence image is publishing
- label info is publishing once and remains latched
- overlay image is visible

Use this checklist:

```text
front_raw:
front_cloud:
front_mask:
front_confidence:
front_label_info:
front_overlay:

left_raw:
left_cloud:
left_mask:
left_confidence:
left_label_info:
left_overlay:

right_raw:
right_cloud:
right_mask:
right_confidence:
right_label_info:
right_overlay:
```

## Step 7: Nav2 Semantic Costmap Validation

The semantic pipeline is only useful if the costmap consumes it correctly.

Verify:

```bash
ros2 topic hz /local_costmap/costmap
ros2 topic info /perception/front/semantic_mask
ros2 topic info /perception/front/semantic_points
ros2 node info /controller_server
```

Confirm:

- costmap is publishing
- semantic mask topic exists
- semantic pointcloud topic exists
- `controller_server` is alive

### Functional validation

Check that:

- lane pixels appear in the local costmap
- obstacle classes appear in the local costmap
- stale detections decay at the expected rate
- the local costmap frame matches expected robot orientation

## Step 8: Data Collection For Model Improvement

If SegFormer stays in the project, this is the most valuable next step.

Collect:

- front camera sequences
- left camera sequences
- right camera sequences
- cones / barrels
- lane boundaries
- pothole markings
- glare
- shadows
- grass/asphalt transitions

Minimum useful dataset:

- 100 to 300 labeled images

Target classes:

- `lane_white`
- `barrel_orange`
- `pothole`
- `grass`
- `drivable_asphalt`
- `background`

## Step 9: SegFormer Decision Rule

SegFormer is acceptable if it helps semantic context and costmap labeling.

Do not use SegFormer as the only lane solution unless testing proves it.

Recommended split:

```text
SegFormer = semantic context
geometry / BEV = lane structure
Nav2 semantic layer = planner interface
```

## Step 10: First Inputs Needed From You

Send these first:

1. `base_link` definition
2. coordinate convention
3. units
4. one completed sensor geometry sheet
5. one completed camera serial map

Once that is filled in, the next step is:

- URDF / TF reconciliation
- camera launch verification
- perception-to-costmap alignment
