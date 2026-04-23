# Technical Architecture

This document describes the current ROS 2 perception implementation at the runtime-contract level. It is intentionally specific about what the repository does today and what still needs to be implemented for live robot operation.

## Runtime Stack

| Layer | Component | Current implementation |
|---|---|---|
| ROS 2 package | `seg_ros_bridge` | Python `ament_python` package |
| ROS distro | Jazzy | validated locally |
| Road/lane backend | YOLOPv2 TorchScript checkpoint | external `yolopv2.pt` |
| Object backend | Ultralytics YOLOv8 checkpoint | committed Roboflow Logistics `.pt` |
| Image conversion | `cv_bridge` | OpenCV BGR images to ROS `sensor_msgs/msg/Image` |
| Metadata | `vision_msgs/msg/LabelInfo` | semantic class map for road/lane outputs |
| Detection transport | `std_msgs/msg/String` | JSON payload for easy inspection |
| Live input | ROS `sensor_msgs/msg/Image` subscriber | ZED X rectified RGB topic by default |

## Nodes

### `seg_demo_node`

File:

```text
ros2_ws/src/seg_ros_bridge/seg_ros_bridge/seg_demo_node.py
```

Purpose:

- loads YOLOPv2
- loops through a deterministic image directory
- runs drivable-area segmentation, lane-line segmentation, and YOLOPv2 detections
- publishes input image, overlay image, masks, label metadata, and detection JSON

Parameters:

| Parameter | Default | Meaning |
|---|---|---|
| `project_root` | `/home/alexander/Desktop/seg` | path containing YOLOPv2 repo utilities |
| `image_dir` | `${project_root}/data/demo` | static demo image directory |
| `weights_path` | `${project_root}/data/weights/yolopv2.pt` | YOLOPv2 TorchScript checkpoint |
| `device` | `cpu` | PyTorch device string |
| `img_size` | `640` | YOLOPv2 letterbox input size |
| `conf_thres` | `0.30` | YOLOPv2 detection confidence threshold |
| `iou_thres` | `0.45` | YOLOPv2 NMS IoU threshold |
| `publish_rate_hz` | `1.0` | static-image publish rate |

Launch:

```bash
source /opt/ros/jazzy/setup.bash
cd /home/alexander/Desktop/Competiton_Semantic_Segmentation/ros2_ws
ros2 launch seg_ros_bridge seg_demo.launch.py
```

### `competition_objects_node`

File:

```text
ros2_ws/src/seg_ros_bridge/seg_ros_bridge/competition_objects_node.py
```

Purpose:

- loads Roboflow Logistics YOLOv8 checkpoint
- loops through a deterministic image directory
- filters detections to competition-relevant classes
- publishes input image, annotated image, and detection JSON

Parameters:

| Parameter | Default | Meaning |
|---|---|---|
| `image_dir` | `proof/traffic_cones/raw_road_inputs` | static object-demo image directory |
| `model_path` | `models/roboflow_logistics_yolov8.pt` | YOLOv8 object checkpoint |
| `enabled_classes` | `person`, `traffic cone`, `traffic light`, `road sign`, `car`, `truck`, `van` | class allow-list |
| `confidence` | `0.35` | YOLOv8 confidence threshold |
| `device` | `cpu` | Ultralytics device string |
| `publish_rate_hz` | `1.0` | static-image publish rate |

Launch:

```bash
source /opt/ros/jazzy/setup.bash
cd /home/alexander/Desktop/Competiton_Semantic_Segmentation/ros2_ws
ros2 launch seg_ros_bridge competition_objects.launch.py
```

### `live_perception_node`

File:

```text
ros2_ws/src/seg_ros_bridge/seg_ros_bridge/live_perception_node.py
```

Purpose:

- subscribes to a live ROS image topic
- runs YOLOPv2 road/lane segmentation on each selected frame
- runs Roboflow Logistics YOLOv8 object detection on the same frame
- publishes combined overlay, masks, label metadata, detections, and timing JSON
- preserves the input image timestamp and `frame_id` on image outputs

Parameters:

| Parameter | Default | Meaning |
|---|---|---|
| `image_topic` | `/zed/zed_node/rgb/color/rect/image` | ZED X rectified RGB topic to subscribe to |
| `project_root` | `/home/alexander/Desktop/seg` | path containing YOLOPv2 repo utilities |
| `segmentation_weights_path` | `/home/alexander/Desktop/seg/data/weights/yolopv2.pt` | YOLOPv2 TorchScript checkpoint |
| `object_model_path` | `models/roboflow_logistics_yolov8.pt` | YOLOv8 object checkpoint |
| `device` | `cpu` | PyTorch / Ultralytics device string |
| `img_size` | `640` | YOLOPv2 letterbox input size |
| `seg_conf_thres` | `0.30` | YOLOPv2 detection confidence threshold |
| `seg_iou_thres` | `0.45` | YOLOPv2 NMS IoU threshold |
| `object_confidence` | `0.35` | YOLOv8 confidence threshold |
| `enabled_classes` | competition allow-list | comma-separated object classes |
| `process_every_n` | `1` | process every Nth frame |
| `publish_input_image` | `true` | republish the input frame |
| `publish_timing` | `true` | publish per-frame timing JSON |

Launch:

```bash
source /opt/ros/jazzy/setup.bash
cd /home/alexander/Desktop/Competiton_Semantic_Segmentation/ros2_ws
ros2 launch seg_ros_bridge live_perception.launch.py \
  image_topic:=/zed/zed_node/rgb/color/rect/image \
  device:=cpu
```

Legacy ZED ROS 2 topic fallback:

```text
/zed/zed_node/rgb/image_rect_color
```

## Road/Lane Inference Path

Current path:

```text
OpenCV image from disk
  -> resize to 1280x720
  -> YOLOPv2 letterbox to img_size=640
  -> BGR to RGB
  -> HWC to CHW tensor
  -> normalize to 0.0-1.0
  -> torch.jit YOLOPv2 forward pass
  -> detection NMS
  -> drivable-area mask extraction
  -> lane-line mask extraction
  -> overlay rendering
  -> ROS image/mask/topic publication
```

YOLOPv2 model outputs consumed by the node:

| Output | Use |
|---|---|
| `pred_raw` | object detections processed through YOLOPv2 NMS |
| `seg` | drivable-area segmentation tensor |
| `ll` | lane-line segmentation tensor |

Road/lane masks are published as `mono8` images:

```text
0   = background
255 = active mask pixel
```

Semantic labels published in `vision_msgs/msg/LabelInfo`:

| Class ID | Class name |
|---:|---|
| 0 | `background` |
| 1 | `drivable_area` |
| 2 | `lane_marking` |

## Object Detection Path

Current path:

```text
OpenCV image from disk
  -> Ultralytics YOLOv8 predict(imgsz=640)
  -> confidence threshold
  -> class allow-list
  -> annotated image rendering
  -> JSON detection publication
```

Enabled classes:

```text
person
traffic cone
traffic light
road sign
car
truck
van
```

Traffic cones use the `traffic cone` class from the Roboflow Logistics model. The node exports a normalized type string of `traffic_cone` in JSON.

## Live Combined Perception Path

Current live path:

```text
ROS sensor_msgs/msg/Image
  -> cv_bridge bgr8 OpenCV frame
  -> YOLOPv2 road/lane inference
  -> Roboflow YOLOv8 object inference
  -> draw object boxes over road/lane overlay
  -> publish masks, overlay, detections, and timing
```

The live node outputs masks at the same image resolution as the incoming camera frame. Internally, YOLOPv2 still runs through the current 1280x720 preprocessing path, then the masks and overlay are resized back to the source frame dimensions.

## ROS 2 Topic Contract

### Road/Lane Topics

| Topic | Type | Encoding / payload | Publisher |
|---|---|---|---|
| `/seg_ros/input_image` | `sensor_msgs/msg/Image` | `bgr8` | `seg_demo_node` |
| `/seg_ros/overlay_image` | `sensor_msgs/msg/Image` | `bgr8` | `seg_demo_node` |
| `/seg_ros/drivable_mask` | `sensor_msgs/msg/Image` | `mono8` | `seg_demo_node` |
| `/seg_ros/lane_mask` | `sensor_msgs/msg/Image` | `mono8` | `seg_demo_node` |
| `/seg_ros/lane_confidence` | `sensor_msgs/msg/Image` | `mono8` | `seg_demo_node` |
| `/seg_ros/label_info` | `vision_msgs/msg/LabelInfo` | semantic label map | `seg_demo_node` |
| `/seg_ros/detections` | `std_msgs/msg/String` | JSON | `seg_demo_node` |

### Competition Object Topics

| Topic | Type | Encoding / payload | Publisher |
|---|---|---|---|
| `/seg_ros/competition_objects/input_image` | `sensor_msgs/msg/Image` | `bgr8` | `competition_objects_node` |
| `/seg_ros/competition_objects/annotated_image` | `sensor_msgs/msg/Image` | `bgr8` | `competition_objects_node` |
| `/seg_ros/competition_objects/detections` | `std_msgs/msg/String` | JSON | `competition_objects_node` |

### Live Perception Topics

| Topic | Type | Encoding / payload | Publisher |
|---|---|---|---|
| `/seg_ros/live/input_image` | `sensor_msgs/msg/Image` | `bgr8` | `live_perception_node` |
| `/seg_ros/live/overlay_image` | `sensor_msgs/msg/Image` | `bgr8` | `live_perception_node` |
| `/seg_ros/live/drivable_mask` | `sensor_msgs/msg/Image` | `mono8` | `live_perception_node` |
| `/seg_ros/live/lane_mask` | `sensor_msgs/msg/Image` | `mono8` | `live_perception_node` |
| `/seg_ros/live/lane_confidence` | `sensor_msgs/msg/Image` | `mono8` | `live_perception_node` |
| `/seg_ros/live/label_info` | `vision_msgs/msg/LabelInfo` | semantic label map | `live_perception_node` |
| `/seg_ros/live/detections` | `std_msgs/msg/String` | JSON | `live_perception_node` |
| `/seg_ros/live/timing` | `std_msgs/msg/String` | JSON | `live_perception_node` |

## QoS

Most image and detection topics currently use the default publisher depth of `10`.

`/seg_ros/label_info` uses:

```text
reliability: reliable
durability: transient local
depth: 1
```

That lets late subscribers receive the semantic class map after the node has already started.

## Detection JSON Schemas

### Road/Lane YOLOPv2 Detection Payload

Published on:

```text
/seg_ros/detections
```

Shape:

```json
{
  "image": "frame_name.jpg",
  "count": 1,
  "detections": [
    {
      "xyxy": [123.0, 45.0, 320.0, 210.0],
      "conf": 0.81,
      "cls": 2
    }
  ]
}
```

### Competition Object Detection Payload

Published on:

```text
/seg_ros/competition_objects/detections
```

Shape:

```json
{
  "image": "frame_name.jpg",
  "count": 1,
  "detections": [
    {
      "type": "traffic_cone",
      "class_name": "traffic cone",
      "confidence": 0.87,
      "xyxy": [248.0, 315.0, 302.0, 417.0]
    }
  ]
}
```

### Live Combined Detection Payload

Published on:

```text
/seg_ros/live/detections
```

Shape:

```json
{
  "header": {
    "stamp": {
      "sec": 0,
      "nanosec": 0
    },
    "frame_id": "camera_color_optical_frame"
  },
  "segmentation_detections": {
    "count": 0,
    "detections": []
  },
  "competition_objects": {
    "count": 1,
    "detections": [
      {
        "type": "traffic_cone",
        "class_name": "traffic cone",
        "confidence": 0.87,
        "xyxy": [248.0, 315.0, 302.0, 417.0]
      }
    ]
  },
  "timing_ms": 215.4
}
```

Coordinate convention:

- `xyxy` is image-pixel bounding box format.
- Coordinates are `[x_min, y_min, x_max, y_max]`.
- The current static demo uses the image frame as the reference frame.
- For robot navigation, these 2D boxes still need projection through camera intrinsics and depth/lidar before becoming physical obstacles.

## Validation Methodology

Road/lane validation currently checks:

- non-empty drivable-area masks
- non-empty lane-line masks
- visual overlay quality on static road images
- ROS topic publication and message encodings
- live node startup and ROS image subscription
- header preservation on live derived image outputs

Road/lane validation does not yet report IoU or mIoU because the project does not yet have labeled ZED X road/lane ground-truth masks.

Traffic-cone validation currently checks:

- XML annotation parsing
- prediction filtering to `traffic cone`
- IoU-based matching at `iou=0.50`
- precision, recall, and F1
- false positives on road frames without labeled cones

Current recorded cone metrics:

```text
images evaluated: 48
ground-truth cones: 167
predicted cones: 168
true positives: 139
false positives: 29
false negatives: 28
precision: 0.8274
recall: 0.8323
F1: 0.8299
```

Latest live subscriber smoke test:

```text
input: proof/source_images/road_cars_cones_input.jpg published as ROS Image
topic: /codex/test_image
output: /seg_ros/live/detections
segmentation detections: 2
traffic cones detected: 8
people detected: 2
cars detected: 1
CPU timing: about 630 ms/frame
```

## Live Robot Integration Requirements

The first camera-backed subscriber is implemented in `live_perception_node`. The next requirement is hardware validation against the actual ZED X stream on the robot.

Target input topics:

| Topic | Type | Use |
|---|---|---|
| `/zed/zed_node/rgb/color/rect/image` | `sensor_msgs/msg/Image` | rectified RGB inference input |
| `/zed/zed_node/rgb/camera_info` | `sensor_msgs/msg/CameraInfo` | projection and geometry |
| `/zed/zed_node/depth/depth_registered` | `sensor_msgs/msg/Image` | 2D box to 3D estimate |
| `/zed/zed_node/point_cloud/cloud_registered` | `sensor_msgs/msg/PointCloud2` | independent geometric obstacle layer |

Older ZED ROS 2 wrapper topic names may differ. Confirm with:

```bash
ros2 topic list | grep zed
```

Implementation requirements:

1. Preserve the input image timestamp on all derived masks and detections. Implemented for image outputs and serialized in JSON detections.
2. Preserve the input `frame_id`. Implemented for image outputs and serialized in JSON detections.
3. Throttle or drop frames if inference cannot keep up. Basic `process_every_n` frame skipping is implemented.
4. Publish mask dimensions matching the camera image dimensions. Implemented by resizing outputs back to source dimensions.
5. Add configurable topic remaps instead of hard-coded camera names. Implemented through the `image_topic` parameter.
6. Keep segmentation and object detection optional so one failed model does not stop the full perception stack. Not implemented yet.

## Nav2 Integration Boundary

The drivable-area mask is the candidate signal for semantic costmap work.

The lane-line mask should not be used as a collision obstacle by default. It is better treated as:

- lane-centering cue
- debug visualization
- route preference signal
- optional boundary hint after projection is validated

Traffic cones and people should become safety cues only after fusion with geometry:

```text
2D detection
  -> camera intrinsics
  -> depth/lidar association
  -> estimated 3D position
  -> obstacle or keepout region
  -> Nav2 costmap / behavior layer
```

## Technical Debt

Current technical debt:

- static image folders are still kept for deterministic proofs, while `live_perception_node` provides the live subscriber path
- JSON detections instead of typed `vision_msgs/msg/Detection2DArray`
- hard-coded absolute paths in launch defaults
- YOLOPv2 utility imports depend on external repo path
- no latency/FPS benchmark script yet
- no automated ROS integration test
- no local fine-tuned checkpoint yet
- no field validation on actual ZED X road/cone video in this repository yet

These are acceptable for the current proof stage, but they should be addressed before calling the system competition-ready.
