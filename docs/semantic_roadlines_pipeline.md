# Semantic Road-Line Segmentation Pipeline

Living project file. Update this as the ROS 2 semantic road-line segmentation work progresses.

Date started: 2026-04-13  
Owner: AVROS / Alexander  
Goal: accurate pretrained road-line and drivable-road segmentation in ROS 2, then adapt it into our own ROS 2 package and navigation pipeline.

## Current Decision

Use a pretrained lane-line segmentation model first, wrap it in our own ROS 2 package, and keep the model replaceable behind a stable topic interface.

Current source-of-truth documentation for model provenance, datasets, and training ownership:

```text
docs/datasets_and_training.md
docs/technical_architecture.md
models/README.md
```

Primary candidate for the current working implementation:

- **YOLOPv2**
- Local project: `/home/alexander/Desktop/seg`
- Local weights: `/home/alexander/Desktop/seg/data/weights/yolopv2.pt`
- ROS 2 bridge: `/home/alexander/Desktop/Competiton_Semantic_Segmentation/ros2_ws/src/seg_ros_bridge`
- Proof outputs: `/home/alexander/Desktop/Competiton_Semantic_Segmentation/proof`
- Why this is the current implementation: pretrained weights are already present locally, the model runs successfully, and the ROS 2 bridge publishes lane masks, drivable masks, overlays, confidence, label metadata, and detections.

Primary candidate for the next model comparison:

- **TwinLiteNetPlus**
- Repo: https://github.com/chequanghuy/TwinLiteNetPlus
- Why: directly targets **drivable area segmentation** and **lane-line segmentation**, has pretrained models, is designed for embedded deployment, and reports strong BDD100K results.
- Reported result from repo:
  - TwinLiteNetPlus Large: 92.9% drivable area mIoU
  - TwinLiteNetPlus Large: 81.9% lane accuracy
  - TwinLiteNetPlus Large: 34.2% lane IoU
  - 1.94M parameters
- Current limitation: not a ROS 2 package out of the box. We make it ours by creating the ROS 2 wrapper, topic contract, configs, launch files, post-processing, and later local fine-tuning.

Backup / benchmark candidates:

- **YOLOP**
  - Repo: https://github.com/hustvl/YOLOP
  - License: MIT
  - Why: pretrained multitask driving perception model with object detection, drivable-area segmentation, and lane-line segmentation.
  - Reported result from repo:
    - drivable area segmentation mIoU: 91.5%
    - lane detection mIoU: 70.50%
    - lane IoU: 26.20%
    - speed: 41 FPS in their benchmark table
  - Use if TwinLiteNetPlus export/deployment is painful.

- **Ultra-Fast-Lane-Detection-v2**
  - Repo: https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2
  - License: MIT
  - Why: very strong lane-line detection model with pretrained weights for CULane, TuSimple, and CurveLanes, plus ONNX and TensorRT deployment notes.
  - Reported result from repo:
    - CULane ResNet18 F1: 75.0
    - CULane ResNet34 F1: 76.0
    - TuSimple ResNet18 F1: 96.11
    - TuSimple ResNet34 F1: 96.24
    - CurveLanes ResNet18 F1: 80.42
    - CurveLanes ResNet34 F1: 81.34
  - Limitation: lane detector, not semantic segmentation mask. Good for lane geometry, not direct Nav2 semantic mask integration.

## Important Clarification

I did not find a mature, high-quality, ROS 2-native lane-line semantic segmentation package with a modern free pretrained model that we should trust blindly.

The correct engineering approach is:

1. choose a strong pretrained open model,
2. wrap it in our own ROS 2 package,
3. publish standard ROS 2 image and vision topics,
4. benchmark it on our camera data,
5. fine-tune or post-process until it works on our vehicle.

This still makes the final system ours. The pretrained model is the starting checkpoint, not the product.

## Target ROS 2 Pipeline

```text
ZED X / camera image
  /zed/zed_node/rgb/color/rect/image
  /zed/zed_node/rgb/camera_info
        |
        v
avros_lane_segmentation_node
  model backend:
    TwinLiteNetPlus first
    YOLOP fallback
    UFLDv2 optional lane-geometry benchmark
        |
        v
ROS 2 outputs
  /perception/lane_segmentation/mask
  /perception/lane_segmentation/confidence
  /perception/lane_segmentation/overlay
  /perception/drivable_area/mask
  /perception/road_lines/markers
  /perception/road_lines/polygons
  /perception/lane_segmentation/label_info
        |
        v
Post-processing
  resize to camera resolution
  remove small blobs
  temporal smoothing
  lane centerline extraction
  optional bird's-eye projection
        |
        v
Navigation integration
  Nav2 semantic costmap layer
  lane-following behavior
  drivable-road preference
  geometric obstacle layer remains active
```

## ROS 2 Topic Contract

Inputs:

| Topic | Type | Notes |
|---|---|---|
| `/zed/zed_node/rgb/color/rect/image` | `sensor_msgs/msg/Image` | Main rectified RGB input |
| `/zed/zed_node/rgb/camera_info` | `sensor_msgs/msg/CameraInfo` | Needed for projection and overlays |
| `/zed/zed_node/point_cloud/cloud_registered` | `sensor_msgs/msg/PointCloud2` | Needed later for Nav2 semantic costmap projection |

Outputs:

| Topic | Type | Encoding / content |
|---|---|---|
| `/perception/lane_segmentation/mask` | `sensor_msgs/msg/Image` | `mono8`, 0 background, 1 lane marking |
| `/perception/lane_segmentation/confidence` | `sensor_msgs/msg/Image` | `mono8`, 0-255 confidence |
| `/perception/lane_segmentation/overlay` | `sensor_msgs/msg/Image` | RGB debug visualization |
| `/perception/drivable_area/mask` | `sensor_msgs/msg/Image` | `mono8`, 0 background, 1 drivable |
| `/perception/lane_segmentation/label_info` | `vision_msgs/msg/LabelInfo` | transient-local QoS |
| `/perception/road_lines/markers` | `visualization_msgs/msg/MarkerArray` | RViz lane curves |
| `/perception/road_lines/polygons` | `geometry_msgs/msg/PolygonStamped` or custom msg | Optional for lane-following |

LabelInfo class map:

| ID | Name |
|---:|---|
| 0 | background |
| 1 | lane_marking |
| 2 | drivable_area |
| 255 | unknown |

## Implementation Phases

### Phase 0: Workspace and Package Ownership

Status: not started

Create an AVROS-owned ROS 2 package, not just a cloned research repo.

Package name:

```text
avros_lane_segmentation
```

Proposed structure:

```text
avros_ws/src/avros_lane_segmentation/
  package.xml
  setup.py
  resource/avros_lane_segmentation
  avros_lane_segmentation/
    __init__.py
    lane_segmentation_node.py
    model_backend.py
    twinlitenet_backend.py
    yolop_backend.py
    postprocess.py
    class_map.py
  launch/
    lane_segmentation.launch.py
  config/
    lane_segmentation.yaml
    model_paths.yaml
  models/
    README.md
  docs/
    model_research.md
```

Acceptance criteria:

- Package builds with `colcon build --symlink-install`.
- Node starts with no model loaded and prints a clear missing-model error.
- Launch file remaps camera topic cleanly.

### Phase 1: Model Download and Offline Test

Status: not started

Tasks:

1. Download TwinLiteNetPlus pretrained weights from the link in the repo.
2. Run the model on static images from our camera or a BDD100K sample.
3. Confirm it outputs lane-line and drivable-area masks.
4. Save visual outputs in a project folder.
5. Record exact model file name, checksum, and source URL.

Acceptance criteria:

- We can run inference on a single image outside ROS 2.
- The lane mask visibly tracks road lines.
- The drivable mask roughly tracks road area.
- The exact pretrained weight is documented.

### Phase 2: ROS 2 Image Inference Node

Status: not started

Tasks:

1. Subscribe to RGB image.
2. Convert ROS image to OpenCV/Numpy.
3. Preprocess exactly as the model expects.
4. Run inference.
5. Convert model output to masks.
6. Publish lane mask, drivable mask, confidence, overlay, and label info.

Acceptance criteria:

- `ros2 topic hz /perception/lane_segmentation/mask` is stable.
- Output masks have the same timestamp as the input image.
- Output masks can be viewed in RViz and `rqt_image_view`.
- LabelInfo is available to late subscribers.

### Phase 3: Post-Processing for Road Lines

Status: not started

Tasks:

1. Remove small connected components.
2. Apply lane-mask threshold tuning.
3. Skeletonize lane markings if needed.
4. Fit lane curves or polylines.
5. Publish RViz markers.
6. Add temporal smoothing to reduce flicker.

Acceptance criteria:

- Lane-line overlay is stable frame to frame.
- False positive speckles are removed.
- Lane curves are visible in RViz.
- Post-processing latency is measured.

### Phase 4: Nav2 Integration

Status: not started

Tasks:

1. Feed drivable area and non-drivable area into semantic costmap work.
2. Keep geometric obstacle layers active.
3. Use lane lines as navigation hints, not collision truth.
4. Add a debug mode that shows:
   - raw RGB
   - lane mask
   - drivable mask
   - costmap
   - planned path

Acceptance criteria:

- Nav2 prefers drivable road.
- Vehicle does not treat lane-line mask as a physical obstacle by default.
- Lane markings can be used for lane-centering only after projection/calibration is validated.

### Phase 5: Make It Ours

Status: not started

Tasks:

1. Own the ROS 2 package and API.
2. Keep third-party model code isolated under clear license notes.
3. Write our own launch/config/topic documentation.
4. Record our own validation dataset with the ZED X camera.
5. Label a small local dataset for road lines and drivable areas.
6. Fine-tune the model on our camera perspective and local routes.
7. Export our tuned model to ONNX/TensorRT.

Acceptance criteria:

- The runtime package is named and structured as AVROS code.
- Model source and license are documented.
- Local test results are documented.
- Fine-tuned checkpoint is stored separately from upstream pretrained checkpoint.

## Candidate Model Comparison

| Model | Best use | Free pretrained? | ROS 2 native? | Deployment | Notes |
|---|---|---:|---:|---|---|
| TwinLiteNetPlus | lane-line + drivable segmentation | yes | no | PyTorch first, export later | Best starting point for semantic masks |
| YOLOP | multitask driving perception | yes | no | PyTorch, ONNX, TensorRT notes | Strong fallback; MIT license |
| UFLDv2 | lane geometry detection | yes | no | ONNX and TensorRT documented | Accurate lane detector; not mask-first |
| PytorchAutoDrive | benchmarking and training framework | likely via docs/wiki | no | ONNX/TensorRT support | Useful research framework, heavier to integrate |
| ros_deep_learning SegNet | general semantic segmentation in ROS 2 | yes | yes | TensorRT on Jetson | Good for road/sidewalk classes, weak for road-line markings |

## Accuracy Strategy

Road-line segmentation is harder than road/sidewalk segmentation because lane markings are thin, partially occluded, worn, reflective, and lighting-sensitive.

Accuracy plan:

1. Start with TwinLiteNetPlus Large, not Nano, unless speed is unacceptable.
2. Benchmark on our own ZED X frames before integrating with Nav2.
3. Use confidence thresholds and connected-component cleanup.
4. Fuse with drivable-area segmentation so isolated false lane marks outside road are rejected.
5. Add temporal smoothing across frames.
6. Fine-tune with local data once the pipeline works.

First target metrics:

| Metric | Target |
|---|---:|
| RGB input rate | 15-30 FPS |
| inference output | 10+ FPS minimum |
| image-to-mask latency | under 150 ms initially |
| stable lane overlay in RViz | required |
| false lane marks outside drivable area | very low |
| local validation set | at least 100 labeled frames |

## Integration With Existing Semantic Segmentation PDF

Existing reference:

```text
/home/alexander/Downloads/semantic_segmentation_integration.pdf
```

That document focuses on generic semantic road segmentation through Nav2's semantic segmentation layer. This file narrows the plan to **road-line / lane-line segmentation**.

Connection between the two:

- The existing PDF's Nav2 semantic layer plan is still useful for drivable/non-drivable costmaps.
- The lane-line model should publish standard masks and label info, matching the same ROS 2 style.
- Lane lines should not automatically become obstacles. They are navigation cues.
- Drivable-area segmentation can feed Nav2 costmaps.
- Lane-line segmentation can feed lane-centering, visualization, or route constraints after calibration.

## Open Questions

- What ROS 2 distro are we actually deploying on: Humble, Jazzy, or Rolling?
- What Jetson model and JetPack version are installed?
- Do we need CPU fallback on laptop, or only Jetson GPU/TensorRT?
- Is the target environment normal roads, parking lots, campus paths, or painted indoor lanes?
- Do we need lane following, road boundary following, or just road/non-road semantic costmaps?

## Immediate Next Actions

1. Create `avros_lane_segmentation` ROS 2 package.
2. Download TwinLiteNetPlus pretrained model.
3. Run offline inference on sample images.
4. If the pretrained output is acceptable, wrap it in ROS 2.
5. If TwinLiteNetPlus is hard to export or inaccurate on our camera, test YOLOP.
6. Use UFLDv2 only if we need explicit lane curves rather than segmentation masks.

## Current Working Commands

Build the local ROS 2 bridge:

```bash
cd /home/alexander/Desktop/Competiton_Semantic_Segmentation/ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select seg_ros_bridge
```

Run the pretrained YOLOPv2 ROS 2 publisher:

```bash
source /opt/ros/jazzy/setup.bash
cd /home/alexander/Desktop/seg
/home/alexander/github/av-perception/.venv/bin/python \
  /home/alexander/Desktop/Competiton_Semantic_Segmentation/ros2_ws/src/seg_ros_bridge/seg_ros_bridge/seg_demo_node.py \
  --ros-args -p device:=cpu -p publish_rate_hz:=0.5
```

Verify topics:

```bash
source /opt/ros/jazzy/setup.bash
ros2 topic list | grep '^/seg_ros/'
ros2 topic echo /seg_ros/label_info --once
ros2 topic echo /seg_ros/lane_mask --once
```

Export saved proof images:

```bash
/home/alexander/github/av-perception/.venv/bin/python \
  /home/alexander/Desktop/Competiton_Semantic_Segmentation/scripts/export_roadline_proof.py \
  --limit 6 --device cpu
```

## Research Sources

- TwinLiteNetPlus: https://github.com/chequanghuy/TwinLiteNetPlus
- YOLOP: https://github.com/hustvl/YOLOP
- Ultra-Fast-Lane-Detection-v2: https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2
- PytorchAutoDrive: https://github.com/voldemortX/pytorch-auto-drive
- ROS 2 `vision_msgs`: https://github.com/ros-perception/vision_msgs
- `vision_msgs/msg/LabelInfo`: https://docs.ros.org/en/ros2_packages/humble/api/vision_msgs/msg/LabelInfo.html
- `ros_deep_learning`: https://github.com/dusty-nv/ros_deep_learning
- Nav2 semantic segmentation tutorial: https://docs.nav2.org/tutorials/docs/navigation2_with_semantic_segmentation.html

## Update Log

### 2026-04-13

- Created this pipeline file.
- Searched for ROS 2-compatible road-line semantic segmentation options.
- Found no mature ROS 2-native lane-line segmentation package with a modern pretrained model that should be adopted directly.
- Selected TwinLiteNetPlus as the primary model candidate because it directly outputs lane-line and drivable-area segmentation and has pretrained weights.
- Kept YOLOP as fallback and UFLDv2 as lane-geometry benchmark.
- Promoted the locally available YOLOPv2 project to the first working implementation because the pretrained weights are already installed at `/home/alexander/Desktop/seg/data/weights/yolopv2.pt`.
- Ran the YOLOPv2 ROS 2 bridge successfully.
- Verified published topics:
  - `/seg_ros/detections`
  - `/seg_ros/drivable_mask`
  - `/seg_ros/input_image`
  - `/seg_ros/label_info`
  - `/seg_ros/lane_confidence`
  - `/seg_ros/lane_mask`
  - `/seg_ros/overlay_image`
- Updated the ROS 2 bridge to publish `vision_msgs/msg/LabelInfo` and `/seg_ros/lane_confidence`.
- Built `seg_ros_bridge` successfully with `colcon build --packages-select seg_ros_bridge`.
- Exported proof images to `/home/alexander/Desktop/roadline_demo_proof`.
- Proof run processed six demo images and produced non-empty lane and drivable masks:
  - `all2.jpg`: 22,216 lane pixels, 342,950 drivable pixels
  - `all3.jpg`: 30,512 lane pixels, 252,702 drivable pixels
  - `fs1.jpg`: 29,901 lane pixels, 284,413 drivable pixels
  - `fs2.jpg`: 20,362 lane pixels, 168,026 drivable pixels
  - `fs3.jpg`: 2,759 lane pixels, 145,975 drivable pixels
  - `lane1.jpg`: 15,326 lane pixels, 73,835 drivable pixels

### 2026-04-15

- Added professional dataset and training documentation to `docs/datasets_and_training.md`.
- Clarified that YOLOPv2 and Roboflow Logistics are upstream pretrained checkpoints.
- Documented local validation data separately from upstream training data.
- Updated README and model docs so the repository does not overclaim custom training or production safety.
