# ZED X Validation Workflow

This workflow contains only steps that are practical to run and verify. It does not claim field accuracy until ZED X frames are recorded and labeled.

## Goal

Build a small validation set from the actual robot camera, then measure:

- traffic-cone detection precision, recall, and F1
- live inference timing
- road/lane mask quality after labels exist

## Record ZED X Frames

Start the ZED ROS 2 wrapper first, then confirm the image topic:

```bash
ros2 topic list | grep zed
```

Current default topic:

```text
/zed/zed_node/rgb/color/rect/image
```

Older fallback topic:

```text
/zed/zed_node/rgb/image_rect_color
```

Record validation frames:

```bash
source /opt/ros/jazzy/setup.bash
source /home/alexander/Desktop/Competiton_Semantic_Segmentation/ros2_ws/install/setup.bash

ros2 launch seg_ros_bridge zed_image_recorder.launch.py \
  image_topic:=/zed/zed_node/rgb/color/rect/image \
  output_dir:=/home/alexander/Desktop/Competiton_Semantic_Segmentation/validation/zed_frames \
  max_frames:=200 \
  save_every_n:=5
```

Output:

```text
validation/zed_frames/
  frame_000001.jpg
  frame_000002.jpg
  ...
  manifest.json
```

## Extract Frames From A Video

If a normal video file is available instead of a live ROS topic:

```bash
/home/alexander/github/av-perception/.venv/bin/python \
  scripts/extract_validation_frames.py \
  --input-video /path/to/zed_recording.mp4 \
  --output-dir validation/zed_frames_from_video \
  --max-frames 200 \
  --step 30
```

## Benchmark Current Models

Run the combined segmentation/object pipeline on an image folder and write a JSON report:

```bash
/home/alexander/github/av-perception/.venv/bin/python \
  scripts/benchmark_live_perception.py \
  --image-dir validation/zed_frames \
  --output-json validation/benchmark_report.json \
  --device cpu \
  --limit 200
```

This reports timing and whether masks/detections are produced. It is not an accuracy measurement unless labels exist.

## Labeling Targets

Minimum labels for useful validation:

| Target | Label type |
|---|---|
| traffic cone | bounding box |
| person | bounding box |
| drivable road | segmentation mask |
| lane marking / road line | segmentation mask |
| road anomaly / debris | bounding box or mask |

## Accuracy Requirements

Object detection can be measured once boxes are labeled:

```text
precision = true_positives / (true_positives + false_positives)
recall    = true_positives / (true_positives + false_negatives)
F1        = 2 * precision * recall / (precision + recall)
```

Road/lane segmentation can be measured once masks are labeled:

```text
IoU = intersection(predicted_mask, ground_truth_mask) / union(predicted_mask, ground_truth_mask)
```

## Current Known Result

Existing local cone dataset:

```text
images tested: 48
ground-truth cones: 167
predicted cones: 168
precision: 0.8274
recall: 0.8323
F1: 0.8299
```

Live ROS smoke test using the saved road/cone proof image:

```text
traffic cones detected: 8
people detected: 2
cars detected: 1
segmentation detections: 2
CPU inference time: about 630 ms/frame
```

## Completion Boundary

This workflow can be completed locally up to frame recording, frame extraction, benchmarking, and object evaluation when labels exist.

It cannot prove robot field reliability until:

- the ZED X camera is running on the robot
- representative road/cone/person scenes are recorded
- labels are created
- evaluation is run on those labeled frames
