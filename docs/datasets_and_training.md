# Dataset And Training Notes

This file documents what data was used by the current pretrained models, what data was used locally for validation, and what still needs to be collected before claiming competition-level reliability.

## Current Position

The repository currently integrates pretrained models. It does not yet contain a custom-trained road segmentation model.

Project-owned work so far:

- ROS 2 Jazzy package structure
- ROS 2 publishers and launch files
- proof image generation scripts
- traffic-cone evaluation script
- combined road segmentation plus cone detection proof
- documentation and reproducible commands

Upstream-owned work:

- YOLOPv2 pretrained road/lane perception checkpoint
- Roboflow Logistics YOLOv8 object-detection checkpoint

## Model 1: YOLOPv2 Road And Lane Segmentation

Purpose:

- drivable-area segmentation
- lane-line segmentation
- driving-scene object detections from the upstream multitask model

Current checkpoint:

```text
/home/alexander/Desktop/seg/data/weights/yolopv2.pt
```

Upstream source:

```text
https://github.com/CAIC-AD/YOLOPv2
https://github.com/CAIC-AD/YOLOPv2/releases/download/V0.0.1/yolopv2.pt
```

Training / dataset notes:

- The checkpoint comes from the upstream YOLOPv2 project.
- YOLOPv2 is a multitask driving perception model built for object detection, drivable-area segmentation, and lane-line segmentation.
- The upstream project documents the model around BDD100K-style autonomous-driving perception tasks.
- This repository has not retrained YOLOPv2 and has not modified the YOLOPv2 architecture.
- The checkpoint is approximately 150 MB and is intentionally not committed to Git.

Why this model is used now:

- It is already available locally.
- It outputs both drivable-area masks and lane-line masks, which matches the competition road segmentation need.
- It can be wrapped cleanly behind ROS 2 image topics while keeping the model replaceable later.

Known risk:

- BDD100K-style driving data is not the same as the competition robot camera viewpoint.
- The current checkpoint may underperform on low camera height, parking lots, campus roads, unusual lighting, worn paint, and cones close to the robot.
- Real reliability requires local validation and likely local fine-tuning or replacement.

## Model 2: Roboflow Logistics YOLOv8 Object Detector

Purpose:

- traffic cone detection
- person detection
- traffic light, road sign, vehicle, and logistics-object detection

Current checkpoint:

```text
models/roboflow_logistics_yolov8.pt
```

Upstream source:

```text
https://universe.roboflow.com/wen-8qxpo/logistics-sz9jr-yvvjw
https://blog.roboflow.com/logistics-object-detection-model/
```

Training / dataset notes from Roboflow Universe:

- Dataset size: 99,238 images
- Classes: 20 logistics-focused classes
- Reported model metric: 76% mAP
- Roboflow notes that part of the dataset was auto-labeled using Autodistill DETIC.
- Relevant classes for this project include `traffic cone`, `person`, `traffic light`, `road sign`, `car`, `truck`, and `van`.

Why this model is used now:

- It includes a native `traffic cone` class.
- Standard COCO YOLO models normally do not provide a traffic-cone class.
- The checkpoint is small enough to commit to the repository.
- Local proof showed useful cone detections on road scenes.

Known risk:

- The model is not trained specifically for this robot, camera, or course.
- It should be treated as a starting detector, not a final safety system.
- People and cones should be fused with depth/geometric obstacle sensing before navigation decisions.

## Local Validation Data Used So Far

Road/lane proof images:

```text
proof/contact_sheet.jpg
proof/all2_overlay.jpg
proof/all3_overlay.jpg
proof/fs1_overlay.jpg
proof/fs2_overlay.jpg
proof/fs3_overlay.jpg
proof/lane1_overlay.jpg
```

Combined road + cone proof:

```text
proof/source_images/road_cars_cones_input.jpg
proof/combined/semantic_segmentation_plus_cones_contact_sheet.jpg
proof/combined/semantic_segmentation_plus_cones_road.jpg
```

Traffic-cone validation data:

```text
/home/alexander/Desktop/HERE_Object_Anomaly/cone_test/cone_dataset
proof/traffic_cones/traffic_cone_eval.json
proof/traffic_cones/actual_road_cone_test.json
```

The full cone dataset is local and is not committed. The repository includes proof images and JSON summaries.

Current traffic-cone evaluation result:

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

Road/lane segmentation does not yet have project-owned ZED X ground-truth masks, so road/lane IoU and mIoU are not reported yet.

Road-scene smoke test:

```text
non-cone road frames tested: 72
cone false positives on those frames: 0
selected road-cone scenes: 12
traffic cones detected: 59
```

## What Has Not Been Trained Yet

No project-owned fine-tuned checkpoint has been produced yet.

Still needed before claiming a custom-trained model:

1. Record local ZED X video from the actual robot camera.
2. Extract representative frames from competition-like routes.
3. Label drivable area, lane markings, traffic cones, people, and unusual road anomalies.
4. Split the data into train, validation, and test sets.
5. Fine-tune a selected model.
6. Save the model config, exact weights, dataset version, training command, and evaluation metrics.
7. Compare against the current pretrained baseline.

## Recommended Local Dataset Plan

Minimum useful collection:

| Split | Frames | Purpose |
|---|---:|---|
| Train | 500-1,000 | Fine-tuning road/lane/cone behavior |
| Validation | 100-200 | Tune thresholds and model selection |
| Test | 100-200 | Report final held-out metrics |

Scene coverage:

- straight roads
- curves and intersections
- cones near and far
- cones partially occluded by vehicles
- people near the road
- road debris or anomalies
- worn lane markings
- shadows and bright sun
- dusk or low-light frames if the robot will run then
- wet or reflective pavement if relevant

Annotation targets:

| Target | Annotation type |
|---|---|
| drivable road | segmentation mask |
| lane marking / road line | segmentation mask |
| traffic cone | bounding box first, mask optional later |
| person | bounding box |
| road anomaly / debris | bounding box or segmentation mask depending on size |
| ignore regions | mask for ambiguous/unsafe labels |

Recommended dataset structure:

```text
dataset_v001/
  images/
    train/
    val/
    test/
  labels_detection/
    train/
    val/
    test/
  masks_drivable/
    train/
    val/
    test/
  masks_lane/
    train/
    val/
    test/
  manifests/
    dataset_manifest.json
    split_manifest.json
    label_policy.md
```

Recommended manifest fields:

```json
{
  "dataset_name": "competition_road_perception",
  "version": "v001",
  "camera": "Stereolabs ZED X",
  "ros_distro": "jazzy",
  "image_topic": "/zed/zed_node/rgb/color/rect/image",
  "frame_id": "camera_color_optical_frame",
  "label_classes": ["drivable_area", "lane_marking", "traffic_cone", "person", "road_anomaly"],
  "splits": {
    "train": 0,
    "val": 0,
    "test": 0
  }
}
```

Split rules:

- Do not split adjacent video frames across train/val/test.
- Keep complete driving segments in only one split.
- Keep difficult scenes in all splits, not only the test split.
- The final test split should stay frozen after the first published baseline.
- Record camera height, ZED X calibration/profile settings, resolution, route, lighting, and weather for every session.

## Training Plan

Baseline:

- Keep YOLOPv2 and Roboflow Logistics as frozen pretrained baselines.
- Run them on all local validation frames.
- Save metrics and failure examples before training anything.

Fine-tuning path:

1. Start with the object detector because cone/person boxes are faster to label and evaluate.
2. Fine-tune the detector on local cone/person/anomaly images.
3. Keep a held-out test split that is never used for threshold tuning.
4. For road/lane masks, compare YOLOPv2 against a mask-first model such as TwinLiteNetPlus before committing to a training path.
5. Train or fine-tune the selected road/lane model only after the model input/output format is stable in ROS 2.

Metrics to report:

| Task | Metric |
|---|---|
| traffic cones | precision, recall, F1, mAP50 |
| people | precision, recall, F1, mAP50 |
| drivable area | IoU / mIoU |
| lane markings | IoU plus visual stability |
| live ROS 2 node | FPS, latency, dropped frames |
| navigation integration | false obstacle rate, missed obstacle rate, behavior in RViz/Nav2 |

Metric definitions:

```text
precision = true_positives / (true_positives + false_positives)
recall    = true_positives / (true_positives + false_negatives)
F1        = 2 * precision * recall / (precision + recall)
IoU       = intersection(mask_prediction, mask_ground_truth) / union(mask_prediction, mask_ground_truth)
```

Training artifacts that must be saved with any future custom model:

- dataset version
- git commit SHA
- training command
- model config
- pretrained checkpoint source
- final checkpoint path
- validation metrics
- held-out test metrics
- failure-case image folder
- inference latency on the target machine

## Professional Claim Boundary

It is accurate to say:

- The repo integrates pretrained road/lane segmentation and traffic-cone detection into ROS 2.
- The repo includes proof images and reproducible validation scripts.
- The current cone detector produced about 0.83 F1 on the local cone validation set.
- The current road/lane segmentation is a pretrained baseline and needs live camera validation.

It is not accurate yet to say:

- The project trained its own road segmentation model.
- The current system is safe for autonomous driving.
- The current model is fully reliable on the competition vehicle.
- Nav2 semantic costmap integration is complete.
