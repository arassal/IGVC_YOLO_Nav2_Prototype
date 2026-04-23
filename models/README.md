# Model Weights

This directory documents the model weights used by the ROS 2 perception proof.

## Included Weight

```text
models/roboflow_logistics_yolov8.pt
```

Use:

- traffic cone detection
- person detection
- traffic light detection
- road sign detection
- car, truck, and van detection

Source:

```text
https://universe.roboflow.com/wen-8qxpo/logistics-sz9jr-yvvjw
https://blog.roboflow.com/logistics-object-detection-model/
```

Dataset and training summary:

- Roboflow Logistics dataset
- 99,238 images
- 20 object classes
- reported 76% mAP
- part of the dataset was auto-labeled with Autodistill DETIC according to Roboflow

Relevant class names:

```text
person
traffic cone
traffic light
road sign
car
truck
van
```

This checkpoint is committed because it is small enough for normal Git usage.

## External Weight

```text
/home/alexander/Desktop/seg/data/weights/yolopv2.pt
```

Use:

- drivable-area segmentation
- lane-line segmentation
- upstream YOLOPv2 driving-scene detections

Source project:

```text
https://github.com/CAIC-AD/YOLOPv2
```

Source release:

```text
https://github.com/CAIC-AD/YOLOPv2/releases/download/V0.0.1/yolopv2.pt
```

Training summary:

- upstream pretrained YOLOPv2 checkpoint
- multitask driving perception model
- documented around BDD100K-style driving perception tasks
- not trained or modified in this repository

This checkpoint is roughly 150 MB and is intentionally excluded from this repository. If it needs to be versioned later, use Git LFS or attach it to a GitHub Release instead of committing it as a normal blob.

## Training Ownership

The current repository owns the ROS 2 wrapper, proof generation, evaluation scripts, and integration documentation. It does not yet own a fine-tuned road segmentation checkpoint.

See:

```text
docs/datasets_and_training.md
```
