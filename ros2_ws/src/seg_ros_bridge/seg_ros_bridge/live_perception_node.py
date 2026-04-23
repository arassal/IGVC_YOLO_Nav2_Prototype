import json
import os
import sys
import time

import cv2
import numpy as np
import rclpy
import torch
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from ultralytics import YOLO
from vision_msgs.msg import LabelInfo, VisionClass


DEFAULT_PROJECT_ROOT = '/home/alexander/Desktop/seg'
DEFAULT_SEG_WEIGHTS = '/home/alexander/Desktop/seg/data/weights/yolopv2.pt'
DEFAULT_OBJECT_MODEL = (
    '/home/alexander/Desktop/Competiton_Semantic_Segmentation/'
    'models/roboflow_logistics_yolov8.pt'
)
DEFAULT_CLASSES = [
    'person',
    'traffic cone',
    'traffic light',
    'road sign',
    'car',
    'truck',
    'van',
]
DEFAULT_CLASSES_PARAM = ','.join(DEFAULT_CLASSES)


class LivePerceptionNode(Node):
    """Run road/lane segmentation and object detection on live ROS images."""

    def __init__(self) -> None:
        super().__init__('live_perception_node')

        self.declare_parameter('image_topic', '/zed/zed_node/rgb/color/rect/image')
        self.declare_parameter('project_root', DEFAULT_PROJECT_ROOT)
        self.declare_parameter('segmentation_weights_path', DEFAULT_SEG_WEIGHTS)
        self.declare_parameter('object_model_path', DEFAULT_OBJECT_MODEL)
        self.declare_parameter('device', 'cpu')
        self.declare_parameter('img_size', 640)
        self.declare_parameter('seg_conf_thres', 0.30)
        self.declare_parameter('seg_iou_thres', 0.45)
        self.declare_parameter('object_confidence', 0.35)
        self.declare_parameter('enabled_classes', DEFAULT_CLASSES_PARAM)
        self.declare_parameter('process_every_n', 1)
        self.declare_parameter('publish_input_image', True)
        self.declare_parameter('publish_timing', True)

        self.image_topic = self.get_parameter('image_topic').value
        self.project_root = self.get_parameter('project_root').value
        self.segmentation_weights_path = self.get_parameter(
            'segmentation_weights_path').value
        self.object_model_path = self.get_parameter('object_model_path').value
        self.device_name = self.get_parameter('device').value
        self.img_size = int(self.get_parameter('img_size').value)
        self.seg_conf_thres = float(self.get_parameter('seg_conf_thres').value)
        self.seg_iou_thres = float(self.get_parameter('seg_iou_thres').value)
        self.object_confidence = float(self.get_parameter('object_confidence').value)
        self.enabled_classes = self._parse_classes(
            self.get_parameter('enabled_classes').value)
        self.process_every_n = max(1, int(self.get_parameter('process_every_n').value))
        self.publish_input_image = bool(self.get_parameter('publish_input_image').value)
        self.publish_timing = bool(self.get_parameter('publish_timing').value)

        self._validate_paths()
        self._load_yolopv2_utils()
        self._load_models()

        self.bridge = CvBridge()
        self.frame_count = 0
        self.last_label_publish = 0.0

        self.input_pub = self.create_publisher(
            Image, '/seg_ros/live/input_image', 10)
        self.overlay_pub = self.create_publisher(
            Image, '/seg_ros/live/overlay_image', 10)
        self.drivable_pub = self.create_publisher(
            Image, '/seg_ros/live/drivable_mask', 10)
        self.lane_pub = self.create_publisher(
            Image, '/seg_ros/live/lane_mask', 10)
        self.lane_confidence_pub = self.create_publisher(
            Image, '/seg_ros/live/lane_confidence', 10)

        label_qos = QoSProfile(depth=1)
        label_qos.reliability = ReliabilityPolicy.RELIABLE
        label_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.label_pub = self.create_publisher(
            LabelInfo, '/seg_ros/live/label_info', label_qos)

        self.detection_pub = self.create_publisher(
            String, '/seg_ros/live/detections', 10)
        self.timing_pub = self.create_publisher(
            String, '/seg_ros/live/timing', 10)

        self.image_sub = self.create_subscription(
            Image, self.image_topic, self._image_cb, 10)

        self._publish_label_info()
        self.get_logger().info(f'Subscribed to live image topic: {self.image_topic}')
        self.get_logger().info(
            'Publishing live perception outputs under /seg_ros/live/*')

    def _validate_paths(self) -> None:
        if not os.path.isdir(self.project_root):
            raise RuntimeError(f'YOLOPv2 project_root does not exist: {self.project_root}')
        if not os.path.isfile(self.segmentation_weights_path):
            raise RuntimeError(
                'YOLOPv2 segmentation checkpoint does not exist: '
                f'{self.segmentation_weights_path}')
        if not os.path.isfile(self.object_model_path):
            raise RuntimeError(
                f'Object detector checkpoint does not exist: {self.object_model_path}')

    def _load_yolopv2_utils(self) -> None:
        if self.project_root not in sys.path:
            sys.path.insert(0, self.project_root)

        from utils.utils import (
            driving_area_mask,
            lane_line_mask,
            letterbox,
            non_max_suppression,
            scale_coords,
            select_device,
            show_seg_result,
            split_for_trace_model,
        )

        self.driving_area_mask = driving_area_mask
        self.lane_line_mask = lane_line_mask
        self.letterbox = letterbox
        self.non_max_suppression = non_max_suppression
        self.scale_coords = scale_coords
        self.select_device = select_device
        self.show_seg_result = show_seg_result
        self.split_for_trace_model = split_for_trace_model

    def _load_models(self) -> None:
        self.device = self.select_device(self.device_name)
        self.segmentation_model = torch.jit.load(
            self.segmentation_weights_path,
            map_location=self.device,
        ).to(self.device)
        self.half = self.device.type != 'cpu'
        if self.half:
            self.segmentation_model.half()
        self.segmentation_model.eval()

        self.object_model = YOLO(self.object_model_path)
        self.get_logger().info(
            f'Loaded YOLOPv2 segmentation: {self.segmentation_weights_path}')
        self.get_logger().info(
            f'Loaded object detector: {self.object_model_path}')

    def _parse_classes(self, value):
        if isinstance(value, str):
            return {part.strip() for part in value.split(',') if part.strip()}
        return {str(part) for part in value}

    def _image_cb(self, msg: Image) -> None:
        self.frame_count += 1
        if self.frame_count % self.process_every_n != 0:
            return

        start = time.perf_counter()
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().warning(f'Failed to convert input image: {exc}')
            return

        try:
            overlay, da_mask, ll_mask, seg_detections = self._run_segmentation(frame)
            object_detections = self._run_object_detection(frame)
        except Exception as exc:
            self.get_logger().warning(f'Live perception inference failed: {exc}')
            return

        combined_overlay = overlay.copy()
        for det in object_detections:
            self._draw_detection(combined_overlay, det)

        self._publish_images(msg, frame, combined_overlay, da_mask, ll_mask)
        self._publish_detections(msg, seg_detections, object_detections, start)
        self._publish_label_info_throttled()

    def _run_segmentation(self, frame):
        original_h, original_w = frame.shape[:2]
        model_frame = cv2.resize(
            frame.copy(), (1280, 720), interpolation=cv2.INTER_LINEAR)
        img = self.letterbox(model_frame, self.img_size, stride=32)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        tensor = torch.from_numpy(img).to(self.device)
        tensor = tensor.half() if self.half else tensor.float()
        tensor /= 255.0
        if tensor.ndimension() == 3:
            tensor = tensor.unsqueeze(0)

        with torch.no_grad():
            pred_raw, seg, ll = self.segmentation_model(tensor)

        if isinstance(pred_raw, (list, tuple)) and len(pred_raw) == 2:
            pred = self.split_for_trace_model(pred_raw[0], pred_raw[1])
        else:
            pred = self.split_for_trace_model(pred_raw, None)
        pred = self.non_max_suppression(
            pred, self.seg_conf_thres, self.seg_iou_thres)

        da_mask = self.driving_area_mask(seg)
        ll_mask = self.lane_line_mask(ll)

        overlay = model_frame.copy()
        self.show_seg_result(overlay, (da_mask, ll_mask), is_demo=True)

        detections = []
        for det in pred:
            if len(det):
                det[:, :4] = self.scale_coords(
                    tensor.shape[2:], det[:, :4], overlay.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    detections.append({
                        'xyxy': [float(v) for v in xyxy],
                        'confidence': float(conf),
                        'class_id': int(cls),
                    })

        if (original_w, original_h) != (1280, 720):
            overlay = cv2.resize(
                overlay, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
            da_mask = cv2.resize(
                da_mask.astype(np.uint8),
                (original_w, original_h),
                interpolation=cv2.INTER_NEAREST,
            )
            ll_mask = cv2.resize(
                ll_mask.astype(np.uint8),
                (original_w, original_h),
                interpolation=cv2.INTER_NEAREST,
            )

        return overlay, da_mask, ll_mask, detections

    def _run_object_detection(self, frame):
        result = self.object_model.predict(
            frame,
            conf=self.object_confidence,
            imgsz=640,
            verbose=False,
            device=self.device_name,
        )[0]

        detections = []
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = result.names[class_id]
            if class_name not in self.enabled_classes:
                continue
            detections.append({
                'type': class_name.replace(' ', '_'),
                'class_name': class_name,
                'confidence': float(box.conf[0]),
                'xyxy': [float(v) for v in box.xyxy[0]],
            })
        return detections

    def _publish_images(self, source_msg, frame, overlay, da_mask, ll_mask):
        if self.publish_input_image:
            input_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            input_msg.header = source_msg.header
            self.input_pub.publish(input_msg)

        overlay_msg = self.bridge.cv2_to_imgmsg(overlay, encoding='bgr8')
        overlay_msg.header = source_msg.header
        self.overlay_pub.publish(overlay_msg)

        drivable_msg = self.bridge.cv2_to_imgmsg(
            (da_mask * 255).astype(np.uint8), encoding='mono8')
        drivable_msg.header = source_msg.header
        self.drivable_pub.publish(drivable_msg)

        lane_mask = (ll_mask * 255).astype(np.uint8)
        lane_msg = self.bridge.cv2_to_imgmsg(lane_mask, encoding='mono8')
        lane_msg.header = source_msg.header
        self.lane_pub.publish(lane_msg)

        confidence_msg = self.bridge.cv2_to_imgmsg(lane_mask, encoding='mono8')
        confidence_msg.header = source_msg.header
        self.lane_confidence_pub.publish(confidence_msg)

    def _publish_detections(self, source_msg, seg_detections, object_detections, start):
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        payload = {
            'header': {
                'stamp': {
                    'sec': source_msg.header.stamp.sec,
                    'nanosec': source_msg.header.stamp.nanosec,
                },
                'frame_id': source_msg.header.frame_id,
            },
            'segmentation_detections': {
                'count': len(seg_detections),
                'detections': seg_detections,
            },
            'competition_objects': {
                'count': len(object_detections),
                'detections': object_detections,
            },
            'timing_ms': elapsed_ms,
        }
        msg = String()
        msg.data = json.dumps(payload)
        self.detection_pub.publish(msg)

        if self.publish_timing:
            timing_msg = String()
            timing_msg.data = json.dumps({
                'frame_id': source_msg.header.frame_id,
                'timing_ms': elapsed_ms,
                'process_every_n': self.process_every_n,
            })
            self.timing_pub.publish(timing_msg)

    def _publish_label_info_throttled(self):
        now = time.monotonic()
        if now - self.last_label_publish > 5.0:
            self._publish_label_info()

    def _publish_label_info(self):
        msg = LabelInfo()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_color_optical_frame'
        msg.threshold = float(self.seg_conf_thres)
        msg.class_map = [
            VisionClass(class_id=0, class_name='background'),
            VisionClass(class_id=1, class_name='drivable_area'),
            VisionClass(class_id=2, class_name='lane_marking'),
        ]
        self.label_pub.publish(msg)
        self.last_label_publish = time.monotonic()

    def _draw_detection(self, image, det):
        color = self._color_for(det['class_name'])
        x1, y1, x2, y2 = [int(v) for v in det['xyxy']]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
        cv2.putText(
            image,
            f"{det['class_name']} {det['confidence']:.2f}",
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )

    def _color_for(self, class_name):
        if class_name == 'traffic cone':
            return (0, 140, 255)
        if class_name == 'person':
            return (0, 220, 255)
        if class_name in {'car', 'truck', 'van'}:
            return (255, 130, 0)
        if class_name in {'traffic light', 'road sign'}:
            return (60, 220, 60)
        return (255, 255, 255)


def main(args=None):
    rclpy.init(args=args)
    node = LivePerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
