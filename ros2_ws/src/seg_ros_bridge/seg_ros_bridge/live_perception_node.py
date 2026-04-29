import json
import os
import sys
import time

import cv2
import numpy as np
import rclpy
import torch
from cv_bridge import CvBridge
from nav2_msgs.msg import CostmapFilterInfo
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from ultralytics import YOLO
from vision_msgs.msg import LabelInfo, VisionClass


DEFAULT_PROJECT_ROOT = '/home/alexander/github/av-perception'
DEFAULT_SEG_WEIGHTS = '/home/alexander/github/av-perception/data/weights/yolopv2.pt'
DEFAULT_OBJECT_MODEL = (
    '/home/alexander/Desktop/IGVC_Nav2_SegFormer/'
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
        self.declare_parameter('nav2_publish_grid', True)
        self.declare_parameter('nav2_grid_resolution', 0.05)
        self.declare_parameter('nav2_x_range', [0.0, 15.0])
        self.declare_parameter('nav2_y_range', [-10.0, 10.0])
        self.declare_parameter('src_bottom_y', 0.98)
        self.declare_parameter('src_top_y', 0.62)
        self.declare_parameter('src_bottom_left_x', 0.05)
        self.declare_parameter('src_bottom_right_x', 0.95)
        self.declare_parameter('src_top_left_x', 0.35)
        self.declare_parameter('src_top_right_x', 0.65)

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
        self.nav2_publish_grid = bool(self.get_parameter('nav2_publish_grid').value)
        self.nav2_grid_resolution = float(self.get_parameter('nav2_grid_resolution').value)
        self.nav2_x_range = [float(v) for v in self.get_parameter('nav2_x_range').value]
        self.nav2_y_range = [float(v) for v in self.get_parameter('nav2_y_range').value]
        self.src_bottom_y = float(self.get_parameter('src_bottom_y').value)
        self.src_top_y = float(self.get_parameter('src_top_y').value)
        self.src_bottom_left_x = float(self.get_parameter('src_bottom_left_x').value)
        self.src_bottom_right_x = float(self.get_parameter('src_bottom_right_x').value)
        self.src_top_left_x = float(self.get_parameter('src_top_left_x').value)
        self.src_top_right_x = float(self.get_parameter('src_top_right_x').value)

        self.nav2_x_min, self.nav2_x_max = self.nav2_x_range
        self.nav2_y_min, self.nav2_y_max = self.nav2_y_range
        self.nav2_grid_width_m = self.nav2_y_max - self.nav2_y_min
        self.nav2_grid_length_m = self.nav2_x_max - self.nav2_x_min
        self.nav2_grid_width_cells = max(
            1, int(round(self.nav2_grid_width_m / self.nav2_grid_resolution)))
        self.nav2_grid_height_cells = max(
            1, int(round(self.nav2_grid_length_m / self.nav2_grid_resolution)))

        self._validate_paths()
        self._load_yolopv2_utils()
        self._load_models()

        self.bridge = CvBridge()
        self.frame_count = 0
        self.last_label_publish = 0.0

        self.input_pub = self.create_publisher(Image, '/seg_ros/live/input_image', 10)
        self.overlay_pub = self.create_publisher(Image, '/seg_ros/live/overlay_image', 10)
        self.drivable_pub = self.create_publisher(Image, '/seg_ros/live/drivable_mask', 10)
        self.lane_pub = self.create_publisher(Image, '/seg_ros/live/lane_mask', 10)
        self.lane_confidence_pub = self.create_publisher(
            Image, '/seg_ros/live/lane_confidence', 10)
        self.bev_pub = self.create_publisher(Image, '/seg_ros/live/nav2/bev_debug', 10)

        label_qos = QoSProfile(depth=1)
        label_qos.reliability = ReliabilityPolicy.RELIABLE
        label_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.label_pub = self.create_publisher(
            LabelInfo, '/seg_ros/live/label_info', label_qos)

        nav2_qos = QoSProfile(depth=1)
        nav2_qos.reliability = ReliabilityPolicy.RELIABLE
        nav2_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.nav2_mask_pub = self.create_publisher(
            OccupancyGrid, '/seg_ros/live/nav2/filter_mask', nav2_qos)
        self.nav2_drivable_pub = self.create_publisher(
            OccupancyGrid, '/seg_ros/live/nav2/drivable_grid', nav2_qos)
        self.nav2_filter_info_pub = self.create_publisher(
            CostmapFilterInfo, '/seg_ros/live/nav2/costmap_filter_info', nav2_qos)

        self.detection_pub = self.create_publisher(
            String, '/seg_ros/live/detections', 10)
        self.timing_pub = self.create_publisher(
            String, '/seg_ros/live/timing', 10)

        self.image_sub = self.create_subscription(
            Image, self.image_topic, self._image_cb, 10)

        self._publish_label_info()
        self._publish_filter_info()
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
            filtered_drivable, keepout_mask, drivable_grid, bev_debug = (
                self._build_nav2_products(frame, da_mask, ll_mask, object_detections)
            )
        except Exception as exc:
            self.get_logger().warning(f'Live perception inference failed: {exc}')
            return

        combined_overlay = overlay.copy()
        for det in object_detections:
            self._draw_detection(combined_overlay, det)

        self._publish_images(
            msg,
            frame,
            combined_overlay,
            filtered_drivable,
            ll_mask,
            bev_debug,
        )
        self._publish_nav2_grids(msg, keepout_mask, drivable_grid)
        self._publish_detections(
            msg,
            seg_detections,
            object_detections,
            keepout_mask,
            drivable_grid,
            start,
        )
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

    def _build_nav2_products(self, frame, da_mask, ll_mask, object_detections):
        filtered_drivable = np.where(da_mask > 0, 255, 0).astype(np.uint8)

        # Remove detected dynamic / static obstacles from drivable space.
        for det in object_detections:
            x1, y1, x2, y2 = [int(round(v)) for v in det['xyxy']]
            x1 = max(0, min(frame.shape[1] - 1, x1))
            y1 = max(0, min(frame.shape[0] - 1, y1))
            x2 = max(0, min(frame.shape[1], x2))
            y2 = max(0, min(frame.shape[0], y2))
            if x2 > x1 and y2 > y1:
                cv2.rectangle(filtered_drivable, (x1, y1), (x2, y2), 0, thickness=-1)

        filtered_drivable = cv2.morphologyEx(
            filtered_drivable, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        lane_mask = np.where(ll_mask > 0, 255, 0).astype(np.uint8)

        transform = self._nav2_perspective_transform(frame.shape[1], frame.shape[0])
        drivable_grid = cv2.warpPerspective(
            filtered_drivable,
            transform,
            (self.nav2_grid_width_cells, self.nav2_grid_height_cells),
            flags=cv2.INTER_NEAREST,
        )
        lane_grid = cv2.warpPerspective(
            lane_mask,
            transform,
            (self.nav2_grid_width_cells, self.nav2_grid_height_cells),
            flags=cv2.INTER_NEAREST,
        )

        drivable_grid = cv2.morphologyEx(
            drivable_grid, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        drivable_grid = cv2.bitwise_or(drivable_grid, lane_grid)
        keepout_mask = np.where(drivable_grid > 0, 0, 100).astype(np.uint8)

        bev_debug = np.zeros((self.nav2_grid_height_cells, self.nav2_grid_width_cells, 3), dtype=np.uint8)
        bev_debug[drivable_grid > 0] = (255, 255, 255)
        bev_debug[lane_grid > 0] = (0, 255, 255)

        return filtered_drivable, keepout_mask, drivable_grid, bev_debug

    def _nav2_perspective_transform(self, width, height):
        src = np.float32([
            [width * self.src_bottom_left_x, height * self.src_bottom_y],
            [width * self.src_bottom_right_x, height * self.src_bottom_y],
            [width * self.src_top_right_x, height * self.src_top_y],
            [width * self.src_top_left_x, height * self.src_top_y],
        ])
        dst = np.float32([
            [0, self.nav2_grid_height_cells - 1],
            [self.nav2_grid_width_cells - 1, self.nav2_grid_height_cells - 1],
            [self.nav2_grid_width_cells - 1, 0],
            [0, 0],
        ])
        return cv2.getPerspectiveTransform(src, dst)

    def _publish_images(self, source_msg, frame, overlay, da_mask, ll_mask, bev_debug):
        if self.publish_input_image:
            input_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            input_msg.header = source_msg.header
            self.input_pub.publish(input_msg)

        overlay_msg = self.bridge.cv2_to_imgmsg(overlay, encoding='bgr8')
        overlay_msg.header = source_msg.header
        self.overlay_pub.publish(overlay_msg)

        drivable_msg = self.bridge.cv2_to_imgmsg(
            da_mask.astype(np.uint8), encoding='mono8')
        drivable_msg.header = source_msg.header
        self.drivable_pub.publish(drivable_msg)

        lane_mask = ll_mask.astype(np.uint8)
        lane_msg = self.bridge.cv2_to_imgmsg(lane_mask, encoding='mono8')
        lane_msg.header = source_msg.header
        self.lane_pub.publish(lane_msg)

        confidence_msg = self.bridge.cv2_to_imgmsg(lane_mask, encoding='mono8')
        confidence_msg.header = source_msg.header
        self.lane_confidence_pub.publish(confidence_msg)

        bev_msg = self.bridge.cv2_to_imgmsg(bev_debug, encoding='bgr8')
        bev_msg.header = source_msg.header
        self.bev_pub.publish(bev_msg)

    def _build_grid_message(self, source_msg, grid_data):
        grid = OccupancyGrid()
        grid.header.stamp = source_msg.header.stamp
        grid.header.frame_id = 'base_link'
        grid.info.resolution = self.nav2_grid_resolution
        grid.info.width = self.nav2_grid_width_cells
        grid.info.height = self.nav2_grid_height_cells
        grid.info.origin.position.x = self.nav2_x_min
        grid.info.origin.position.y = self.nav2_y_min
        grid.info.origin.position.z = 0.0
        grid.info.origin.orientation.w = 1.0
        grid.data = grid_data.flatten().tolist()
        return grid

    def _publish_nav2_grids(self, source_msg, keepout_mask, drivable_mask):
        if not self.nav2_publish_grid:
            return
        keepout_msg = self._build_grid_message(source_msg, keepout_mask.astype(np.int8))
        drivable_grid = np.where(drivable_mask > 0, 0, 100).astype(np.int8)
        drivable_msg = self._build_grid_message(source_msg, drivable_grid)
        self.nav2_mask_pub.publish(keepout_msg)
        self.nav2_drivable_pub.publish(drivable_msg)

    def _publish_detections(
        self,
        source_msg,
        seg_detections,
        object_detections,
        keepout_mask,
        drivable_grid,
        start,
    ):
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
            'nav2_grid': {
                'keepout_cells': int(np.count_nonzero(keepout_mask == 100)),
                'drivable_cells': int(np.count_nonzero(drivable_grid > 0)),
                'resolution': self.nav2_grid_resolution,
                'x_range': self.nav2_x_range,
                'y_range': self.nav2_y_range,
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

    def _publish_filter_info(self):
        if not self.nav2_publish_grid:
            return
        msg = CostmapFilterInfo()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.type = 0
        msg.filter_mask_topic = '/seg_ros/live/nav2/filter_mask'
        msg.base = 0.0
        msg.multiplier = 1.0
        self.nav2_filter_info_pub.publish(msg)

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
