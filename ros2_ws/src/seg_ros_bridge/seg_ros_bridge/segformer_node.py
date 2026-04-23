import json
import time

import cv2
import numpy as np
import rclpy
import torch
from cv_bridge import CvBridge
from nav2_msgs.msg import CostmapFilterInfo
from nav_msgs.msg import OccupancyGrid
from PIL import Image as PilImage
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String
from vision_msgs.msg import LabelInfo, VisionClass


DEFAULT_MODEL_ID = 'nvidia/segformer-b0-finetuned-cityscapes-512-1024'
DEFAULT_IMAGE_TOPIC = '/zed/zed_node/rgb/color/rect/image'


class SegFormerNode(Node):
    """Optional SegFormer semantic segmentation backend for comparison testing."""

    def __init__(self):
        super().__init__('segformer_node')

        self.declare_parameter('image_topic', DEFAULT_IMAGE_TOPIC)
        self.declare_parameter('model_id', DEFAULT_MODEL_ID)
        self.declare_parameter('device', 'cpu')
        self.declare_parameter('process_every_n', 1)
        self.declare_parameter('publish_input_image', True)
        self.declare_parameter('publish_timing', True)
        self.declare_parameter('enable_hsv_refinement', True)
        self.declare_parameter('nav2_publish_grid', True)
        self.declare_parameter('nav2_grid_resolution', 0.05)
        self.declare_parameter('nav2_x_range', [0.0, 15.0])
        self.declare_parameter('nav2_y_range', [-10.0, 10.0])
        self.declare_parameter('camera_mount_x', 0.35)
        self.declare_parameter('camera_mount_y', 0.0)
        self.declare_parameter('camera_mount_z', 0.75)
        self.declare_parameter('camera_mount_yaw', 0.0)
        self.declare_parameter('lane_detect_on_threshold', 0.30)
        self.declare_parameter('lane_detect_off_threshold', 0.18)
        self.declare_parameter('lane_corridor_target_cells', 220)
        self.declare_parameter('lane_bev_target_pixels', 140)
        self.declare_parameter('nav2_src_bottom_y', 0.98)
        self.declare_parameter('nav2_src_top_y', 0.62)
        self.declare_parameter('nav2_src_bottom_left_x', 0.05)
        self.declare_parameter('nav2_src_bottom_right_x', 0.95)
        self.declare_parameter('nav2_src_top_left_x', 0.35)
        self.declare_parameter('nav2_src_top_right_x', 0.65)
        self.declare_parameter('enable_temporal_smoothing', True)
        self.declare_parameter('temporal_alpha', 0.65)

        self.image_topic = self.get_parameter('image_topic').value
        self.model_id = self.get_parameter('model_id').value
        self.device_name = self.get_parameter('device').value
        self.process_every_n = max(1, int(self.get_parameter('process_every_n').value))
        self.publish_input_image = bool(self.get_parameter('publish_input_image').value)
        self.publish_timing = bool(self.get_parameter('publish_timing').value)
        self.enable_hsv_refinement = bool(
            self.get_parameter('enable_hsv_refinement').value)
        self.nav2_publish_grid = bool(self.get_parameter('nav2_publish_grid').value)
        self.nav2_grid_resolution = float(self.get_parameter('nav2_grid_resolution').value)
        self.nav2_x_range = [float(v) for v in self.get_parameter('nav2_x_range').value]
        self.nav2_y_range = [float(v) for v in self.get_parameter('nav2_y_range').value]
        self.camera_mount_x = float(self.get_parameter('camera_mount_x').value)
        self.camera_mount_y = float(self.get_parameter('camera_mount_y').value)
        self.camera_mount_z = float(self.get_parameter('camera_mount_z').value)
        self.camera_mount_yaw = float(self.get_parameter('camera_mount_yaw').value)
        self.lane_detect_on_threshold = float(
            self.get_parameter('lane_detect_on_threshold').value)
        self.lane_detect_off_threshold = float(
            self.get_parameter('lane_detect_off_threshold').value)
        self.lane_corridor_target_cells = float(
            self.get_parameter('lane_corridor_target_cells').value)
        self.lane_bev_target_pixels = float(
            self.get_parameter('lane_bev_target_pixels').value)
        self.nav2_src_bottom_y = float(self.get_parameter('nav2_src_bottom_y').value)
        self.nav2_src_top_y = float(self.get_parameter('nav2_src_top_y').value)
        self.nav2_src_bottom_left_x = float(
            self.get_parameter('nav2_src_bottom_left_x').value)
        self.nav2_src_bottom_right_x = float(
            self.get_parameter('nav2_src_bottom_right_x').value)
        self.nav2_src_top_left_x = float(self.get_parameter('nav2_src_top_left_x').value)
        self.nav2_src_top_right_x = float(self.get_parameter('nav2_src_top_right_x').value)
        self.enable_temporal_smoothing = bool(
            self.get_parameter('enable_temporal_smoothing').value)
        self.temporal_alpha = float(self.get_parameter('temporal_alpha').value)
        self.nav2_x_min, self.nav2_x_max = self.nav2_x_range
        self.nav2_y_min, self.nav2_y_max = self.nav2_y_range
        self.nav2_grid_width_m = self.nav2_y_max - self.nav2_y_min
        self.nav2_grid_length_m = self.nav2_x_max - self.nav2_x_min
        self.nav2_grid_width_cells = max(
            1, int(round(self.nav2_grid_width_m / self.nav2_grid_resolution)))
        self.nav2_grid_height_cells = max(
            1, int(round(self.nav2_grid_length_m / self.nav2_grid_resolution)))

        self.bridge = CvBridge()
        self.frame_count = 0
        self.prev_lane_corridor = None
        self.prev_nav2_drivable = None
        self.lane_detected = False
        self.device = self._resolve_device(self.device_name)
        self._load_model()

        self.input_pub = self.create_publisher(
            Image, '/seg_ros/segformer/input_image', 10)
        self.overlay_pub = self.create_publisher(
            Image, '/seg_ros/segformer/overlay_image', 10)
        self.class_mask_pub = self.create_publisher(
            Image, '/seg_ros/segformer/class_mask', 10)
        self.road_mask_raw_pub = self.create_publisher(
            Image, '/seg_ros/segformer/road_mask_raw', 10)
        self.road_mask_pub = self.create_publisher(
            Image, '/seg_ros/segformer/road_mask', 10)
        self.sidewalk_mask_pub = self.create_publisher(
            Image, '/seg_ros/segformer/sidewalk_mask', 10)
        self.lane_hint_pub = self.create_publisher(
            Image, '/seg_ros/segformer/lane_hint_mask', 10)
        self.igvc_white_mask_pub = self.create_publisher(
            Image, '/seg_ros/segformer/igvc_white_mask', 10)
        self.igvc_lane_bev_pub = self.create_publisher(
            Image, '/seg_ros/segformer/igvc_lane_bev', 10)
        self.igvc_lane_corridor_pub = self.create_publisher(
            Image, '/seg_ros/segformer/igvc_lane_corridor_mask', 10)
        self.bev_mask_pub = self.create_publisher(
            Image, '/seg_ros/segformer/nav2/bev_keepout_mask', 10)
        self.metadata_pub = self.create_publisher(
            String, '/seg_ros/segformer/metadata', 10)
        self.timing_pub = self.create_publisher(
            String, '/seg_ros/segformer/timing', 10)
        self.lane_detected_pub = self.create_publisher(
            Bool, '/seg_ros/segformer/lane_detected', 10)
        self.mode_hint_pub = self.create_publisher(
            String, '/seg_ros/segformer/planner_mode_hint', 10)

        label_qos = QoSProfile(depth=1)
        label_qos.reliability = ReliabilityPolicy.RELIABLE
        label_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.label_pub = self.create_publisher(
            LabelInfo, '/seg_ros/segformer/label_info', label_qos)
        nav2_qos = QoSProfile(depth=1)
        nav2_qos.reliability = ReliabilityPolicy.RELIABLE
        nav2_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.nav2_mask_pub = self.create_publisher(
            OccupancyGrid, '/seg_ros/segformer/nav2/filter_mask', nav2_qos)
        self.nav2_drivable_pub = self.create_publisher(
            OccupancyGrid, '/seg_ros/segformer/nav2/drivable_grid', nav2_qos)
        self.nav2_filter_info_pub = self.create_publisher(
            CostmapFilterInfo,
            '/seg_ros/segformer/nav2/costmap_filter_info',
            nav2_qos,
        )

        self.sub = self.create_subscription(Image, self.image_topic, self._image_cb, 10)
        self._publish_label_info()
        self._publish_filter_info()
        self.get_logger().info(f'Loaded SegFormer model: {self.model_id}')
        self.get_logger().info(f'Subscribed to image topic: {self.image_topic}')

    def _resolve_device(self, value):
        if value.startswith('cuda') and not torch.cuda.is_available():
            self.get_logger().warning('CUDA requested but unavailable; falling back to CPU')
            return torch.device('cpu')
        return torch.device(value)

    def _load_model(self):
        try:
            from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
        except ImportError as exc:
            raise RuntimeError(
                'SegFormer support requires the optional transformers dependency. '
                'Install it in the runtime environment with: '
                '/home/alexander/github/av-perception/.venv/bin/python -m pip install transformers'
            ) from exc

        self.processor = SegformerImageProcessor.from_pretrained(self.model_id)
        self.model = SegformerForSemanticSegmentation.from_pretrained(self.model_id)
        self.model.to(self.device)
        self.model.eval()
        self.id2label = {
            int(k): v for k, v in self.model.config.id2label.items()
        }
        self.label2id = {
            label.lower(): idx for idx, label in self.id2label.items()
        }

    def _image_cb(self, msg):
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
            class_mask = self._infer(frame)
        except Exception as exc:
            self.get_logger().warning(f'SegFormer inference failed: {exc}')
            return

        road_mask_raw = self._binary_mask(class_mask, 'road')
        sidewalk_mask = self._binary_mask(class_mask, 'sidewalk')
        road_mask, lane_hint_mask = self._refine_masks(
            frame, road_mask_raw, sidewalk_mask)
        igvc_white_mask, igvc_lane_bev, igvc_lane_corridor = self._extract_igvc_lane_features(
            frame,
            road_mask,
            lane_hint_mask,
            frame.shape[1],
            frame.shape[0],
        )
        igvc_lane_corridor = self._smooth_binary_mask(
            igvc_lane_corridor, 'lane_corridor')
        lane_confidence, lane_detected = self._estimate_lane_confidence(
            igvc_lane_bev, igvc_lane_corridor)
        igvc_lane_corridor_image = self._project_bev_to_image(
            igvc_lane_corridor, frame.shape[1], frame.shape[0])
        overlay = self._make_overlay(
            frame,
            class_mask,
            road_mask,
            lane_hint_mask,
            igvc_lane_corridor_image,
        )
        nav2_keepout_mask, nav2_drivable_mask = self._project_nav2_grids(
            road_mask,
            lane_hint_mask,
            igvc_lane_corridor,
            lane_detected,
            frame.shape[1],
            frame.shape[0],
        )
        nav2_drivable_mask = self._smooth_binary_mask(
            nav2_drivable_mask, 'nav2_drivable')
        nav2_keepout_mask = np.where(nav2_drivable_mask > 0, 0, 100).astype(np.uint8)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        self._publish_images(
            msg,
            frame,
            overlay,
            class_mask,
            road_mask_raw,
            road_mask,
            sidewalk_mask,
            lane_hint_mask,
            igvc_white_mask,
            igvc_lane_bev,
            igvc_lane_corridor,
            nav2_keepout_mask,
        )
        self._publish_nav2_grids(msg, nav2_keepout_mask, nav2_drivable_mask)
        self._publish_lane_state(lane_detected)
        self._publish_metadata(
            msg,
            class_mask,
            road_mask_raw,
            road_mask,
            sidewalk_mask,
            lane_hint_mask,
            igvc_white_mask,
            igvc_lane_bev,
            igvc_lane_corridor,
            nav2_keepout_mask,
            lane_confidence,
            lane_detected,
            elapsed_ms,
        )

    def _infer(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image = PilImage.fromarray(frame_rgb)
        inputs = self.processor(images=pil_image, return_tensors='pt')
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            logits = torch.nn.functional.interpolate(
                logits,
                size=frame_bgr.shape[:2],
                mode='bilinear',
                align_corners=False,
            )
            class_mask = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        return class_mask

    def _binary_mask(self, class_mask, class_name):
        class_id = self.label2id.get(class_name)
        if class_id is None:
            return np.zeros(class_mask.shape, dtype=np.uint8)
        return (class_mask == class_id).astype(np.uint8) * 255

    def _refine_masks(self, frame, road_mask_raw, sidewalk_mask):
        if not self.enable_hsv_refinement:
            return road_mask_raw, np.zeros_like(road_mask_raw)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        height, width = road_mask_raw.shape
        roi = np.zeros((height, width), dtype=np.uint8)
        roi[int(height * 0.35):, :] = 255

        asphalt_mask = cv2.inRange(hsv, (0, 0, 35), (179, 80, 185))
        white_mask = cv2.inRange(hsv, (0, 0, 180), (179, 55, 255))
        yellow_mask = cv2.inRange(hsv, (12, 55, 110), (42, 255, 255))
        lane_hint_mask = cv2.bitwise_or(white_mask, yellow_mask)

        kernel_large = np.ones((21, 21), np.uint8)
        kernel_small = np.ones((5, 5), np.uint8)
        road_neighborhood = cv2.dilate(road_mask_raw, kernel_large, iterations=1)

        asphalt_support = cv2.bitwise_and(asphalt_mask, road_neighborhood)
        asphalt_support = cv2.bitwise_and(asphalt_support, roi)
        asphalt_support = cv2.bitwise_and(
            asphalt_support,
            cv2.bitwise_not(sidewalk_mask),
        )

        road_mask = cv2.bitwise_or(road_mask_raw, asphalt_support)
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel_large)
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, kernel_small)

        lane_hint_mask = cv2.bitwise_and(lane_hint_mask, roi)
        lane_hint_mask = cv2.bitwise_and(
            lane_hint_mask,
            cv2.dilate(road_mask, kernel_large, iterations=1),
        )
        lane_hint_mask = cv2.morphologyEx(
            lane_hint_mask,
            cv2.MORPH_OPEN,
            np.ones((3, 3), np.uint8),
        )
        lane_hint_mask = cv2.dilate(lane_hint_mask, kernel_small, iterations=1)

        return road_mask, lane_hint_mask

    def _make_overlay(
        self,
        frame,
        class_mask,
        road_mask,
        lane_hint_mask,
        igvc_lane_corridor_image,
    ):
        overlay = frame.copy()
        colors = {
            'road': (70, 70, 70),
            'sidewalk': (120, 120, 120),
            'person': (0, 220, 255),
            'car': (255, 130, 0),
            'truck': (255, 80, 0),
            'bus': (255, 80, 80),
            'traffic light': (60, 220, 60),
            'traffic sign': (30, 180, 30),
        }
        color_mask = np.zeros_like(frame)
        for label, color in colors.items():
            class_id = self.label2id.get(label)
            if class_id is not None:
                color_mask[class_mask == class_id] = color
        color_mask[road_mask > 0] = (80, 80, 80)
        color_mask[lane_hint_mask > 0] = (0, 255, 255)
        color_mask[igvc_lane_corridor_image > 0] = (255, 255, 0)
        active = np.any(color_mask != 0, axis=2)
        overlay[active] = cv2.addWeighted(frame, 0.55, color_mask, 0.45, 0)[active]
        return overlay

    def _extract_igvc_lane_features(self, frame, road_mask, lane_hint_mask, width, height):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        roi = np.zeros((height, width), dtype=np.uint8)
        roi[int(height * 0.45):, :] = 255
        white_mask = cv2.inRange(hsv, (0, 0, 165), (179, 70, 255))
        white_mask = cv2.bitwise_and(white_mask, roi)
        road_support = cv2.dilate(road_mask, np.ones((25, 25), np.uint8), iterations=1)
        white_mask = cv2.bitwise_and(white_mask, road_support)
        white_mask = cv2.bitwise_or(white_mask, lane_hint_mask)
        white_mask = cv2.morphologyEx(
            white_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        white_mask = cv2.dilate(white_mask, np.ones((3, 3), np.uint8), iterations=1)

        transform = self._nav2_perspective_transform(width, height)
        white_bev = cv2.warpPerspective(
            white_mask,
            transform,
            (self.nav2_grid_width_cells, self.nav2_grid_height_cells),
            flags=cv2.INTER_NEAREST,
        )
        white_bev = cv2.morphologyEx(
            white_bev, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        white_bev = cv2.dilate(white_bev, np.ones((3, 3), np.uint8), iterations=1)
        left_boundary, right_boundary = self._select_lane_boundaries(white_bev)
        corridor = self._build_lane_corridor(left_boundary, right_boundary)
        if np.count_nonzero(corridor) == 0:
            corridor = white_bev.copy()
        return white_mask, white_bev, corridor

    def _nav2_perspective_transform(self, width, height):
        src = np.float32([
            [width * self.nav2_src_bottom_left_x, height * self.nav2_src_bottom_y],
            [width * self.nav2_src_bottom_right_x, height * self.nav2_src_bottom_y],
            [width * self.nav2_src_top_right_x, height * self.nav2_src_top_y],
            [width * self.nav2_src_top_left_x, height * self.nav2_src_top_y],
        ])
        dst = np.float32([
            [0, self.nav2_grid_height_cells - 1],
            [self.nav2_grid_width_cells - 1, self.nav2_grid_height_cells - 1],
            [self.nav2_grid_width_cells - 1, 0],
            [0, 0],
        ])
        return cv2.getPerspectiveTransform(src, dst)

    def _project_bev_to_image(self, bev_mask, width, height):
        transform = self._nav2_perspective_transform(width, height)
        inverse = np.linalg.inv(transform)
        return cv2.warpPerspective(
            bev_mask,
            inverse,
            (width, height),
            flags=cv2.INTER_NEAREST,
        )

    def _select_lane_boundaries(self, white_bev):
        left_half = white_bev[:, : self.nav2_grid_width_cells // 2]
        right_half = white_bev[:, self.nav2_grid_width_cells // 2 :]
        left_boundary = self._largest_lane_component(left_half)
        right_boundary = self._largest_lane_component(right_half)
        right_full = np.zeros_like(white_bev)
        left_full = np.zeros_like(white_bev)
        left_full[:, : self.nav2_grid_width_cells // 2] = left_boundary
        right_full[:, self.nav2_grid_width_cells // 2 :] = right_boundary
        return left_full, right_full

    def _largest_lane_component(self, mask):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels <= 1:
            return np.zeros_like(mask)
        best_index = -1
        best_score = 0
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            width = stats[label, cv2.CC_STAT_WIDTH]
            height = stats[label, cv2.CC_STAT_HEIGHT]
            if area < 8 or height < 6:
                continue
            score = area + (height * 2) - width
            if score > best_score:
                best_score = score
                best_index = label
        if best_index == -1:
            return np.zeros_like(mask)
        return np.where(labels == best_index, 255, 0).astype(np.uint8)

    def _build_lane_corridor(self, left_boundary, right_boundary):
        corridor = np.zeros_like(left_boundary)
        for row in range(self.nav2_grid_height_cells):
            left_cols = np.flatnonzero(left_boundary[row] > 0)
            right_cols = np.flatnonzero(right_boundary[row] > 0)
            if left_cols.size == 0 or right_cols.size == 0:
                continue
            left_x = int(np.max(left_cols))
            right_x = int(np.min(right_cols))
            if right_x <= left_x:
                continue
            corridor[row, left_x:right_x + 1] = 255
        corridor = cv2.morphologyEx(
            corridor, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        return corridor

    def _estimate_lane_confidence(self, igvc_lane_bev, igvc_lane_corridor):
        corridor_cells = float(np.count_nonzero(igvc_lane_corridor))
        bev_pixels = float(np.count_nonzero(igvc_lane_bev))
        corridor_score = min(1.0, corridor_cells / max(1.0, self.lane_corridor_target_cells))
        bev_score = min(1.0, bev_pixels / max(1.0, self.lane_bev_target_pixels))
        confidence = (0.7 * corridor_score) + (0.3 * bev_score)
        if self.lane_detected:
            self.lane_detected = confidence >= self.lane_detect_off_threshold
        else:
            self.lane_detected = confidence >= self.lane_detect_on_threshold
        return confidence, self.lane_detected

    def _smooth_binary_mask(self, mask, kind):
        if not self.enable_temporal_smoothing:
            return mask
        alpha = min(max(self.temporal_alpha, 0.0), 0.98)
        mask_f = (mask > 0).astype(np.float32)
        if kind == 'lane_corridor':
            prev = self.prev_lane_corridor
        else:
            prev = self.prev_nav2_drivable
        if prev is None:
            smoothed = mask_f
        else:
            smoothed = (alpha * prev) + ((1.0 - alpha) * mask_f)
        out = np.where(smoothed >= 0.45, 255, 0).astype(np.uint8)
        if kind == 'lane_corridor':
            self.prev_lane_corridor = smoothed
        else:
            self.prev_nav2_drivable = smoothed
        return out

    def _project_nav2_grids(
        self,
        road_mask,
        lane_hint_mask,
        igvc_lane_corridor,
        lane_detected,
        width,
        height,
    ):
        if not self.nav2_publish_grid:
            empty = np.zeros((self.nav2_grid_height_cells, self.nav2_grid_width_cells), dtype=np.uint8)
            return empty, empty

        transform = self._nav2_perspective_transform(width, height)
        road_bev = cv2.warpPerspective(
            road_mask,
            transform,
            (self.nav2_grid_width_cells, self.nav2_grid_height_cells),
            flags=cv2.INTER_NEAREST,
        )
        lane_bev = cv2.warpPerspective(
            lane_hint_mask,
            transform,
            (self.nav2_grid_width_cells, self.nav2_grid_height_cells),
            flags=cv2.INTER_NEAREST,
        )
        road_bev = cv2.morphologyEx(
            road_bev,
            cv2.MORPH_CLOSE,
            np.ones((5, 5), np.uint8),
        )
        drivable = np.where(road_bev > 0, 255, 0).astype(np.uint8)
        if lane_detected and np.count_nonzero(igvc_lane_corridor) > 0:
            drivable = cv2.bitwise_and(drivable, cv2.dilate(
                igvc_lane_corridor, np.ones((9, 9), np.uint8), iterations=1))
        drivable = cv2.bitwise_or(drivable, lane_bev)
        drivable = cv2.morphologyEx(
            drivable, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        keepout = np.where(drivable > 0, 0, 100).astype(np.uint8)
        return keepout, drivable

    def _publish_images(
        self,
        source_msg,
        frame,
        overlay,
        class_mask,
        road_mask_raw,
        road_mask,
        sidewalk_mask,
        lane_hint_mask,
        igvc_white_mask,
        igvc_lane_bev,
        igvc_lane_corridor,
        nav2_keepout_mask,
    ):
        if self.publish_input_image:
            input_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            input_msg.header = source_msg.header
            self.input_pub.publish(input_msg)

        overlay_msg = self.bridge.cv2_to_imgmsg(overlay, encoding='bgr8')
        overlay_msg.header = source_msg.header
        self.overlay_pub.publish(overlay_msg)

        class_msg = self.bridge.cv2_to_imgmsg(class_mask, encoding='mono8')
        class_msg.header = source_msg.header
        self.class_mask_pub.publish(class_msg)

        road_raw_msg = self.bridge.cv2_to_imgmsg(road_mask_raw, encoding='mono8')
        road_raw_msg.header = source_msg.header
        self.road_mask_raw_pub.publish(road_raw_msg)

        road_msg = self.bridge.cv2_to_imgmsg(road_mask, encoding='mono8')
        road_msg.header = source_msg.header
        self.road_mask_pub.publish(road_msg)

        sidewalk_msg = self.bridge.cv2_to_imgmsg(sidewalk_mask, encoding='mono8')
        sidewalk_msg.header = source_msg.header
        self.sidewalk_mask_pub.publish(sidewalk_msg)

        lane_hint_msg = self.bridge.cv2_to_imgmsg(lane_hint_mask, encoding='mono8')
        lane_hint_msg.header = source_msg.header
        self.lane_hint_pub.publish(lane_hint_msg)

        igvc_white_msg = self.bridge.cv2_to_imgmsg(igvc_white_mask, encoding='mono8')
        igvc_white_msg.header = source_msg.header
        self.igvc_white_mask_pub.publish(igvc_white_msg)

        igvc_bev_msg = self.bridge.cv2_to_imgmsg(igvc_lane_bev, encoding='mono8')
        igvc_bev_msg.header = source_msg.header
        self.igvc_lane_bev_pub.publish(igvc_bev_msg)

        igvc_corridor_msg = self.bridge.cv2_to_imgmsg(
            igvc_lane_corridor, encoding='mono8')
        igvc_corridor_msg.header = source_msg.header
        self.igvc_lane_corridor_pub.publish(igvc_corridor_msg)

        bev_msg = self.bridge.cv2_to_imgmsg(nav2_keepout_mask, encoding='mono8')
        bev_msg.header = source_msg.header
        self.bev_mask_pub.publish(bev_msg)

    def _publish_nav2_grids(self, source_msg, keepout_mask, drivable_mask):
        if not self.nav2_publish_grid:
            return
        keepout_msg = self._build_grid_message(source_msg, keepout_mask.astype(np.int8))
        drivable_grid = np.where(drivable_mask > 0, 0, 100).astype(np.int8)
        drivable_msg = self._build_grid_message(source_msg, drivable_grid)
        self.nav2_mask_pub.publish(keepout_msg)
        self.nav2_drivable_pub.publish(drivable_msg)

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

    def _publish_lane_state(self, lane_detected):
        lane_msg = Bool()
        lane_msg.data = bool(lane_detected)
        self.lane_detected_pub.publish(lane_msg)

        mode_msg = String()
        mode_msg.data = 'lane_following' if lane_detected else 'obstacle_avoidance'
        self.mode_hint_pub.publish(mode_msg)

    def _publish_metadata(
        self,
        source_msg,
        class_mask,
        road_mask_raw,
        road_mask,
        sidewalk_mask,
        lane_hint_mask,
        igvc_white_mask,
        igvc_lane_bev,
        igvc_lane_corridor,
        nav2_keepout_mask,
        lane_confidence,
        lane_detected,
        elapsed_ms,
    ):
        counts = {}
        unique, pixels = np.unique(class_mask, return_counts=True)
        for class_id, count in zip(unique.tolist(), pixels.tolist()):
            label = self.id2label.get(int(class_id), str(class_id))
            counts[label] = int(count)

        payload = {
            'header': {
                'stamp': {
                    'sec': source_msg.header.stamp.sec,
                    'nanosec': source_msg.header.stamp.nanosec,
                },
                'frame_id': source_msg.header.frame_id,
            },
            'model_id': self.model_id,
            'hsv_refinement_enabled': self.enable_hsv_refinement,
            'temporal_smoothing_enabled': self.enable_temporal_smoothing,
            'temporal_alpha': self.temporal_alpha,
            'road_pixels_raw': int(np.count_nonzero(road_mask_raw)),
            'road_pixels': int(np.count_nonzero(road_mask)),
            'sidewalk_pixels': int(np.count_nonzero(sidewalk_mask)),
            'lane_hint_pixels': int(np.count_nonzero(lane_hint_mask)),
            'igvc_white_pixels': int(np.count_nonzero(igvc_white_mask)),
            'igvc_lane_bev_pixels': int(np.count_nonzero(igvc_lane_bev)),
            'igvc_lane_corridor_cells': int(np.count_nonzero(igvc_lane_corridor)),
            'lane_confidence': lane_confidence,
            'lane_detected': bool(lane_detected),
            'planner_mode_hint': 'lane_following' if lane_detected else 'obstacle_avoidance',
            'nav2_keepout_cells': int(np.count_nonzero(nav2_keepout_mask == 100)),
            'nav2_grid': {
                'resolution': self.nav2_grid_resolution,
                'x_range': self.nav2_x_range,
                'y_range': self.nav2_y_range,
                'width_m': self.nav2_grid_width_m,
                'length_m': self.nav2_grid_length_m,
            },
            'camera_prior': {
                'mount_x': self.camera_mount_x,
                'mount_y': self.camera_mount_y,
                'mount_z': self.camera_mount_z,
                'mount_yaw': self.camera_mount_yaw,
            },
            'class_pixel_counts': counts,
            'timing_ms': elapsed_ms,
        }
        metadata_msg = String()
        metadata_msg.data = json.dumps(payload)
        self.metadata_pub.publish(metadata_msg)

        if self.publish_timing:
            timing_msg = String()
            timing_msg.data = json.dumps({
                'model_id': self.model_id,
                'frame_id': source_msg.header.frame_id,
                'timing_ms': elapsed_ms,
                'process_every_n': self.process_every_n,
            })
            self.timing_pub.publish(timing_msg)

    def _publish_label_info(self):
        msg = LabelInfo()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_color_optical_frame'
        msg.threshold = 0.0
        msg.class_map = [
            VisionClass(class_id=class_id, class_name=label)
            for class_id, label in sorted(self.id2label.items())
        ]
        self.label_pub.publish(msg)

    def _publish_filter_info(self):
        if not self.nav2_publish_grid:
            return
        msg = CostmapFilterInfo()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.type = 0
        msg.filter_mask_topic = '/seg_ros/segformer/nav2/filter_mask'
        msg.base = 0.0
        msg.multiplier = 1.0
        self.nav2_filter_info_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = SegFormerNode()
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
