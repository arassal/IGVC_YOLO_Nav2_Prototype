import json
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from nav2_msgs.msg import CostmapFilterInfo
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String


DEFAULT_IMAGE_TOPIC = '/zed/zed_node/rgb/color/rect/image'


class IgvcBevNode(Node):
    def __init__(self):
        super().__init__('igvc_bev_node')

        self.declare_parameter('image_topic', DEFAULT_IMAGE_TOPIC)
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
        self.declare_parameter('enable_temporal_smoothing', True)
        self.declare_parameter('temporal_alpha', 0.65)
        self.declare_parameter('lane_detect_on_threshold', 0.30)
        self.declare_parameter('lane_detect_off_threshold', 0.18)
        self.declare_parameter('lane_corridor_target_cells', 220)
        self.declare_parameter('lane_bev_target_pixels', 140)
        self.declare_parameter('roi_start_ratio', 0.45)
        self.declare_parameter('asphalt_hsv_low', [0, 0, 30])
        self.declare_parameter('asphalt_hsv_high', [179, 90, 190])
        self.declare_parameter('white_hsv_low', [0, 0, 165])
        self.declare_parameter('white_hsv_high', [179, 70, 255])
        self.declare_parameter('yellow_hsv_low', [12, 50, 105])
        self.declare_parameter('yellow_hsv_high', [42, 255, 255])

        self.image_topic = self.get_parameter('image_topic').value
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
        self.enable_temporal_smoothing = bool(
            self.get_parameter('enable_temporal_smoothing').value)
        self.temporal_alpha = float(self.get_parameter('temporal_alpha').value)
        self.lane_detect_on_threshold = float(
            self.get_parameter('lane_detect_on_threshold').value)
        self.lane_detect_off_threshold = float(
            self.get_parameter('lane_detect_off_threshold').value)
        self.lane_corridor_target_cells = float(
            self.get_parameter('lane_corridor_target_cells').value)
        self.lane_bev_target_pixels = float(
            self.get_parameter('lane_bev_target_pixels').value)
        self.roi_start_ratio = float(self.get_parameter('roi_start_ratio').value)
        self.asphalt_hsv_low = tuple(
            int(v) for v in self.get_parameter('asphalt_hsv_low').value)
        self.asphalt_hsv_high = tuple(
            int(v) for v in self.get_parameter('asphalt_hsv_high').value)
        self.white_hsv_low = tuple(
            int(v) for v in self.get_parameter('white_hsv_low').value)
        self.white_hsv_high = tuple(
            int(v) for v in self.get_parameter('white_hsv_high').value)
        self.yellow_hsv_low = tuple(
            int(v) for v in self.get_parameter('yellow_hsv_low').value)
        self.yellow_hsv_high = tuple(
            int(v) for v in self.get_parameter('yellow_hsv_high').value)

        self.nav2_x_min, self.nav2_x_max = self.nav2_x_range
        self.nav2_y_min, self.nav2_y_max = self.nav2_y_range
        self.nav2_grid_width_m = self.nav2_y_max - self.nav2_y_min
        self.nav2_grid_length_m = self.nav2_x_max - self.nav2_x_min
        self.nav2_grid_width_cells = max(
            1, int(round(self.nav2_grid_width_m / self.nav2_grid_resolution)))
        self.nav2_grid_height_cells = max(
            1, int(round(self.nav2_grid_length_m / self.nav2_grid_resolution)))

        self.bridge = CvBridge()
        self.prev_lane_corridor = None
        self.prev_nav2_drivable = None
        self.lane_detected = False

        self.input_pub = self.create_publisher(Image, '/igvc_bev/input_image', 10)
        self.overlay_pub = self.create_publisher(Image, '/igvc_bev/overlay_image', 10)
        self.asphalt_pub = self.create_publisher(Image, '/igvc_bev/asphalt_mask', 10)
        self.white_pub = self.create_publisher(Image, '/igvc_bev/white_mask', 10)
        self.yellow_pub = self.create_publisher(Image, '/igvc_bev/yellow_mask', 10)
        self.lane_hint_pub = self.create_publisher(Image, '/igvc_bev/lane_hint_mask', 10)
        self.lane_bev_pub = self.create_publisher(Image, '/igvc_bev/lane_bev', 10)
        self.corridor_pub = self.create_publisher(Image, '/igvc_bev/lane_corridor_mask', 10)
        self.road_bev_pub = self.create_publisher(Image, '/igvc_bev/road_bev', 10)
        self.bev_mask_pub = self.create_publisher(Image, '/igvc_bev/nav2/bev_keepout_mask', 10)
        self.metadata_pub = self.create_publisher(String, '/igvc_bev/metadata', 10)
        self.timing_pub = self.create_publisher(String, '/igvc_bev/timing', 10)
        self.lane_detected_pub = self.create_publisher(Bool, '/igvc_bev/lane_detected', 10)
        self.mode_hint_pub = self.create_publisher(String, '/igvc_bev/planner_mode_hint', 10)

        nav2_qos = QoSProfile(depth=1)
        nav2_qos.reliability = ReliabilityPolicy.RELIABLE
        nav2_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.nav2_mask_pub = self.create_publisher(
            OccupancyGrid, '/igvc_bev/nav2/filter_mask', nav2_qos)
        self.nav2_drivable_pub = self.create_publisher(
            OccupancyGrid, '/igvc_bev/nav2/drivable_grid', nav2_qos)
        self.nav2_filter_info_pub = self.create_publisher(
            CostmapFilterInfo, '/igvc_bev/nav2/costmap_filter_info', nav2_qos)

        self.sub = self.create_subscription(Image, self.image_topic, self._image_cb, 10)
        self._publish_filter_info()
        self.get_logger().info(f'Subscribed to image topic: {self.image_topic}')

    def _image_cb(self, msg):
        start = time.perf_counter()
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().warning(f'Failed to convert input image: {exc}')
            return

        masks = self._extract_lane_products(frame)
        lane_corridor = self._smooth_binary_mask(masks['lane_corridor'], 'lane_corridor')
        lane_confidence, lane_detected = self._estimate_lane_confidence(
            masks['lane_bev'], lane_corridor)

        road_bev, nav2_keepout_mask, nav2_drivable_mask = self._project_nav2_grids(
            masks['road_mask'],
            masks['lane_hint_mask'],
            lane_corridor,
            lane_detected,
            frame.shape[1],
            frame.shape[0],
        )
        nav2_drivable_mask = self._smooth_binary_mask(nav2_drivable_mask, 'nav2_drivable')
        nav2_keepout_mask = np.where(nav2_drivable_mask > 0, 0, 100).astype(np.uint8)
        corridor_image = self._project_bev_to_image(
            lane_corridor, frame.shape[1], frame.shape[0])
        overlay = self._make_overlay(frame, masks, corridor_image)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        self._publish_images(
            msg,
            frame,
            overlay,
            masks['asphalt_mask'],
            masks['white_mask'],
            masks['yellow_mask'],
            masks['lane_hint_mask'],
            masks['lane_bev'],
            lane_corridor,
            road_bev,
            nav2_keepout_mask,
        )
        self._publish_nav2_grids(msg, nav2_keepout_mask, nav2_drivable_mask)
        self._publish_lane_state(lane_detected)
        self._publish_metadata(
            msg,
            masks,
            road_bev,
            lane_corridor,
            nav2_keepout_mask,
            lane_confidence,
            lane_detected,
            elapsed_ms,
        )

    def _extract_lane_products(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        height, width = frame.shape[:2]
        roi = np.zeros((height, width), dtype=np.uint8)
        roi[int(height * self.roi_start_ratio):, :] = 255

        asphalt_mask = cv2.inRange(hsv, self.asphalt_hsv_low, self.asphalt_hsv_high)
        white_mask = cv2.inRange(hsv, self.white_hsv_low, self.white_hsv_high)
        yellow_mask = cv2.inRange(hsv, self.yellow_hsv_low, self.yellow_hsv_high)

        asphalt_mask = cv2.bitwise_and(asphalt_mask, roi)
        white_mask = cv2.bitwise_and(white_mask, roi)
        yellow_mask = cv2.bitwise_and(yellow_mask, roi)

        kernel_large = np.ones((21, 21), np.uint8)
        kernel_medium = np.ones((7, 7), np.uint8)
        kernel_small = np.ones((3, 3), np.uint8)

        asphalt_mask = cv2.morphologyEx(asphalt_mask, cv2.MORPH_CLOSE, kernel_large)
        asphalt_mask = cv2.morphologyEx(asphalt_mask, cv2.MORPH_OPEN, kernel_medium)

        lane_hint_mask = cv2.bitwise_or(white_mask, yellow_mask)
        lane_hint_mask = cv2.bitwise_and(
            lane_hint_mask, cv2.dilate(asphalt_mask, kernel_large, iterations=1))
        lane_hint_mask = cv2.morphologyEx(lane_hint_mask, cv2.MORPH_OPEN, kernel_small)
        lane_hint_mask = cv2.dilate(lane_hint_mask, kernel_small, iterations=1)

        road_mask = cv2.bitwise_or(asphalt_mask, cv2.dilate(lane_hint_mask, kernel_medium, iterations=1))
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, kernel_large)
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_OPEN, kernel_medium)

        transform = self._nav2_perspective_transform(width, height)
        lane_bev = cv2.warpPerspective(
            lane_hint_mask,
            transform,
            (self.nav2_grid_width_cells, self.nav2_grid_height_cells),
            flags=cv2.INTER_NEAREST,
        )
        lane_bev = cv2.morphologyEx(lane_bev, cv2.MORPH_OPEN, kernel_small)
        lane_bev = cv2.dilate(lane_bev, kernel_small, iterations=1)

        left_boundary, right_boundary = self._select_lane_boundaries(lane_bev)
        lane_corridor = self._build_lane_corridor(left_boundary, right_boundary)
        if np.count_nonzero(lane_corridor) == 0:
            lane_corridor = lane_bev.copy()

        return {
            'asphalt_mask': asphalt_mask,
            'white_mask': white_mask,
            'yellow_mask': yellow_mask,
            'lane_hint_mask': lane_hint_mask,
            'road_mask': road_mask,
            'lane_bev': lane_bev,
            'lane_corridor': lane_corridor,
        }

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

    def _project_bev_to_image(self, bev_mask, width, height):
        inverse = np.linalg.inv(self._nav2_perspective_transform(width, height))
        return cv2.warpPerspective(
            bev_mask,
            inverse,
            (width, height),
            flags=cv2.INTER_NEAREST,
        )

    def _select_lane_boundaries(self, lane_bev):
        half = self.nav2_grid_width_cells // 2
        left_half = lane_bev[:, :half]
        right_half = lane_bev[:, half:]
        left_boundary = self._largest_lane_component(left_half)
        right_boundary = self._largest_lane_component(right_half)
        left_full = np.zeros_like(lane_bev)
        right_full = np.zeros_like(lane_bev)
        left_full[:, :half] = left_boundary
        right_full[:, half:] = right_boundary
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
        return cv2.morphologyEx(corridor, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    def _estimate_lane_confidence(self, lane_bev, lane_corridor):
        corridor_cells = float(np.count_nonzero(lane_corridor))
        bev_pixels = float(np.count_nonzero(lane_bev))
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
        prev = self.prev_lane_corridor if kind == 'lane_corridor' else self.prev_nav2_drivable
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

    def _project_nav2_grids(self, road_mask, lane_hint_mask, lane_corridor, lane_detected, width, height):
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
        road_bev = cv2.morphologyEx(road_bev, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        drivable = np.where(road_bev > 0, 255, 0).astype(np.uint8)
        if lane_detected and np.count_nonzero(lane_corridor) > 0:
            drivable = cv2.bitwise_and(
                drivable,
                cv2.dilate(lane_corridor, np.ones((9, 9), np.uint8), iterations=1),
            )
        drivable = cv2.bitwise_or(drivable, lane_bev)
        drivable = cv2.morphologyEx(drivable, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        keepout = np.where(drivable > 0, 0, 100).astype(np.uint8)
        return road_bev, keepout, drivable

    def _make_overlay(self, frame, masks, lane_corridor_image):
        overlay = frame.copy()
        color_mask = np.zeros_like(frame)
        color_mask[masks['road_mask'] > 0] = (70, 70, 70)
        color_mask[masks['white_mask'] > 0] = (255, 255, 255)
        color_mask[masks['yellow_mask'] > 0] = (0, 215, 255)
        color_mask[lane_corridor_image > 0] = (255, 255, 0)
        active = np.any(color_mask != 0, axis=2)
        overlay[active] = cv2.addWeighted(frame, 0.55, color_mask, 0.45, 0)[active]
        return overlay

    def _publish_images(
        self,
        source_msg,
        frame,
        overlay,
        asphalt_mask,
        white_mask,
        yellow_mask,
        lane_hint_mask,
        lane_bev,
        lane_corridor,
        road_bev,
        nav2_keepout_mask,
    ):
        if self.publish_input_image:
            msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            msg.header = source_msg.header
            self.input_pub.publish(msg)

        for publisher, image, encoding in (
            (self.overlay_pub, overlay, 'bgr8'),
            (self.asphalt_pub, asphalt_mask, 'mono8'),
            (self.white_pub, white_mask, 'mono8'),
            (self.yellow_pub, yellow_mask, 'mono8'),
            (self.lane_hint_pub, lane_hint_mask, 'mono8'),
            (self.lane_bev_pub, lane_bev, 'mono8'),
            (self.corridor_pub, lane_corridor, 'mono8'),
            (self.road_bev_pub, road_bev, 'mono8'),
            (self.bev_mask_pub, nav2_keepout_mask, 'mono8'),
        ):
            msg = self.bridge.cv2_to_imgmsg(image, encoding=encoding)
            msg.header = source_msg.header
            publisher.publish(msg)

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
        masks,
        road_bev,
        lane_corridor,
        nav2_keepout_mask,
        lane_confidence,
        lane_detected,
        elapsed_ms,
    ):
        payload = {
            'header': {
                'stamp': {
                    'sec': source_msg.header.stamp.sec,
                    'nanosec': source_msg.header.stamp.nanosec,
                },
                'frame_id': source_msg.header.frame_id,
            },
            'lane_detected': bool(lane_detected),
            'lane_confidence': lane_confidence,
            'planner_mode_hint': 'lane_following' if lane_detected else 'obstacle_avoidance',
            'asphalt_pixels': int(np.count_nonzero(masks['asphalt_mask'])),
            'white_pixels': int(np.count_nonzero(masks['white_mask'])),
            'yellow_pixels': int(np.count_nonzero(masks['yellow_mask'])),
            'lane_hint_pixels': int(np.count_nonzero(masks['lane_hint_mask'])),
            'road_pixels': int(np.count_nonzero(masks['road_mask'])),
            'road_bev_pixels': int(np.count_nonzero(road_bev)),
            'lane_bev_pixels': int(np.count_nonzero(masks['lane_bev'])),
            'lane_corridor_cells': int(np.count_nonzero(lane_corridor)),
            'nav2_keepout_cells': int(np.count_nonzero(nav2_keepout_mask == 100)),
            'nav2_grid': {
                'resolution': self.nav2_grid_resolution,
                'x_range': self.nav2_x_range,
                'y_range': self.nav2_y_range,
                'width_m': self.nav2_grid_width_m,
                'length_m': self.nav2_grid_length_m,
            },
            'timing_ms': elapsed_ms,
        }
        metadata_msg = String()
        metadata_msg.data = json.dumps(payload)
        self.metadata_pub.publish(metadata_msg)

        if self.publish_timing:
            timing_msg = String()
            timing_msg.data = json.dumps({
                'timing_ms': elapsed_ms,
                'frame_id': source_msg.header.frame_id,
            })
            self.timing_pub.publish(timing_msg)

    def _publish_filter_info(self):
        if not self.nav2_publish_grid:
            return
        msg = CostmapFilterInfo()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.type = 0
        msg.filter_mask_topic = '/igvc_bev/nav2/filter_mask'
        msg.base = 0.0
        msg.multiplier = 1.0
        self.nav2_filter_info_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = IgvcBevNode()
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
