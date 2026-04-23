import json
import os
import sys
from glob import glob

import cv2
import numpy as np
import rclpy
import torch
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from vision_msgs.msg import LabelInfo, VisionClass


class SegDemoNode(Node):
    def __init__(self) -> None:
        super().__init__('seg_demo_node')

        self.declare_parameter('project_root', '/home/alexander/Desktop/seg')
        self.declare_parameter('image_dir', '')
        self.declare_parameter('weights_path', '')
        self.declare_parameter('device', 'cpu')
        self.declare_parameter('img_size', 640)
        self.declare_parameter('conf_thres', 0.30)
        self.declare_parameter('iou_thres', 0.45)
        self.declare_parameter('publish_rate_hz', 1.0)

        self.project_root = self.get_parameter('project_root').value
        if self.project_root not in sys.path:
            sys.path.insert(0, self.project_root)

        from utils.utils import (
            driving_area_mask,
            lane_line_mask,
            letterbox,
            non_max_suppression,
            plot_one_box,
            scale_coords,
            select_device,
            show_seg_result,
            split_for_trace_model,
        )

        self.driving_area_mask = driving_area_mask
        self.lane_line_mask = lane_line_mask
        self.letterbox = letterbox
        self.non_max_suppression = non_max_suppression
        self.plot_one_box = plot_one_box
        self.scale_coords = scale_coords
        self.select_device = select_device
        self.show_seg_result = show_seg_result
        self.split_for_trace_model = split_for_trace_model

        self.image_dir = self.get_parameter('image_dir').value or os.path.join(self.project_root, 'data', 'demo')
        self.weights_path = self.get_parameter('weights_path').value or os.path.join(self.project_root, 'data', 'weights', 'yolopv2.pt')
        self.img_size = int(self.get_parameter('img_size').value)
        self.conf_thres = float(self.get_parameter('conf_thres').value)
        self.iou_thres = float(self.get_parameter('iou_thres').value)

        self.bridge = CvBridge()
        self.input_pub = self.create_publisher(Image, '/seg_ros/input_image', 10)
        self.overlay_pub = self.create_publisher(Image, '/seg_ros/overlay_image', 10)
        self.drivable_pub = self.create_publisher(Image, '/seg_ros/drivable_mask', 10)
        self.lane_pub = self.create_publisher(Image, '/seg_ros/lane_mask', 10)
        self.lane_confidence_pub = self.create_publisher(
            Image, '/seg_ros/lane_confidence', 10)
        label_qos = QoSProfile(depth=1)
        label_qos.reliability = ReliabilityPolicy.RELIABLE
        label_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.label_pub = self.create_publisher(
            LabelInfo, '/seg_ros/label_info', label_qos)
        self.detection_pub = self.create_publisher(String, '/seg_ros/detections', 10)

        self.image_files = self._collect_images(self.image_dir)
        if not self.image_files:
            raise RuntimeError(f'No images found in {self.image_dir}')

        self.device = self.select_device(self.get_parameter('device').value)
        self.model = torch.jit.load(self.weights_path, map_location=self.device).to(self.device)
        self.half = self.device.type != 'cpu'
        if self.half:
            self.model.half()
        self.model.eval()

        self.image_index = 0
        period = 1.0 / float(self.get_parameter('publish_rate_hz').value)
        self.timer = self.create_timer(period, self._timer_cb)

        self.get_logger().info(f'Loaded model: {self.weights_path}')
        self.get_logger().info(f'Publishing from: {self.image_dir} ({len(self.image_files)} images)')
        self._publish_label_info()

    def _collect_images(self, image_dir: str):
        image_patterns = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp', '*.tif', '*.tiff']
        files = []
        for pat in image_patterns:
            files.extend(glob(os.path.join(image_dir, pat)))
        files.sort()
        return files

    def _timer_cb(self):
        image_path = self.image_files[self.image_index]
        frame = cv2.imread(image_path)
        if frame is None:
            self.get_logger().warning(f'Failed to read image: {image_path}')
            self.image_index = (self.image_index + 1) % len(self.image_files)
            return

        overlay, da_mask, ll_mask, detections = self._infer(frame)

        stamp = self.get_clock().now().to_msg()
        frame_id = 'camera_color_optical_frame'

        input_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        input_msg.header.stamp = stamp
        input_msg.header.frame_id = frame_id
        self.input_pub.publish(input_msg)

        overlay_msg = self.bridge.cv2_to_imgmsg(overlay, encoding='bgr8')
        overlay_msg.header.stamp = stamp
        overlay_msg.header.frame_id = frame_id
        self.overlay_pub.publish(overlay_msg)

        drivable_msg = self.bridge.cv2_to_imgmsg(
            (da_mask * 255).astype(np.uint8), encoding='mono8')
        drivable_msg.header.stamp = stamp
        drivable_msg.header.frame_id = frame_id
        self.drivable_pub.publish(drivable_msg)

        lane_mask = (ll_mask * 255).astype(np.uint8)
        lane_msg = self.bridge.cv2_to_imgmsg(lane_mask, encoding='mono8')
        lane_msg.header.stamp = stamp
        lane_msg.header.frame_id = frame_id
        self.lane_pub.publish(lane_msg)

        confidence_msg = self.bridge.cv2_to_imgmsg(lane_mask, encoding='mono8')
        confidence_msg.header.stamp = stamp
        confidence_msg.header.frame_id = frame_id
        self.lane_confidence_pub.publish(confidence_msg)

        msg = String()
        msg.data = json.dumps({'image': os.path.basename(image_path), 'count': len(detections), 'detections': detections})
        self.detection_pub.publish(msg)

        self.get_logger().info(f'Published {os.path.basename(image_path)} with {len(detections)} detections')
        self.image_index = (self.image_index + 1) % len(self.image_files)

    def _publish_label_info(self):
        msg = LabelInfo()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_color_optical_frame'
        msg.threshold = float(self.conf_thres)
        msg.class_map = [
            VisionClass(class_id=0, class_name='background'),
            VisionClass(class_id=1, class_name='drivable_area'),
            VisionClass(class_id=2, class_name='lane_marking'),
        ]
        self.label_pub.publish(msg)

    def _infer(self, frame):
        im0 = cv2.resize(frame.copy(), (1280, 720), interpolation=cv2.INTER_LINEAR)
        img = self.letterbox(im0, self.img_size, stride=32)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        tensor = torch.from_numpy(img).to(self.device)
        tensor = tensor.half() if self.half else tensor.float()
        tensor /= 255.0
        if tensor.ndimension() == 3:
            tensor = tensor.unsqueeze(0)

        pred_raw, seg, ll = self.model(tensor)
        if isinstance(pred_raw, (list, tuple)) and len(pred_raw) == 2:
            pred = self.split_for_trace_model(pred_raw[0], pred_raw[1])
        else:
            pred = self.split_for_trace_model(pred_raw, None)
        pred = self.non_max_suppression(pred, self.conf_thres, self.iou_thres)

        da_seg_mask = self.driving_area_mask(seg)
        ll_seg_mask = self.lane_line_mask(ll)

        overlay = im0.copy()
        self.show_seg_result(overlay, (da_seg_mask, ll_seg_mask), is_demo=True)

        detections = []
        for det in pred:
            if len(det):
                det[:, :4] = self.scale_coords(tensor.shape[2:], det[:, :4], overlay.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    self.plot_one_box(xyxy, overlay, line_thickness=3)
                    detections.append({'xyxy': [float(v) for v in xyxy], 'conf': float(conf), 'cls': int(cls)})

        return overlay, da_seg_mask, ll_seg_mask, detections


def main(args=None):
    rclpy.init(args=args)
    node = SegDemoNode()
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
