import json
import os
from glob import glob

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from ultralytics import YOLO


DEFAULT_MODEL = (
    '/home/alexander/Desktop/Competiton_Semantic_Segmentation/'
    'models/roboflow_logistics_yolov8.pt'
)

DEFAULT_IMAGE_DIR = (
    '/home/alexander/Desktop/Competiton_Semantic_Segmentation/'
    'proof/traffic_cones/raw_road_inputs'
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


class CompetitionObjectsNode(Node):
    def __init__(self):
        super().__init__('competition_objects_node')

        self.declare_parameter('image_dir', DEFAULT_IMAGE_DIR)
        self.declare_parameter('model_path', DEFAULT_MODEL)
        self.declare_parameter('enabled_classes', DEFAULT_CLASSES)
        self.declare_parameter('confidence', 0.35)
        self.declare_parameter('device', 'cpu')
        self.declare_parameter('publish_rate_hz', 1.0)

        self.image_dir = self.get_parameter('image_dir').value
        self.enabled_classes = set(self.get_parameter('enabled_classes').value)
        self.confidence = float(self.get_parameter('confidence').value)
        self.device = self.get_parameter('device').value

        self.bridge = CvBridge()
        self.input_pub = self.create_publisher(
            Image, '/seg_ros/competition_objects/input_image', 10)
        self.annotated_pub = self.create_publisher(
            Image, '/seg_ros/competition_objects/annotated_image', 10)
        self.detection_pub = self.create_publisher(
            String, '/seg_ros/competition_objects/detections', 10)

        self.model = YOLO(self.get_parameter('model_path').value)
        self.image_files = self._collect_images(self.image_dir)
        if not self.image_files:
            raise RuntimeError(f'No images found in {self.image_dir}')

        self.image_index = 0
        period = 1.0 / float(self.get_parameter('publish_rate_hz').value)
        self.timer = self.create_timer(period, self._timer_cb)

        self.get_logger().info(
            f'Loaded {len(self.image_files)} demo images and classes: '
            f'{sorted(self.enabled_classes)}')

    def _collect_images(self, image_dir):
        files = []
        for pattern in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
            files.extend(glob(os.path.join(image_dir, pattern)))
        return sorted(files)

    def _timer_cb(self):
        image_path = self.image_files[self.image_index]
        image = cv2.imread(image_path)
        if image is None:
            self.get_logger().warning(f'Could not read {image_path}')
            self.image_index = (self.image_index + 1) % len(self.image_files)
            return

        detections = self._detect(image_path)
        annotated = image.copy()
        for det in detections:
            self._draw_detection(annotated, det)

        stamp = self.get_clock().now().to_msg()
        frame_id = 'camera_color_optical_frame'

        input_msg = self.bridge.cv2_to_imgmsg(image, encoding='bgr8')
        input_msg.header.stamp = stamp
        input_msg.header.frame_id = frame_id
        self.input_pub.publish(input_msg)

        annotated_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
        annotated_msg.header.stamp = stamp
        annotated_msg.header.frame_id = frame_id
        self.annotated_pub.publish(annotated_msg)

        msg = String()
        msg.data = json.dumps({
            'image': os.path.basename(image_path),
            'count': len(detections),
            'detections': detections,
        })
        self.detection_pub.publish(msg)

        self.get_logger().info(
            f'Published {os.path.basename(image_path)} with '
            f'{len(detections)} competition object detections')
        self.image_index = (self.image_index + 1) % len(self.image_files)

    def _detect(self, image_path):
        result = self.model.predict(
            image_path,
            conf=self.confidence,
            imgsz=640,
            verbose=False,
            device=self.device,
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
    node = CompetitionObjectsNode()
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
