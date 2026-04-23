import json
import time
from pathlib import Path

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


class ZedImageRecorderNode(Node):
    """Record a bounded set of ZED X image frames for offline validation."""

    def __init__(self):
        super().__init__('zed_image_recorder_node')

        self.declare_parameter('image_topic', '/zed/zed_node/rgb/color/rect/image')
        self.declare_parameter(
            'output_dir',
            '/home/alexander/Desktop/Competiton_Semantic_Segmentation/'
            'validation/zed_frames',
        )
        self.declare_parameter('max_frames', 200)
        self.declare_parameter('save_every_n', 5)
        self.declare_parameter('image_extension', 'jpg')
        self.declare_parameter('jpeg_quality', 95)

        self.image_topic = self.get_parameter('image_topic').value
        self.output_dir = Path(self.get_parameter('output_dir').value)
        self.max_frames = int(self.get_parameter('max_frames').value)
        self.save_every_n = max(1, int(self.get_parameter('save_every_n').value))
        self.image_extension = self.get_parameter('image_extension').value.lower()
        self.jpeg_quality = int(self.get_parameter('jpeg_quality').value)

        if self.image_extension not in {'jpg', 'jpeg', 'png'}:
            raise RuntimeError('image_extension must be jpg, jpeg, or png')

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.bridge = CvBridge()
        self.received = 0
        self.saved = 0
        self.manifest = []
        self.start_time = time.time()

        self.sub = self.create_subscription(
            Image, self.image_topic, self._image_cb, 10)

        self.get_logger().info(f'Recording from: {self.image_topic}')
        self.get_logger().info(f'Writing frames to: {self.output_dir}')

    def _image_cb(self, msg):
        self.received += 1
        if self.received % self.save_every_n != 0:
            return
        if self.saved >= self.max_frames:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().warning(f'Failed to convert image: {exc}')
            return

        self.saved += 1
        stem = f'frame_{self.saved:06d}'
        image_path = self.output_dir / f'{stem}.{self.image_extension}'

        params = []
        if self.image_extension in {'jpg', 'jpeg'}:
            params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]

        if not cv2.imwrite(str(image_path), frame, params):
            self.get_logger().warning(f'Failed to write {image_path}')
            return

        self.manifest.append({
            'file': image_path.name,
            'source_topic': self.image_topic,
            'frame_id': msg.header.frame_id,
            'stamp': {
                'sec': msg.header.stamp.sec,
                'nanosec': msg.header.stamp.nanosec,
            },
            'width': int(msg.width),
            'height': int(msg.height),
            'encoding': msg.encoding,
        })
        self._write_manifest()

        if self.saved % 10 == 0 or self.saved == self.max_frames:
            elapsed = max(0.001, time.time() - self.start_time)
            self.get_logger().info(
                f'Saved {self.saved}/{self.max_frames} frames '
                f'({self.received} received, {self.saved / elapsed:.2f} saved/s)')

        if self.saved >= self.max_frames:
            self.get_logger().info('Reached max_frames; recorder will stay idle.')

    def _write_manifest(self):
        payload = {
            'image_topic': self.image_topic,
            'output_dir': str(self.output_dir),
            'max_frames': self.max_frames,
            'save_every_n': self.save_every_n,
            'saved_frames': self.saved,
            'received_frames': self.received,
            'frames': self.manifest,
        }
        (self.output_dir / 'manifest.json').write_text(json.dumps(payload, indent=2))


def main(args=None):
    rclpy.init(args=args)
    node = ZedImageRecorderNode()
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
