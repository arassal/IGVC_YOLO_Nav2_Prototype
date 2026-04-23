from pathlib import Path

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


class ImageReplayNode(Node):
    """Publish images from a folder as a repeatable ROS image stream."""

    def __init__(self):
        super().__init__('image_replay_node')
        self.declare_parameter('image_dir', '/home/alexander/Desktop/img')
        self.declare_parameter('image_topic', '/seg_ros/demo/input_image')
        self.declare_parameter('frame_id', 'zed_camera_center')
        self.declare_parameter('fps', 1.0)
        self.declare_parameter('loop', True)

        image_dir = Path(self.get_parameter('image_dir').value)
        self.image_topic = self.get_parameter('image_topic').value
        self.frame_id = self.get_parameter('frame_id').value
        self.loop = bool(self.get_parameter('loop').value)
        fps = max(0.1, float(self.get_parameter('fps').value))

        self.images = []
        for pattern in ('*.jpg', '*.jpeg', '*.png', '*.bmp'):
            self.images.extend(sorted(image_dir.glob(pattern)))
        if not self.images:
            raise RuntimeError(f'No replay images found in {image_dir}')

        self.bridge = CvBridge()
        self.publisher = self.create_publisher(Image, self.image_topic, 10)
        self.index = 0
        self.timer = self.create_timer(1.0 / fps, self._tick)
        self.get_logger().info(
            f'Publishing {len(self.images)} replay images to {self.image_topic}')

    def _tick(self):
        if self.index >= len(self.images):
            if not self.loop:
                self.get_logger().info('Replay finished')
                self.timer.cancel()
                return
            self.index = 0

        path = self.images[self.index]
        frame = cv2.imread(str(path))
        if frame is None:
            self.get_logger().warning(f'Failed to read image: {path}')
            self.index += 1
            return

        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        self.publisher.publish(msg)
        self.index += 1


def main(args=None):
    rclpy.init(args=args)
    node = ImageReplayNode()
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
