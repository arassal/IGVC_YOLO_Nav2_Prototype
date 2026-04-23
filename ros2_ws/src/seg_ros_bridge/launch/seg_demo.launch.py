from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='seg_ros_bridge',
            executable='seg_demo_node',
            name='seg_demo_node',
            output='screen',
            parameters=[
                {
                    'project_root': '/home/alexander/Desktop/seg',
                    'image_dir': '/home/alexander/Desktop/seg/data/demo',
                    'weights_path': '/home/alexander/Desktop/seg/data/weights/yolopv2.pt',
                    'device': 'cpu',
                    'publish_rate_hz': 1.0,
                }
            ],
        )
    ])
