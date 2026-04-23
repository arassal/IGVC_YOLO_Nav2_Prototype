from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'venv_python',
            default_value='/home/alexander/github/av-perception/.venv/bin/python',
            description='Python interpreter with ultralytics installed.',
        ),
        DeclareLaunchArgument(
            'node_script',
            default_value=(
                '/home/alexander/Desktop/Competiton_Semantic_Segmentation/'
                'ros2_ws/src/seg_ros_bridge/seg_ros_bridge/competition_objects_node.py'
            ),
            description='Path to the competition object detection node.',
        ),
        DeclareLaunchArgument(
            'image_dir',
            default_value=(
                '/home/alexander/Desktop/Competiton_Semantic_Segmentation/'
                'proof/traffic_cones/raw_road_inputs'
            ),
            description='Image directory for repeatable object detection demo.',
        ),
        DeclareLaunchArgument(
            'model_path',
            default_value=(
                '/home/alexander/Desktop/Competiton_Semantic_Segmentation/'
                'models/roboflow_logistics_yolov8.pt'
            ),
            description='YOLOv8 checkpoint with traffic cone class.',
        ),
        DeclareLaunchArgument('device', default_value='cpu'),
        DeclareLaunchArgument('confidence', default_value='0.35'),
        DeclareLaunchArgument('publish_rate_hz', default_value='1.0'),

        ExecuteProcess(
            cmd=[
                LaunchConfiguration('venv_python'),
                LaunchConfiguration('node_script'),
                '--ros-args',
                '-p', ['image_dir:=', LaunchConfiguration('image_dir')],
                '-p', ['model_path:=', LaunchConfiguration('model_path')],
                '-p', ['device:=', LaunchConfiguration('device')],
                '-p', ['confidence:=', LaunchConfiguration('confidence')],
                '-p', ['publish_rate_hz:=', LaunchConfiguration('publish_rate_hz')],
            ],
            output='screen',
        ),
    ])
