from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'venv_python',
            default_value='/home/alexander/github/av-perception/.venv/bin/python',
            description='Python interpreter with ROS bindings and cv_bridge.',
        ),
        DeclareLaunchArgument(
            'node_script',
            default_value=(
                '/home/alexander/Desktop/Competiton_Semantic_Segmentation/'
                'ros2_ws/src/seg_ros_bridge/seg_ros_bridge/zed_image_recorder_node.py'
            ),
            description='Path to the ZED X image recorder node.',
        ),
        DeclareLaunchArgument(
            'image_topic',
            default_value='/zed/zed_node/rgb/color/rect/image',
            description='ZED X image topic to record.',
        ),
        DeclareLaunchArgument(
            'output_dir',
            default_value=(
                '/home/alexander/Desktop/Competiton_Semantic_Segmentation/'
                'validation/zed_frames'
            ),
            description='Directory where validation frames and manifest are written.',
        ),
        DeclareLaunchArgument('max_frames', default_value='200'),
        DeclareLaunchArgument('save_every_n', default_value='5'),
        DeclareLaunchArgument('image_extension', default_value='jpg'),
        DeclareLaunchArgument('jpeg_quality', default_value='95'),

        ExecuteProcess(
            cmd=[
                LaunchConfiguration('venv_python'),
                LaunchConfiguration('node_script'),
                '--ros-args',
                '-p', ['image_topic:=', LaunchConfiguration('image_topic')],
                '-p', ['output_dir:=', LaunchConfiguration('output_dir')],
                '-p', ['max_frames:=', LaunchConfiguration('max_frames')],
                '-p', ['save_every_n:=', LaunchConfiguration('save_every_n')],
                '-p', ['image_extension:=', LaunchConfiguration('image_extension')],
                '-p', ['jpeg_quality:=', LaunchConfiguration('jpeg_quality')],
            ],
            output='screen',
        ),
    ])
