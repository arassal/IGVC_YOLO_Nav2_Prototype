from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'venv_python',
            default_value='/home/alexander/github/av-perception/.venv/bin/python',
        ),
        DeclareLaunchArgument(
            'image_dir',
            default_value='/home/alexander/Desktop/img',
        ),
        DeclareLaunchArgument(
            'replay_topic',
            default_value='/seg_ros/demo/input_image',
        ),
        DeclareLaunchArgument(
            'fps',
            default_value='1.0',
        ),
        DeclareLaunchArgument(
            'enable_temporal_smoothing',
            default_value='true',
        ),
        DeclareLaunchArgument(
            'temporal_alpha',
            default_value='0.65',
        ),
        DeclareLaunchArgument(
            'use_rviz',
            default_value='true',
        ),
        DeclareLaunchArgument(
            'nav2_x_range',
            default_value='[0.0, 15.0]',
        ),
        DeclareLaunchArgument(
            'nav2_y_range',
            default_value='[-10.0, 10.0]',
        ),
        DeclareLaunchArgument(
            'camera_mount_x',
            default_value='0.35',
        ),
        DeclareLaunchArgument(
            'camera_mount_y',
            default_value='0.0',
        ),
        DeclareLaunchArgument(
            'camera_mount_z',
            default_value='0.75',
        ),
        DeclareLaunchArgument(
            'camera_mount_yaw',
            default_value='0.0',
        ),
        DeclareLaunchArgument(
            'rviz_config',
            default_value=(
                '/home/alexander/Desktop/Competiton_Semantic_Segmentation/'
                'ros2_ws/src/seg_ros_bridge/rviz/segformer.rviz'
            ),
        ),
        ExecuteProcess(
            cmd=[
                'ros2', 'run', 'seg_ros_bridge', 'image_replay_node',
                '--ros-args',
                '-p', ['image_dir:=', LaunchConfiguration('image_dir')],
                '-p', ['image_topic:=', LaunchConfiguration('replay_topic')],
                '-p', ['fps:=', LaunchConfiguration('fps')],
            ],
            output='screen',
        ),
        ExecuteProcess(
            cmd=[
                LaunchConfiguration('venv_python'),
                (
                    '/home/alexander/Desktop/Competiton_Semantic_Segmentation/'
                    'ros2_ws/src/seg_ros_bridge/seg_ros_bridge/segformer_node.py'
                ),
                '--ros-args',
                '-p', ['image_topic:=', LaunchConfiguration('replay_topic')],
                '-p', 'publish_input_image:=true',
                '-p', 'enable_hsv_refinement:=true',
                '-p', 'nav2_publish_grid:=true',
                '-p', ['enable_temporal_smoothing:=', LaunchConfiguration('enable_temporal_smoothing')],
                '-p', ['temporal_alpha:=', LaunchConfiguration('temporal_alpha')],
                '-p', ['nav2_x_range:=', LaunchConfiguration('nav2_x_range')],
                '-p', ['nav2_y_range:=', LaunchConfiguration('nav2_y_range')],
                '-p', ['camera_mount_x:=', LaunchConfiguration('camera_mount_x')],
                '-p', ['camera_mount_y:=', LaunchConfiguration('camera_mount_y')],
                '-p', ['camera_mount_z:=', LaunchConfiguration('camera_mount_z')],
                '-p', ['camera_mount_yaw:=', LaunchConfiguration('camera_mount_yaw')],
            ],
            output='screen',
        ),
        ExecuteProcess(
            cmd=['rviz2', '-d', LaunchConfiguration('rviz_config')],
            condition=IfCondition(LaunchConfiguration('use_rviz')),
            output='screen',
        ),
    ])
