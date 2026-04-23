from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('image_topic', default_value='/zed/zed_node/rgb/color/rect/image'),
        DeclareLaunchArgument('use_rviz', default_value='true'),
        DeclareLaunchArgument('nav2_x_range', default_value='[0.0, 15.0]'),
        DeclareLaunchArgument('nav2_y_range', default_value='[-10.0, 10.0]'),
        DeclareLaunchArgument('enable_temporal_smoothing', default_value='true'),
        DeclareLaunchArgument('temporal_alpha', default_value='0.65'),
        DeclareLaunchArgument(
            'rviz_config',
            default_value=(
                '/home/alexander/Desktop/IGVC_Nav2_SegFormer/'
                'ros2_ws/src/seg_ros_bridge/rviz/igvc_bev.rviz'
            ),
        ),
        ExecuteProcess(
            cmd=[
                'ros2', 'run', 'seg_ros_bridge', 'igvc_bev_node',
                '--ros-args',
                '-p', ['image_topic:=', LaunchConfiguration('image_topic')],
                '-p', 'publish_input_image:=true',
                '-p', 'nav2_publish_grid:=true',
                '-p', ['enable_temporal_smoothing:=', LaunchConfiguration('enable_temporal_smoothing')],
                '-p', ['temporal_alpha:=', LaunchConfiguration('temporal_alpha')],
                '-p', ['nav2_x_range:=', LaunchConfiguration('nav2_x_range')],
                '-p', ['nav2_y_range:=', LaunchConfiguration('nav2_y_range')],
            ],
            output='screen',
        ),
        ExecuteProcess(
            cmd=['rviz2', '-d', LaunchConfiguration('rviz_config')],
            condition=IfCondition(LaunchConfiguration('use_rviz')),
            output='screen',
        ),
    ])
