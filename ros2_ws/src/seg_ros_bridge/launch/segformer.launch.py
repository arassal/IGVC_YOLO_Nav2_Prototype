from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'venv_python',
            default_value='/home/alexander/github/av-perception/.venv/bin/python',
            description='Python interpreter with torch, transformers, and ROS bindings.',
        ),
        DeclareLaunchArgument(
            'node_script',
            default_value=(
                '/home/alexander/Desktop/Competiton_Semantic_Segmentation/'
                'ros2_ws/src/seg_ros_bridge/seg_ros_bridge/segformer_node.py'
            ),
            description='Path to the SegFormer comparison node.',
        ),
        DeclareLaunchArgument(
            'image_topic',
            default_value='/zed/zed_node/rgb/color/rect/image',
            description='ZED X image topic to segment.',
        ),
        DeclareLaunchArgument(
            'model_id',
            default_value='nvidia/segformer-b0-finetuned-cityscapes-512-1024',
            description='Hugging Face SegFormer model id.',
        ),
        DeclareLaunchArgument('device', default_value='cpu'),
        DeclareLaunchArgument('process_every_n', default_value='1'),
        DeclareLaunchArgument('publish_input_image', default_value='true'),
        DeclareLaunchArgument('publish_timing', default_value='true'),
        DeclareLaunchArgument('enable_hsv_refinement', default_value='true'),
        DeclareLaunchArgument('nav2_publish_grid', default_value='true'),
        DeclareLaunchArgument('nav2_grid_resolution', default_value='0.05'),
        DeclareLaunchArgument('nav2_x_range', default_value='[0.0, 15.0]'),
        DeclareLaunchArgument('nav2_y_range', default_value='[-10.0, 10.0]'),
        DeclareLaunchArgument('camera_mount_x', default_value='0.35'),
        DeclareLaunchArgument('camera_mount_y', default_value='0.0'),
        DeclareLaunchArgument('camera_mount_z', default_value='0.75'),
        DeclareLaunchArgument('camera_mount_yaw', default_value='0.0'),
        DeclareLaunchArgument('use_rviz', default_value='true'),
        DeclareLaunchArgument(
            'rviz_config',
            default_value=(
                '/home/alexander/Desktop/Competiton_Semantic_Segmentation/'
                'ros2_ws/src/seg_ros_bridge/rviz/segformer.rviz'
            ),
            description='RViz config for the SegFormer comparison pipeline.',
        ),

        ExecuteProcess(
            cmd=[
                LaunchConfiguration('venv_python'),
                LaunchConfiguration('node_script'),
                '--ros-args',
                '-p', ['image_topic:=', LaunchConfiguration('image_topic')],
                '-p', ['model_id:=', LaunchConfiguration('model_id')],
                '-p', ['device:=', LaunchConfiguration('device')],
                '-p', ['process_every_n:=', LaunchConfiguration('process_every_n')],
                '-p', ['publish_input_image:=', LaunchConfiguration('publish_input_image')],
                '-p', ['publish_timing:=', LaunchConfiguration('publish_timing')],
                '-p', ['enable_hsv_refinement:=', LaunchConfiguration('enable_hsv_refinement')],
                '-p', ['nav2_publish_grid:=', LaunchConfiguration('nav2_publish_grid')],
                '-p', ['nav2_grid_resolution:=', LaunchConfiguration('nav2_grid_resolution')],
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
            cmd=[
                'rviz2',
                '-d',
                LaunchConfiguration('rviz_config'),
            ],
            condition=IfCondition(LaunchConfiguration('use_rviz')),
            output='screen',
        ),
    ])
