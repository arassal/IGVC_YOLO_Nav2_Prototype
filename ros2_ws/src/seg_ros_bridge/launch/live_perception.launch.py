from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'venv_python',
            default_value='/home/alexander/github/av-perception/.venv/bin/python',
            description='Python interpreter with PyTorch, Ultralytics, and ROS bindings.',
        ),
        DeclareLaunchArgument(
            'node_script',
            default_value=(
                '/home/alexander/Desktop/Competiton_Semantic_Segmentation/'
                'ros2_ws/src/seg_ros_bridge/seg_ros_bridge/live_perception_node.py'
            ),
            description='Path to the live perception node.',
        ),
        DeclareLaunchArgument(
            'image_topic',
            default_value='/zed/zed_node/rgb/color/rect/image',
            description='Live ZED X rectified RGB image topic to subscribe to.',
        ),
        DeclareLaunchArgument(
            'project_root',
            default_value='/home/alexander/Desktop/seg',
            description='Path to the YOLOPv2 source tree used for utility imports.',
        ),
        DeclareLaunchArgument(
            'segmentation_weights_path',
            default_value='/home/alexander/Desktop/seg/data/weights/yolopv2.pt',
            description='YOLOPv2 TorchScript checkpoint.',
        ),
        DeclareLaunchArgument(
            'object_model_path',
            default_value=(
                '/home/alexander/Desktop/Competiton_Semantic_Segmentation/'
                'models/roboflow_logistics_yolov8.pt'
            ),
            description='YOLOv8 object detection checkpoint.',
        ),
        DeclareLaunchArgument('device', default_value='cpu'),
        DeclareLaunchArgument('img_size', default_value='640'),
        DeclareLaunchArgument('seg_conf_thres', default_value='0.30'),
        DeclareLaunchArgument('seg_iou_thres', default_value='0.45'),
        DeclareLaunchArgument('object_confidence', default_value='0.35'),
        DeclareLaunchArgument(
            'enabled_classes',
            default_value='person,traffic cone,traffic light,road sign,car,truck,van',
            description='Comma-separated object classes allowed in output.',
        ),
        DeclareLaunchArgument(
            'process_every_n',
            default_value='1',
            description='Process every Nth image frame.',
        ),
        DeclareLaunchArgument('publish_input_image', default_value='true'),
        DeclareLaunchArgument('publish_timing', default_value='true'),

        ExecuteProcess(
            cmd=[
                LaunchConfiguration('venv_python'),
                LaunchConfiguration('node_script'),
                '--ros-args',
                '-p', ['image_topic:=', LaunchConfiguration('image_topic')],
                '-p', ['project_root:=', LaunchConfiguration('project_root')],
                '-p', [
                    'segmentation_weights_path:=',
                    LaunchConfiguration('segmentation_weights_path'),
                ],
                '-p', ['object_model_path:=', LaunchConfiguration('object_model_path')],
                '-p', ['device:=', LaunchConfiguration('device')],
                '-p', ['img_size:=', LaunchConfiguration('img_size')],
                '-p', ['seg_conf_thres:=', LaunchConfiguration('seg_conf_thres')],
                '-p', ['seg_iou_thres:=', LaunchConfiguration('seg_iou_thres')],
                '-p', ['object_confidence:=', LaunchConfiguration('object_confidence')],
                '-p', ['enabled_classes:=', LaunchConfiguration('enabled_classes')],
                '-p', ['process_every_n:=', LaunchConfiguration('process_every_n')],
                '-p', ['publish_input_image:=', LaunchConfiguration('publish_input_image')],
                '-p', ['publish_timing:=', LaunchConfiguration('publish_timing')],
            ],
            output='screen',
        ),
    ])
