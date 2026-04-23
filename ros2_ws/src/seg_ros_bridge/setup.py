from glob import glob
from os.path import join

from setuptools import setup

package_name = 'seg_ros_bridge'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob(join('launch', '*.launch.py'))),
        ('share/' + package_name + '/rviz', glob(join('rviz', '*.rviz'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Alexander',
    maintainer_email='alexander@assalfamily.com',
    description='ROS2 bridge for YOLOPv2 perception demo assets.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'seg_demo_node = seg_ros_bridge.seg_demo_node:main',
            'competition_objects_node = seg_ros_bridge.competition_objects_node:main',
            'live_perception_node = seg_ros_bridge.live_perception_node:main',
            'zed_image_recorder_node = seg_ros_bridge.zed_image_recorder_node:main',
            'segformer_node = seg_ros_bridge.segformer_node:main',
            'image_replay_node = seg_ros_bridge.image_replay_node:main',
        ],
    },
)
