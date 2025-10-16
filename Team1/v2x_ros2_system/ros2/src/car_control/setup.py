
from setuptools import setup, find_packages
from glob import glob
import os
package_name = 'car_control'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(include=['car_control', 'car_control.*']),
    data_files=[
        (os.path.join('share', 'ament_index', 'resource_index', 'packages'), ['resource/' + package_name]),
        (os.path.join('share', package_name), ['package.xml']),
        (os.path.join('share', package_name, 'launch'), ['launch/v2x_avoidance.launch.py']),
        (os.path.join('share', package_name, 'config'), ['config/v2x_avoidance.yaml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ksj',
    maintainer_email='seolihan651@gmail.com',
    description='Rule-based V2X avoidance node',
    license='Apache-2.0',
    entry_points={'console_scripts': [
        'v2x_avoidance_node = car_control.v2x_avoidance_node:main',
        'carla_event_bridge = car_control.carla_event_bridge:main',
        'dummy_carla_collision_pub = car_control.dummy_carla_collision_pub:main',
        'bt_cmd_bridge = car_control.bt_cmd_bridge:main',
    ]},
)
