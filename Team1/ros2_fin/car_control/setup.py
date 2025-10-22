
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

    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ksj',
    maintainer_email='seolihan651@gmail.com',
    description='Rule-based V2X avoidance node',
    license='Apache-2.0',
    entry_points={'console_scripts': [
        'motor_controller_node = car_control.motor_controller_node:main',
    ]},
)
