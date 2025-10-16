from setuptools import setup, find_packages

package_name = 'car_comms'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/v2x_alert_bridge.launch.py']),
    ],
    install_requires=['setuptools', 'rclpy', 'car_msgs', 'std_msgs'],
    zip_safe=True,
    maintainer='ksj',
    maintainer_email='seolihan651@gmail.com',
    description='Communication bridge for V2X alert integration',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'v2x_alert_bridge = car_comms.v2x_alert_bridge:main',
        ],
    },
)
