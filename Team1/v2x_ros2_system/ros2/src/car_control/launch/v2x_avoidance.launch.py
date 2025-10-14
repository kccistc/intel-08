from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_share = get_package_share_directory('car_control')
    config_path = os.path.join(pkg_share, 'config', 'v2x_avoidance.yaml')

    return LaunchDescription([
        Node(
            package='car_control',
            executable='v2x_avoidance_node',
            name='v2x_avoidance_node',
            output='screen',
            parameters=[config_path],
        ),
    ])
