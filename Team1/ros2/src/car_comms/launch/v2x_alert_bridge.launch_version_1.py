# car_comms/launch/v2x_alert_bridge.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    alert_in  = LaunchConfiguration('alert_in')
    alert_out = LaunchConfiguration('alert_out')
    log_level = LaunchConfiguration('log_level')
    ns        = LaunchConfiguration('ns')

    return LaunchDescription([
        DeclareLaunchArgument('alert_in',  default_value='/v2x/alert'),
        DeclareLaunchArgument('alert_out', default_value='/v2x/alert_struct'),
        DeclareLaunchArgument('log_level', default_value='info'),
        DeclareLaunchArgument('ns',        default_value=''),

        Node(
            package='car_comms',
            executable='v2x_alert_bridge',
            name='v2x_alert_bridge',
            namespace=ns,
            output='screen',
            emulate_tty=True,
            respawn=True,
            respawn_delay=2.0,
            arguments=['--ros-args', '--log-level', log_level],
            remappings=[
                ('/v2x/alert', alert_in),
                ('/v2x/alert_struct', alert_out),
            ],
        ),
    ])
