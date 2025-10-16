from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    alert_in    = LaunchConfiguration('alert_in')
    alert_out   = LaunchConfiguration('alert_out')
    log_level   = LaunchConfiguration('log_level')
    ns          = LaunchConfiguration('ns')
    drop_exp    = LaunchConfiguration('drop_expired')
    reliability = LaunchConfiguration('reliability')
    history     = LaunchConfiguration('history')
    depth       = LaunchConfiguration('depth')

    return LaunchDescription([
        DeclareLaunchArgument('alert_in',     default_value='/v2x/alert'),
        DeclareLaunchArgument('alert_out',    default_value='/v2x/alert_struct'),
        DeclareLaunchArgument('log_level',    default_value='info'),
        DeclareLaunchArgument('ns',           default_value=''),
        DeclareLaunchArgument('drop_expired', default_value='true'),
        DeclareLaunchArgument('reliability',  default_value='reliable'),  # reliable|besteffort
        DeclareLaunchArgument('history',      default_value='keeplast'),   # keeplast|keepall
        DeclareLaunchArgument('depth',        default_value='10'),

        Node(
            package='car_comms',
            executable='v2x_alert_bridge',
            name='v2x_alert_bridge',
            namespace=ns,
            output='screen',
            emulate_tty=True,
            respawn=True,
            respawn_delay=2.0,
            parameters=[{
                'alert_in': alert_in,
                'alert_out': alert_out,
                'drop_expired': drop_exp,
                'reliability': reliability,
                'history': history,
                'depth': depth,
            }],
            arguments=['--ros-args', '--log-level', log_level],
        ),
    ])
