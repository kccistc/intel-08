from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_share = get_package_share_directory('car_planning')
    default_params = os.path.join(pkg_share, 'config', 'decision_maker.yaml')

    alert_in_arg  = DeclareLaunchArgument('alert_in',  default_value='/v2x/alert')
    alert_out_arg = DeclareLaunchArgument('alert_out', default_value='/v2x/alert_struct')
    params_arg    = DeclareLaunchArgument('params_file', default_value=default_params)

    bridge = Node(
        package='car_comms',
        executable='v2x_alert_bridge',
        name='v2x_alert_bridge',
        output='screen',
        # 브릿지가 기본으로 /v2x/alert → /v2x/alert_struct 를 쓰도록 만들었다면
        # remap으로 원하는 이름으로 바꿀 수 있게 합니다.
        remappings=[
            ('/v2x/alert',        LaunchConfiguration('alert_in')),
            ('/v2x/alert_struct', LaunchConfiguration('alert_out')),
        ],
    )

    decision = Node(
        package='car_planning',
        executable='decision_maker',
        name='decision_maker',
        output='screen',
        parameters=[LaunchConfiguration('params_file')],
        remappings=[
            ('/v2x/alert_struct', LaunchConfiguration('alert_out')),
            # '/vehicle/cmd'는 기본 그대로 사용 (필요시 여기서도 remap 가능)
        ],
    )

    return LaunchDescription([
        alert_in_arg, alert_out_arg, params_arg,
        bridge, decision
    ])
