from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_share = get_package_share_directory('car_planning')
    default_params = os.path.join(pkg_share, 'config', 'decision_maker.yaml')

    # Arguments
    alert_in  = DeclareLaunchArgument('alert_in',  default_value='/v2x/alert')
    alert_out = DeclareLaunchArgument('alert_out', default_value='/v2x/alert_struct')
    params    = DeclareLaunchArgument('params_file', default_value=default_params)

    # Optional toggles
    enable_serial = DeclareLaunchArgument('enable_serial', default_value='false')   # car_control 모터 제어 실행 여부
    serial_port   = DeclareLaunchArgument('serial_port',   default_value='/dev/ttyUSB0')
    baudrate      = DeclareLaunchArgument('baudrate',      default_value='115200')

    enable_bag    = DeclareLaunchArgument('enable_bag', default_value='false')     # rosbag 기록

    # Nodes
    v2c_comm_node = Node(
        package='car_comms',
        executable='v2c_comm',
        name='v2c_comm_node',
        output='screen',
        respawn=True,
        parameters=[{
            'alert_out': LaunchConfiguration('alert_out'),
            # 'hmac_key': 'your_secret_key', # HMAC 키가 필요한 경우 여기에 추가
            # 'drop_expired': True,
        }],
        # v2c_comm은 입력 토픽이 없으므로 remappings에서 alert_in 제거
    )

    decision = Node(
        package='car_planning',
        executable='decision_maker',
        name='decision_maker',
        output='screen',
        respawn=True,
        parameters=[LaunchConfiguration('params_file')],
        remappings=[
            ('/v2x/alert_struct', LaunchConfiguration('alert_out')),
            # '/vehicle/cmd' 기본 그대로
        ],
    )

    # Optional: UART 모터 컨트롤러 (car_control 패키지가 제공한다고 가정)
    motor = Node(
        condition=None,  # 런치 인자에 따라 동적으로 제어 (아래 GroupAction에서 처리)
        package='car_control',
        executable='motor_controller_node',
        name='motor_controller_node',
        output='screen',
        respawn=True,
        parameters=[{
            'serial_port': LaunchConfiguration('serial_port'),
            'baudrate': LaunchConfiguration('baudrate'),
        }],
        remappings=[
            ('/cmd_vel', '/vehicle/cmd'),  # decision_maker 출력과 연결
        ],
    )

    # enable_serial 플래그 처리
    serial_group = GroupAction(actions=[motor])

    # Optional: rosbag2 기록 (topics: 입력/출력)
    # enable_bag=true면 /v2x/alert_struct /vehicle/cmd 저장
    from launch.actions import OpaqueFunction
    from launch_ros.substitutions import FindPackageShare

    def maybe_bag(context, *args, **kwargs):
        if LaunchConfiguration('enable_bag').perform(context).lower() != 'true':
            return []
        # rosbag2 를 Node가 아니라 CLI 호출로 띄우고 싶다면 ExecuteProcess 사용 가능
        from launch.actions import ExecuteProcess
        return [ExecuteProcess(
            cmd=[
                'ros2','bag','record',
                '/v2x/alert_struct','/vehicle/cmd'
            ],
            output='screen'
        )]

    bag = OpaqueFunction(function=maybe_bag)

    # enable_serial=true일 때만 motor 포함시키기 위한 OpaqueFunction
    def maybe_serial(context, *args, **kwargs):
        if LaunchConfiguration('enable_serial').perform(context).lower() == 'true':
            return [motor]
        return []

    serial = OpaqueFunction(function=maybe_serial)

    return LaunchDescription([
        alert_in, alert_out, params,
        enable_serial, serial_port, baudrate,
        enable_bag,

        v2c_comm_node,
        decision,
        serial,   # 조건부 모터 노드
        bag,      # 조건부 bag 기록
    ])
