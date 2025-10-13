import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/jeongseon/workspace/intel-08/Team1/ksj/ros2_workspace/install/car_comms'
