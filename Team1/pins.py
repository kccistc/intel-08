"""
Input / Output Pin map for the smart factory demo kit
"""
from enum import IntEnum


class Outputs(IntEnum):
    BEACON_RED = 2
    BEACON_ORANGE = 3
    BEACON_GREEN = 4

    BEACON_BUZZER = 5

    ACTUATOR_1 = 6
    ACTUATOR_2 = 7

    CONVEYOR_EN = 8
    CONVEYOR_PWM = 9


class Inputs(IntEnum):
    START_BUTTON = 10
    STOP_BUTTON = 11
    PHOTOELECTRIC_SENSOR_1 = 12
    PHOTOELECTRIC_SENSOR_2 = 13
