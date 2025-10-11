"""
Smart Factory HW module controller
"""

from logging import getLogger
from os import listdir
from sys import platform
from time import sleep
from typing import Optional, Union

from iotdemo.common.debounce import debounce
from iotdemo.factory_controller.pins import Inputs, Outputs
from iotdemo.factory_controller.pyduino import PyDuino
from iotdemo.factory_controller.pyft232 import PyFt232

__all__ = ('FactoryController', )


class FactoryController:
    """
    Factory HW controller class
    """

    DEV_ON = False
    DEV_OFF = True

    def __init__(self,
                 port: Optional[Union[str, int]] = None,
                 *,
                 debug: bool = True):
        self.debug = debug
        if self.debug:
            self.logger = getLogger('CONTROLLER')

        self.__force_stop = False

        self.__device = None
        self.__device_name = None

        # open device
        self.port = self.__detect_serial(port if port is not None else -1)
        if self.port is not None and 'ttyUSB' in self.port:
            try:
                self.__device = PyFt232(self.port, debug=debug)
                self.__device_name = 'ft232'
            except Exception:
                self.__device = None
        else:
            try:
                self.__device = PyDuino(self.port, debug=debug)
                self.__device_name = 'arduino'
            except Exception:
                self.__device = None

            # Arduino beacon indicator
            self.red = False
            self.orange = True
            self.green = False

            self.defect_sensor_status = False
            self.color_sensor_status = False

            # interrupt handler
            if not self.is_dummy:
                self.__device.watch(Inputs.START_BUTTON,
                                    self.__button_interrupt)
                self.__device.watch(Inputs.STOP_BUTTON,
                                    self.__button_interrupt)

                self.__device.watch(Inputs.PHOTOELECTRIC_SENSOR_1,
                                    self.__sensor_interrupt)
                self.__device.watch(Inputs.PHOTOELECTRIC_SENSOR_2,
                                    self.__sensor_interrupt)

        if self.debug:
            self.logger.info(
                f'use {"Dummy" if self.is_dummy else "Arduino"} Controller')

    def __detect_serial(self, port):
        if isinstance(port, str):
            return port

        if not isinstance(port, int):
            raise RuntimeError(
                f'Invalid port argument type: {port} - {type(port)}')

        if platform in {'win32', 'cygwin'}:
            return f'COM{port}'

        if not platform.startswith('linux'):
            raise RuntimeError('Not supported OS')

        if port == -1:
            for path in listdir("/dev"):
                if path.startswith("ttyACM"):
                    return f"/dev/{path}"
        else:
            return f'/dev/ttyACM{port}'

        return None

    def __del__(self):
        self.close()

    def __enter__(self):
        if not self.is_dummy:
            self.system_start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.is_dummy:
            self.system_stop()
        self.close()

    def __set(self, pin, value):
        if self.is_dummy or self.__force_stop:
            return

        self.__device.set(pin, value)

    def __get(self, pin):
        if self.is_dummy:
            return None

        return self.__device.get(pin)

    def __led(self, pin, on):
        if self.__device_name == 'arduino':
            self.__set(
                pin,
                FactoryController.DEV_ON if on else FactoryController.DEV_OFF)

    def __actuator(self, actuator_id):
        if self.__device_name == 'arduino':
            self.__set(actuator_id, FactoryController.DEV_ON)
            sleep(0.1)
            self.__set(actuator_id, FactoryController.DEV_OFF)
        elif self.__device_name == 'ft232':
            self.__set(PyFt232.PKT_CMD_DETECTION, actuator_id)
            sleep(0.5)
            self.__set(PyFt232.PKT_CMD_DETECTION, PyFt232.PKT_CMD_DETECTION_0)

    @debounce(0.1)
    def __button_interrupt(self, pin: int, value: int, *_):
        if value != 0 or self.__force_stop:
            return

        if pin == Inputs.START_BUTTON:
            self.system_start()

        elif pin == Inputs.STOP_BUTTON:
            self.system_stop()

    @debounce(0.3)
    def __sensor_interrupt(self, pin: int, value: int, *_):
        status = value == 0

        if pin == Inputs.PHOTOELECTRIC_SENSOR_1:
            self.defect_sensor_status = status

        elif pin == Inputs.PHOTOELECTRIC_SENSOR_2:
            self.color_sensor_status = status

    ###########################################################################
    # Public properties

    @property
    def is_dummy(self):
        return self.__device is None

    @property
    def red(self):
        return bool(self.__get(Outputs.BEACON_RED))

    @red.setter
    def red(self, on):
        self.__led(Outputs.BEACON_RED, on)

    @property
    def orange(self):
        return bool(self.__get(Outputs.BEACON_ORANGE))

    @orange.setter
    def orange(self, on):
        self.__led(Outputs.BEACON_ORANGE, on)

    @property
    def green(self):
        return bool(self.__get(Outputs.BEACON_GREEN))

    @green.setter
    def green(self, on):
        self.__led(Outputs.BEACON_GREEN, on)

    @property
    def conveyor(self):
        return bool(self.__get(Outputs.CONVEYOR_EN))

    @conveyor.setter
    def conveyor(self, on):
        if on:
            self.__set(Outputs.CONVEYOR_EN, FactoryController.DEV_ON)
            self.__set(Outputs.CONVEYOR_PWM, 255)
        else:
            self.__set(Outputs.CONVEYOR_PWM, 0)
            self.__set(Outputs.CONVEYOR_EN, FactoryController.DEV_OFF)

    def push_actuator(self, num):
        if self.__device_name == 'arduino':
            if num == 1:
                self.__actuator(Outputs.ACTUATOR_1)
            else:
                self.__actuator(Outputs.ACTUATOR_2)
        elif self.__device_name == 'ft232':
            self.__actuator(num)

    ###########################################################################
    # Public methods

    def system_start(self) -> None:
        if self.debug:
            self.logger.info("Start System")

        if self.__device_name == 'ft232':
            self.__set(PyFt232.PKT_CMD_START, PyFt232.PKT_CMD_START_START)
            self.__set(PyFt232.PKT_CMD_SPEED, PyFt232.PKT_CMD_SPEED_UP)
            self.__set(PyFt232.PKT_CMD_SPEED, PyFt232.PKT_CMD_SPEED_UP)
            self.__set(PyFt232.PKT_CMD_SPEED, PyFt232.PKT_CMD_SPEED_UP)
            self.__set(PyFt232.PKT_CMD_SPEED, PyFt232.PKT_CMD_SPEED_UP)
        else:
            self.red = False
            self.green = True
            self.conveyor = True

    def system_stop(self) -> None:
        if self.debug:
            self.logger.info("Stop System")

        if self.__device_name == 'ft232':
            self.__set(PyFt232.PKT_CMD_START, PyFt232.PKT_CMD_START_STOP)
        else:
            self.red = True
            self.green = False
            self.conveyor = False

    def close(self):
        if self.debug:
            self.logger.info("Shutdown System")

        self.__force_stop = True
        if self.is_dummy:
            return

        self.__device.close()
