"""
Raw level FT232 controll module
"""

from logging import getLogger
from queue import Queue
from threading import Condition, Lock, Thread
from typing import Callable, Optional

from serial import Serial

__all__ = ('PyFt232', )


def _dummy(*_):
    pass


class PyFt232:
    """
    Raw level FT232 controll class
    """

    PKT_START = 0xaa
    PKT_END = 0xaa

    PKT_CMD_ACK = 0x00

    PKT_CMD_START = 0x01
    PKT_CMD_START_STOP = 0x00
    PKT_CMD_START_START = 0x01

    PKT_CMD_LOADING = 0x02
    PKT_CMD_LOADING_STOP = 0x00
    PKT_CMD_LOADING_LOAD = 0x01

    PKT_CMD_DETECTION = 0x03
    PKT_CMD_DETECTION_0 = 0x00
    PKT_CMD_DETECTION_1 = 0x01
    PKT_CMD_DETECTION_2 = 0x02
    PKT_CMD_DETECTION_3 = 0x03

    PKT_CMD_SPEED = 0x04
    PKT_CMD_SPEED_STOP = 0x00
    PKT_CMD_SPEED_UP = 0x01
    PKT_CMD_SPEED_DOWN = 0x02

    #PKT_CMD_AUTO_LOADING = 0x05
    #PKT_CMD_VERSION = 0x64

    PKT_ID_MAX = 199
    PKT_ID_ERROR = 0xff

    def __init__(self,
                 port: str = '/dev/ttyUSB0',
                 baudrate: int = 19200,
                 *,
                 debug: bool = False):
        self.debug = debug
        if self.debug:
            self.logger = getLogger('PYFT232')
            self.logger.info("serial port %s (%d)", port, baudrate)

        self.__ft232 = None
        self.__stop_requested = False  # to support graceful exit
        self.__force_stop = False
        self.__values = [1] * 14  # Pin status
        for idx in (0, 1):
            self.__values[idx] = 0xff  # NC

        # init arduio
        ft232 = Serial(port=port, baudrate=baudrate, timeout=1)
        ft232.reset_input_buffer()
        ft232.reset_output_buffer()
        self.__ft232 = ft232

        self.__rx = Thread(target=self.__receiver)
        self.__rx.name = 'Arduino Rx'
        self.__rx.start()

        self.__init()

        self.__tx_lock = Lock()
        self.__tx_cv = Condition()
        self.__tx_queue = Queue()
        self.__tx = Thread(target=self.__transmitter)
        self.__tx.name = 'Arduino Tx'
        self.__tx.start()

    def __del__(self):
        self.close()

    def __checksum(self, length: int, cmd: int, data: list):
        return ~((length + cmd + sum(data)) & 0xff) & 0xff

    def __init(self):
        pass

    # Tx Thread
    def __transmitter(self):
        ft232 = self.__ft232

        while True:
            packet = self.__tx_queue.get()
            if self.debug:
                self.logger.info(packet)

            if len(packet) != 2:
                break

            cmd, data = packet
            self.__tx_queue.task_done()
            if cmd is None:
                break  # End of Tx

            ft232.write(
                bytes([
                    PyFt232.PKT_START, 0x05, cmd, data, self.__checksum(0x05, cmd, [data])
                ]))

    # Rx Thread
    def __receiver(self):
        while self.__force_stop is False:
            packet = self.__receive_packet()
            if self.debug:
                self.logger.info(packet)
            if packet is None:
                continue

            # TODO : What need to be done?
            cmd, data = packet

            # run callback
            with self.__tx_cv:
                self.__tx_cv.notify()

    def __receive_packet(self):
        ft232 = self.__ft232

        data = ft232.read()
        if len(data) == 0 or data[0] != PyFt232.PKT_START:
            return None

        packet = []
        while len(packet) != 5:
            data = ft232.read()
            if len(data) == 0:
                return None  # Timeout

            packet.append(data[0])

        length, cmd, data, dataerr, checksum = packet
        if length != 6:
            return None

        if cmd != 0x00 or dataerr != 0x00:
            return None

        if self.__checksum(length, cmd, [data, dataerr]) != checksum:
            return None

        return (cmd, data)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self) -> None:
        """
        Close existing ft232 connection
        """
        if self.__ft232 is None:
            return

        self.__stop_requested = True
        if self.__force_stop:
            return

        # post end of tx marker
        self.__tx_queue.put((None, None, None))
        self.__tx.join()

        # Stop Rx
        self.__force_stop = True
        self.__rx.join()

        # close serial
        self.__ft232.close()

        if self.debug:
            self.logger.info("stopped")

    def set(self, cmd: int, value: int) -> bool:
        """
        Set ft232 pin with given value
        """
        if self.__stop_requested:
            return False

        with self.__tx_lock:
            with self.__tx_cv:
                self.__tx_queue.put((cmd, value))
                self.__tx_cv.wait()

        return True

    def get(self, pin: int) -> int:
        """
        Get current ft232 pin value from proxy
        """
        return self.__values[pin]
