"""
Raw level arduino GPIO controll module
"""

from logging import getLogger
from queue import Queue
from threading import Condition, Lock, Thread
from typing import Callable, Optional

from serial import Serial

__all__ = ('PyDuino', )


def _dummy(*_):
    pass


class PyDuino:
    """
    Raw level arduino GPIO controll class
    """

    PACKET_START = 0x55
    PACKET_END = 0xaa

    PACKET_ID_INTERRUPT = 0
    PACKET_ID_MAX = 199
    PACKET_ID_ERROR = 0xff

    def __init__(self,
                 port: str,
                 baudrate: int = 115200,
                 *,
                 debug: bool = False):
        self.debug = debug
        if self.debug:
            self.logger = getLogger('PYDUINO')
            self.logger.info("serial port %s (%d)", port, baudrate)

        self.__arduino = None
        self.__stop_requested = False  # to support graceful exit
        self.__force_stop = False
        self.__watcher = {}
        self.__values = [1] * 70  # Pin status
        for idx in (0, 1):
            self.__values[idx] = 0xff  # NC

        # init arduio
        arduino = Serial(port=port, baudrate=baudrate, timeout=1)
        arduino.reset_input_buffer()
        arduino.reset_output_buffer()
        self.__arduino = arduino

        self.__init_cv = Condition()
        self.__init_input_pins = (10, 11, 12, 13)
        self.__init_count = 0

        self.__rx = Thread(target=self.__receiver)
        self.__rx.name = 'Arduino Rx'
        self.__rx.start()

        self.__init()

        self.__tx_lock = Lock()
        self.__tx_cv = Condition()
        self.__tx_queue = Queue()
        self.__tx_packet_id = 1
        self.__tx = Thread(target=self.__transmitter)
        self.__tx.name = 'Arduino Tx'
        self.__tx.start()

    def __del__(self):
        self.close()

    def __wait_for_input_status(self, pin: int, value: int, _):
        self.__values[pin] = value

        with self.__init_cv:
            self.__init_count += 1
            if self.__init_count == len(self.__init_input_pins):
                self.__init_cv.notify()

    def __init(self):
        with self.__init_cv:
            self.__init_cv.wait()

        for idx in self.__init_input_pins:
            self.watch(idx)

# pyduino.py - inside class PyDuino

    def __transmitter(self):
        arduino = self.__arduino

        while True:
            packet = self.__tx_queue.get()
            if self.debug:
                self.logger.info(packet)

            pin, value, packet_id = packet
            self.__tx_queue.task_done()
            if pin is None:
                break  # End of Tx

            pin = int(pin) & 0xff
            value = int(value) & 0xff

            try:
                arduino.write(bytes([
                    PyDuino.PACKET_START, packet_id, pin, value,
                    PyDuino.PACKET_END
                ]))
            except Exception as e:
                # 치명 오류 시: 강제 정지하고 Tx 종료
                try:
                    from serial import SerialException
                except Exception:
                    SerialException = Exception
                if isinstance(e, (SerialException, OSError)):
                    if self.debug:
                        try: self.logger.warning("Tx write failed / serial down: %s", e)
                        except Exception: pass
                    self.__force_stop = True
                    # set()에서 wait중일 수 있으니 깨워준다
                    try:
                        with self.__tx_cv:
                            self.__tx_cv.notify_all()
                    except Exception:
                        pass
                    break
                # 기타 오류는 약간 쉬고 재시도(선택)
                import time; time.sleep(0.02)

    # Rx Thread
    def __receiver(self):
        import time
        # 초기 입력핀 감시 등록(기존 로직 유지)
        for idx in self.__init_input_pins:
            self.__values[idx] = 0xff
            self.watch(idx, self.__wait_for_input_status)

        while self.__force_stop is False:
            try:
                packet = self.__receive_packet()
            except Exception as e:
                # 시리얼 분리/중복접근 등 치명 오류 → 1회 로깅 후 조용히 종료
                try:
                    from serial import SerialException
                except Exception:
                    SerialException = Exception
                if isinstance(e, (SerialException, OSError)):
                    if self.debug:
                        try: self.logger.warning("Serial disconnected/busy: %s", e)
                        except Exception: pass
                    # 상태 플래그(옵션)
                    try: self._connected = False
                    except Exception: pass
                    break
                # 기타 오류는 소프트 리트라이
                time.sleep(0.02)
                continue

            if packet is None:
                # 타임아웃/빈 읽기 → 살짝 쉬고 계속
                time.sleep(0.02)
                continue

            # 정상 패킷 처리
            pin, value, packet_id = packet
            self.__values[pin] = value

            if packet_id == PyDuino.PACKET_ID_INTERRUPT:
                self.__watcher.get(pin, _dummy)(*packet)
            else:
                with self.__tx_cv:
                    self.__tx_cv.notify()

    def __receive_packet(self):
        arduino = self.__arduino

        # 안전 타임아웃 보정(0/None이면 짧게 설정)
        try:
            if getattr(arduino, "timeout", None) in (None, 0):
                arduino.timeout = 0.2
        except Exception:
            pass

        # 첫 바이트 읽기
        try:
            data = arduino.read()
        except Exception:
            # 상위에서 예외 처리
            raise
        if not data or data[0] != PyDuino.PACKET_START:
            return None

        # 나머지 4바이트 조립(타임아웃이면 None 반환)
        packet = []
        while len(packet) != 4:
            try:
                data = arduino.read()
            except Exception:
                raise
            if not data:
                return None  # Timeout → 상위 루프가 부드럽게 스킵
            packet.append(data[0])

        packet_id, pin, value, packet_end = packet
        if packet_id < 0 or packet_id > PyDuino.PACKET_ID_ERROR:
            return None

        if pin < 0 or pin > len(self.__values):
            return None

        if packet_end != PyDuino.PACKET_END:
            return None

        return (pin, value, packet_id)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self) -> None:
        """
        Close existing arduino connection
        """
        if self.__arduino is None:
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
        self.__arduino.close()

        if self.debug:
            self.logger.info("stopped")

    # pyduino.py - inside class PyDuino

    def set(self, pin: int, value: int) -> bool:
        """
        Set arduino pin with given value
        """
        if self.__stop_requested or self.__force_stop:
            return False

        with self.__tx_lock:
            packet_id = self.__tx_packet_id

            ok = False
            with self.__tx_cv:
                self.__tx_queue.put((pin, value, packet_id))
                # wait()에 타임아웃 추가 (예: 0.5s)
                self.__tx_cv.wait(timeout=0.5)
                ok = not self.__force_stop  # Rx/Tx가 죽었으면 False

            self.__tx_packet_id += 1
            if self.__tx_packet_id == PyDuino.PACKET_ID_MAX:
                self.__tx_packet_id = 1

        return ok

    def get(self, pin: int) -> int:
        """
        Get current arduino pin value from proxy
        """
        return self.__values[pin]

    def watch(
            self,
            pin: int,
            callback: Optional[Callable[[int, int, int],
                                        None]] = None) -> None:
        """
        Set callback function for given pin
        """
        if callback is None:
            callback = _dummy

        self.__watcher[pin] = callback
