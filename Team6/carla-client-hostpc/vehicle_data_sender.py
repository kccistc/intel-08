# File: vehicle_data_sender.py
import serial
import time

class VehicleDataSender:
    def __init__(self, serial_port, baud_rate):
        """시리얼 포트를 초기화합니다."""
        try:
            self.ser = serial.Serial(serial_port, baud_rate, timeout=1)
            print(f"✅ Serial port {serial_port} is ready.")
        except serial.SerialException as e:
            raise Exception(f"Failed to open serial port {serial_port}: {e}")

    def send_data(self, vehicle):
        """차량 데이터를 약속된 프로토콜로 만들어 시리얼로 전송합니다."""
        velocity = vehicle.get_velocity()
        speed_kmh = int(3.6 * (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5)
        
        transform = vehicle.get_transform()
        steering_angle = transform.rotation.yaw
        
        # 예시 프로토콜: "S<속도>,A<조향각>\n"
        data_string = f"S{speed_kmh},A{steering_angle:.2f}\n"
        
        self.ser.write(data_string.encode('utf-8'))

    def close(self):
        """시리얼 포트를 닫습니다."""
        self.ser.close()