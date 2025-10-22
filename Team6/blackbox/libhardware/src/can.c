/**
 * @file can.c
 * @brief SocketCAN을 사용하여 CAN 통신 기능을 구현하는 소스 파일.
 * @details
 * 이 파일은 hardware.h에 정의된 CAN API의 실제 동작을 정의합니다.
 * 리눅스 커널이 제공하는 표준 CAN 소켓 인터페이스를 사용하므로,
 * CAN 하드웨어가 리눅스에 올바르게 인식되어 있다면 어떤 장치에서든 동작합니다.
 * 핵심 특징은 '논블로킹(Non-blocking)' 모드로 동작하여, 메인 프로그램의 다른 작업을
 * 방해하지 않고 효율적으로 메시지를 수신할 수 있다는 점입니다.
 */

// --- 1. 필수 헤더 파일 포함 ---
#include <stdio.h>      // 표준 입출력 함수 (perror)
#include <string.h>     // 문자열 및 메모리 처리 함수 (strncpy, memcpy, memset)
#include <unistd.h>     // 유닉스 표준(POSIX) API (close, write, read)
#include <fcntl.h>      // 파일 제어 함수 (fcntl)
#include <sys/ioctl.h>  // 입출력 제어 함수 (ioctl)
#include <sys/socket.h> // 소켓 프로그래밍 함수 (socket, bind)
#include <net/if.h>     // 네트워크 인터페이스 구조체 (ifreq)
#include <linux/can.h>  // 리눅스 CAN 프로토콜 관련 정의 (PF_CAN, CAN_RAW, sockaddr_can, can_frame)
#include <linux/can/raw.h>// CAN RAW 소켓 관련 정의

#include "hardware.h"   // 이 파일에서 구현할 함수의 원형이 담긴 헤더
// #include "main.h"    // PID, 상태 플래그 정의를 포함하기 위해 필요할 수 있습니다.


// --- 2. 내부 전역 변수 ---
// 'static' 키워드는 이 변수가 can.c 파일 내부에서만 접근 가능하다는 것을 의미합니다. (캡슐화)
// CAN 통신에 사용될 소켓의 파일 디스크립터(fd)를 저장합니다.
// -1은 아직 초기화되지 않았거나 유효하지 않은 상태임을 나타내는 일반적인 관례입니다.
static int s_can_fd = -1;

/**
 * @brief CAN 인터페이스를 초기화하고 소켓을 준비합니다.
 * @param interface_name "can0"와 같은 CAN 인터페이스 이름.
 * @return 성공 시 CAN 소켓 파일 디스크립터(fd), 실패 시 -1.
 */
int can_init(const char* interface_name) {
    // 1. CAN RAW 소켓 생성
    s_can_fd = socket(PF_CAN, SOCK_RAW, CAN_RAW);
    if (s_can_fd < 0) {
        perror("socket(PF_CAN) error");
        return -1;
    }

    // 2. 사용할 CAN 인터페이스("can0", "can1" 등)의 인덱스 번호 찾기
    struct ifreq ifr = {0};
    strncpy(ifr.ifr_name, interface_name, IFNAMSIZ - 1);
    if (ioctl(s_can_fd, SIOCGIFINDEX, &ifr) < 0) {
        perror("ioctl(SIOCGIFINDEX) error");
        close(s_can_fd); s_can_fd = -1;
        return -1;
    }

    // 3. 소켓과 CAN 인터페이스를 바인딩(연결)
    struct sockaddr_can addr = {0};
    addr.can_family = AF_CAN;
    addr.can_ifindex = ifr.ifr_ifindex;
    if (bind(s_can_fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        perror("bind(can) error");
        close(s_can_fd); s_can_fd = -1;
        return -1;
    }
    
    // 4. 소켓을 논블로킹(Non-blocking) 모드로 설정 (중요!)
    // read() 함수가 읽을 데이터가 없을 때 기다리지 않고 즉시 리턴되도록 합니다.
    fcntl(s_can_fd, F_SETFL, O_NONBLOCK);
    
    return s_can_fd; // 성공 시, 생성된 파일 디스크립터를 직접 반환
}

int can_send_message(const CANMessage* msg) {
    if (s_can_fd < 0 || !msg) return -1; // 소켓이 유효하지 않거나 msg가 NULL이면 실패

    // CANMessage (API용 구조체) -> can_frame (리눅스 커널용 구조체) 변환
    struct can_frame frame = {0};
    frame.can_id = msg->id;
    frame.can_dlc = msg->dlc;
    memcpy(frame.data, msg->data, frame.can_dlc); // 데이터 복사

    // write() 시스템 콜을 통해 소켓으로 데이터를 전송합니다.
    int n = write(s_can_fd, &frame, sizeof(frame));
    // 요청한 크기만큼 정확히 전송되었는지 확인합니다.
    return (n == (int)sizeof(frame)) ? 0 : -1;
}

/**
 * @brief CAN 메시지를 수신합니다. (논블로킹)
 * @param msg 수신된 메시지를 저장할 CANMessage 구조체 포인터.
 * @return 1: 메시지 수신 성공, 0: 수신된 메시지 없음, <0: 에러 발생.
 */
int can_receive_message(CANMessage* msg) {
    if (s_can_fd < 0 || !msg) return -1; // 소켓이 유효하지 않거나 msg가 NULL이면 실패
    
    struct can_frame frame;
    // read() 시스템 콜을 통해 소켓으로부터 데이터를 읽어옵니다.
    int n = read(s_can_fd, &frame, sizeof(frame));

    // [논블로킹 로직의 핵심]
    // 읽을 데이터가 없으면 read()는 -1을 반환하고 errno를 EAGAIN 또는 EWOULDBLOCK으로 설정합니다.
    // 이 코드에서는 간단히 음수 값을 '수신 없음'으로 간주하여 0을 반환합니다.
    if (n < 0) return 0;

    // 읽어온 데이터가 정상적인 can_frame 크기보다 작으면 에러로 간주합니다.
    if (n < (int)sizeof(frame)) return -1;

    // can_frame (리눅스 커널용 구조체) -> CANMessage (API용 구조체) 변환
    msg->id = frame.can_id;
    msg->dlc = frame.can_dlc;
    memcpy(msg->data, frame.data, frame.can_dlc);
    
    return 1; // 메시지 1개 수신 성공
}


// ===================================================================================
// ====================== 아래부터 새로 추가/수정된 함수들 ========================
// ===================================================================================


/**
 * @brief [요청 함수] 특정 PID 정보를 요청하는 CAN 메시지를 전송합니다.
 * @details main.c에서는 이 함수를 호출하여 필요한 데이터를 ECU에 요청하기만 하면 됩니다.
 * @param pid 요청할 데이터의 Parameter ID (예: PID_VEHICLE_SPEED).
 * @return 성공 시 0, 실패 시 -1.
 */
int can_request_pid(unsigned char pid) {
    // 소켓이 초기화되었는지 먼저 확인합니다.
    if (s_can_fd < 0) return -1;

    // OBD-II 표준 진단 요청 프레임을 구성합니다.
    struct can_frame frame = {0};
    frame.can_id = 0x7DF; // 브로드캐스트 진단 요청 ID (모든 ECU에게 보냄)
    frame.can_dlc = 8;    // OBD-II 요청은 보통 8바이트를 모두 사용합니다.

    // 데이터 필드를 표준에 맞게 채웁니다.
    frame.data[0] = 0x02; // 응답에 필요한 데이터 바이트 수 (PID 값 포함 2바이트)
    frame.data[1] = 0x01; // 서비스 모드 01: 현재 실시간 데이터 요청
    frame.data[2] = pid;  // 실제 요청할 PID 값
    
    // 나머지 사용하지 않는 데이터 필드는 0x55(또는 0x00)와 같은 값으로 채워주는 것이 일반적입니다.
    memset(&frame.data[3], 0x00, 5);

    // write() 시스템 콜을 통해 소켓으로 완성된 프레임을 전송합니다.
    int n = write(s_can_fd, &frame, sizeof(frame));
    
    // 전송한 바이트 수가 실제 프레임 크기와 같은지 확인하여 성공 여부를 반환합니다.
    return (n == sizeof(frame)) ? 0 : -1;
}


/**
 * @brief [응답 해석 함수] 수신된 CAN 메시지를 파싱하여 VehicleData 구조체를 업데이트합니다.
 * @details main.c의 select() 루프에서 CAN 메시지가 수신될 때마다 이 함수를 호출하여,
 * 수신된 메시지가 우리가 기다리던 유효한 응답인지 확인하고 데이터를 갱신합니다.
 * @param msg can_receive_message()로 수신한 CAN 메시지 포인터.
 * @param vehicle_data 파싱된 데이터를 저장하고 업데이트할 차량 데이터 구조체 포인터.
 * @return 어떤 데이터가 업데이트되었는지 나타내는 상태 플래그. (예: STATE_SPEED_RECEIVED)
 * 유효한 응답이 아니면 0을 반환합니다.
 */
void can_parse_and_update_data(const CANMessage* msg, VehicleData* vehicle_data, unsigned char* flag, unsigned char* flag2) {
    // 1. 이 메시지가 ECU의 진단 응답이 맞는지 ID부터 확인합니다. (응답 ID 범위: 0x7E8 ~ 0x7EF)
    if (msg->id < 0x7E8 || msg->id > 0x7EF) {
        return; // 우리가 기다리던 진단 응답이 아니므로 무시.
    }

    // 2. 응답 데이터의 기본 형식이 맞는지 확인합니다.
    //    표준 응답 형식: [Byte 수, 0x41(서비스모드01응답), 요청PID, 값A, 값B, ...]
    if (msg->dlc < 4 || msg->data[1] != 0x41) {
        return; // 서비스 모드 01에 대한 응답(0x41)이 아니거나, 데이터 길이가 너무 짧으면 무시.
    }

    // 3. 이 응답이 어떤 PID에 대한 것인지 확인합니다.
    unsigned char responded_pid = msg->data[2];
    double temp = 0.0;
    float degree = 0.0;

    // 4. switch 문을 통해 PID에 맞는 파싱 로직을 수행합니다.
    switch (responded_pid) {
        case PID_VEHICLE_SPEED:
            // 차량 속도(PID 0x0D)의 계산식: 값 A
            vehicle_data->speed = msg->data[3];
            printf("\n[C] CAN received, PID : %x, vale: %d\n", PID_VEHICLE_SPEED, vehicle_data->speed);
            *flag |= VEHICLE_SPEED_FLAG; // '속도 수신 완료' 깃발 설정
            break; // switch 문 탈출

        case PID_ENGINE_SPEED:
            // 엔진 RPM(PID 0x0C)의 계산식: (A * 256 + B) / 4
            vehicle_data->rpm = ((int)msg->data[3] * 256 + (int)msg->data[4]) / 4;
            *flag |= ENGINE_SPEED_FLAG; // 'RPM 수신 완료' 깃발 설정
            break; // switch 문 탈출
        
        case PID_GEAR_STATE:
            vehicle_data->gear_ratio = ((float)(msg->data[3] * 256 + msg->data[4])) / 1000.0;
            
            switch(((msg->data[5] >> 4) & 0x0F)){
                //case 0x00: vehicle_data->gear_state = 'P'; break;
                //case 0x01: vehicle_data->gear_state = 'R'; break;
                //case 0x02: vehicle_data->gear_state = 'N'; break;
                //case 0x03: vehicle_data->gear_state = 'D'; break;
                //case 0x04: vehicle_data->gear_state = '1'; break;
                //case 0x05: vehicle_data->gear_state = '2'; break;
                //case 0x06: vehicle_data->gear_state = '3'; break;
                //case 0x07: vehicle_data->gear_state = '4'; break;
                //case 0x08: vehicle_data->gear_state = '5'; break;
                //case 0x09: vehicle_data->gear_state = '6'; break;
                //default:vehicle_data->gear_state = '?'; break;
                case 0x00: vehicle_data->gear_state = 'P'; break;
                case 0x01: vehicle_data->gear_state = 'D'; break; 
                case 0x02: vehicle_data->gear_state = 'R'; break;  
                default:   vehicle_data->gear_state = '?'; break;

            }

            *flag |= GEAR_STATE_FLAG;
            break;
        
        case PID_GPS_XDATA:
            temp = (double)msg->data[4] + (double)msg->data[5] / 100 + (double)msg->data[6] / 10000 + (double)msg->data[7] / 1000000;
            vehicle_data->gps_x = (msg->data[3] != 0) ? temp : -temp;
            *flag |= GPS_XDATA_FLAG;
            break;

        case PID_GPS_YDATA:
            temp = (double)msg->data[4] + (double)msg->data[5] / 100 + (double)msg->data[6] / 10000 + (double)msg->data[7] / 1000000;
            vehicle_data->gps_y = (msg->data[3] != 0) ? temp : -temp;
            *flag |= GPS_YDATA_FLAG;
            break;

        case PID_STEERING_DATA:
            degree = (float)msg->data[4] + ((float)msg->data[5]) / 100.0f;
            vehicle_data->degree = (msg->data[3] == 1) ? degree : -degree;
            *flag |= STEERING_DATA_FLAG;
            break;
        
        case PID_BRAKE_DATA:
            vehicle_data->brake_state = msg->data[3];
            *flag|= BRAKE_DATA_FLAG;
            break;
        
        case PID_TIRE_DATA:
            vehicle_data->tire_pressure[0] = msg->data[3]; vehicle_data->tire_pressure[1] = msg->data[4];
            vehicle_data->tire_pressure[2] = msg->data[5]; vehicle_data->tire_pressure[3] = msg->data[6];
            *flag |= TIRE_DATA_FLAG;
            break;
        
        case PID_THROTTLE_DATA:
            vehicle_data->throttle = msg->data[3];
            *flag2 |= THROTTLE_DATA_FLAG;
            break;
        // 여기에 다른 PID(GPS 위도, 경도, 조향각 등)에 대한 case를 계속 추가.
        // case PID_GPS_LATITUDE:
        //     ... 파싱 로직 ...
        //     new_state_flag = STATE_GPS_LAT_RECEIVED;
        //     break;

        default:
            // 우리가 요청하지 않았거나, 아직 처리 로직을 만들지 않은 PID는 그냥 무시.
            break;
    }

    // main.c에 어떤 데이터가 갱신되었는지 알려주기 위해 상태 플래그를 반환.

}


/**
 * @brief CAN 소켓을 닫고 자원을 정리합니다.
 */
void can_close() {
    // 소켓이 유효한 경우(-1이 아닌 경우)에만 닫기 작업을 수행합니다.
    if (s_can_fd >= 0) {
        close(s_can_fd);
        s_can_fd = -1; // 다시 유효하지 않은 상태로 설정
    }
}