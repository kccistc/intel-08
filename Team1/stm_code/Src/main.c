/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2025 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <stdio.h>
#include <string.h>

#include "ssd1306.h"      // <-- OLED 라이브러리 헤더 추가
#include "ssd1306_fonts.h"// <-- OLED 폰트 헤더 추가
#include <math.h>

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
#ifdef __GNUC__
#define PUTCHAR_PROTOTYPE int __io_putchar(int ch)
#else
#define PUTCHAR_PROTOTYPE int fputc(int ch, FILE *f)
#endif /* __GNUC__ */
#define ARR_CNT 5
#define CMD_SIZE 50

// F4xx TIM2 Period (9999)에 맞춰 최대 속도 PWM 값 정의 (약 10000)
#define MAX_PWM_SPEED 9000
#define MIN_PWM_SPEED 500



/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */
// 오른쪽 바퀴 방향 제어 핀 (PB4, PB5)
#define Moter2_In1_Pin GPIO_PIN_4
#define Moter2_In1_GPIO_Port GPIOB
#define Moter1_In2_Pin GPIO_PIN_5
#define Moter1_In2_GPIO_Port GPIOB

// 왼쪽 바퀴 방향 제어 핀 (PA6, PA7)
#define Moter3_In3_Pin GPIO_PIN_6
#define Moter3_In3_GPIO_Port GPIOA
#define Moter4_In4_Pin GPIO_PIN_7
#define Moter4_In4_GPIO_Port GPIOA

#define RIGHT_WHEEL_PWM_CHANNEL TIM_CHANNEL_1 // ENA (TIM2_CH1)
#define LEFT_WHEEL_PWM_CHANNEL TIM_CHANNEL_2  // ENB (TIM2_CH2)

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
I2C_HandleTypeDef hi2c1;

TIM_HandleTypeDef htim2;

UART_HandleTypeDef huart1;
UART_HandleTypeDef huart2;
UART_HandleTypeDef huart3;

/* USER CODE BEGIN PV */
volatile uint16_t right_motor_pwm = 0; // 오른쪽 모터 PWM 값
volatile uint16_t left_motor_pwm = 0;  // 왼쪽 모터 PWM 값



uint8_t tx_buf[50];
uint8_t rx_data;
int count = 0;


uint8_t rx2char;
volatile unsigned char rx2Flag = 0;
volatile char rx2Data[50];
#define UART3_RX_BUFFER_SIZE 50

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_USART2_UART_Init(void);
static void MX_I2C1_Init(void);
static void MX_TIM2_Init(void);
static void MX_USART1_UART_Init(void);
static void MX_USART3_UART_Init(void);
/* USER CODE BEGIN PFP */
void Motor_Right_Forward(uint16_t speed);
void Motor_Left_Forward(uint16_t speed);
void Motor_Forward(uint16_t speed); // 전체 전진 추가
void Motor_Stop(void);
void Motor_Right_Stop(void); // 오른쪽 정지 함수 추가
void Motor_Left_Stop(void);
/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

/**
  * @brief 오른쪽 모터를 전진 방향으로 구동합니다.
  * @param speed: PWM duty cycle (0 ~ 9999)
  */
void Motor_Right_Forward(uint16_t speed) {
    // 속도 제한
    if (speed > 9999) speed = 9999;
    right_motor_pwm = speed;

    // 방향 제어 (In1=SET, In2=RESET)
    HAL_GPIO_WritePin(Moter2_In1_GPIO_Port, Moter2_In1_Pin, GPIO_PIN_SET);
    HAL_GPIO_WritePin(Moter1_In2_GPIO_Port, Moter1_In2_Pin, GPIO_PIN_RESET);
    // PWM 속도 설정
    __HAL_TIM_SET_COMPARE(&htim2, RIGHT_WHEEL_PWM_CHANNEL, speed);
}

/**
  * @brief 오른쪽 모터를 정지합니다.
  */
void Motor_Right_Stop(void) {
    right_motor_pwm = 0;
    // PWM 속도를 0으로 설정
    __HAL_TIM_SET_COMPARE(&htim2, RIGHT_WHEEL_PWM_CHANNEL, 0);
    // 방향 핀도 모두 LOW로 설정하여 안전하게 정지
    HAL_GPIO_WritePin(Moter2_In1_GPIO_Port, Moter2_In1_Pin, GPIO_PIN_RESET);
    HAL_GPIO_WritePin(Moter1_In2_GPIO_Port, Moter1_In2_Pin, GPIO_PIN_RESET);
}

/**
  * @brief 왼쪽 모터를 전진 방향으로 구동합니다.
  * @param speed: PWM duty cycle (0 ~ 9999)
  */
void Motor_Left_Forward(uint16_t speed) {
    // 속도 제한
    if (speed > 9999) speed = 9999;
    left_motor_pwm = speed;

    // 방향 제어 (In3=SET, In4=RESET)
    HAL_GPIO_WritePin(Moter3_In3_GPIO_Port, Moter3_In3_Pin, GPIO_PIN_SET);
    HAL_GPIO_WritePin(Moter4_In4_GPIO_Port, Moter4_In4_Pin, GPIO_PIN_RESET);

    // PWM 속도 설정
    __HAL_TIM_SET_COMPARE(&htim2, LEFT_WHEEL_PWM_CHANNEL, speed);
}

/**
  * @brief 왼쪽 모터를 정지합니다.
  */
void Motor_Left_Stop(void) {
    left_motor_pwm = 0;
    // PWM 속도를 0으로 설정
    __HAL_TIM_SET_COMPARE(&htim2, LEFT_WHEEL_PWM_CHANNEL, 0);
    // 방향 핀도 모두 LOW로 설정하여 안전하게 정지
    HAL_GPIO_WritePin(Moter3_In3_GPIO_Port, Moter3_In3_Pin, GPIO_PIN_RESET);
    HAL_GPIO_WritePin(Moter4_In4_GPIO_Port, Moter4_In4_Pin, GPIO_PIN_RESET);
}

/**
  * @brief 양쪽 모터를 동시에 전진 구동합니다.
  * @param speed: PWM duty cycle (0 ~ 9999)
  */
void Motor_Forward(uint16_t speed) {
    Motor_Right_Forward(speed);
    Motor_Left_Forward(speed);
}

/**
  * @brief 양쪽 모터를 동시에 정지합니다. (PFP에 있던 Motor_Stop 정의)
  */
void Motor_Stop(void) {
    Motor_Right_Stop();
    Motor_Left_Stop();
}


/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_USART2_UART_Init();
  MX_I2C1_Init();
  MX_TIM2_Init();
  MX_USART1_UART_Init();
  MX_USART3_UART_Init();
  /* USER CODE BEGIN 2 */



  printf("STM32F4xx Motor Control Start\r\n");

   // PWM 시작
   // *주의: PA6/PA7이 방향 제어 핀(GPIO)과 PWM 핀(TIM2_CH1/CH2)으로 동시에 설정되면 충돌이 발생합니다.
   // PWM 핀은 CubeMX에서 확인하여 PA5(CH1)와 PA1(CH2) 같은 Alternate Function 핀으로 올바르게 설정되었는지 확인하세요.*
   HAL_TIM_PWM_Start(&htim2, RIGHT_WHEEL_PWM_CHANNEL); // 오른쪽 바퀴 PWM (TIM2_CH1)
   HAL_TIM_PWM_Start(&htim2, LEFT_WHEEL_PWM_CHANNEL);  // 왼쪽 바퀴 PWM (TIM2_CH2)


   ssd1306_Init(); // OLED 초기화
   ssd1306_Fill(Black); // 화면 전체를 검은색으로 지웁니다.
   ssd1306_SetCursor(5, 5); // 글씨를 쓸 시작 위치(x, y)를 정합니다.
   ssd1306_WriteString("AI CAR Project", Font_7x10, White); // "AI CAR Project" 문구 표시
   ssd1306_SetCursor(25, 20);
   ssd1306_WriteString("System Ready!", Font_7x10, White);

   ssd1306_UpdateScreen(); // 화면에 실제로 내용을 업데이트합니다. (필수)
   HAL_Delay(2000); // 2초간 시작 메시지를 보여줍니다.


   // 모터 속도 제어 변수 (F103 코드에서 가져옴)
   uint32_t motor_update_tick = 0;
   uint16_t current_speed = 500; // MIN_PWM_SPEED로 시작
   uint8_t is_accelerating = 1; // 1: 가속 중, 0: 감속 중

   // 초기 모터 상태 정지
   Motor_Stop();


   // ... (UART 초기화 코드 생략)



     uint8_t rx_char;
     char rx_buffer[50];
     uint8_t idx = 0;


   // ... (모터, UART2, TIM4, OLED 초기화 코드 생략)

   printf("start main2()\r\n");

   // ... (초기 OLED 출력 및 Delay 코드 생략)

   // ... (현재 속도 및 가속 변수 정의 코드 생략)



  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
   uint32_t can_tx_tick = 0;
     uint8_t data_counter = 0; // 데이터의 변화를 확인하기 위한 카운터

  while (1)
  {



    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
	  uint8_t rx_char;
	  	      static char rx_buffer[50];
	  	      static uint8_t idx = 0;
	  	      // 라즈베리 파이(huart1)로부터 1글자씩 데이터를 받습니다.
	  	      	      if (HAL_UART_Receive(&huart1, &rx_char, 1, 10) == HAL_OK)
	  	      	      {
	  	      	          // 2. 만약 마지막 문자인 '\n'을 받았다면, 메시지 처리를 시작합니다.
	  	      	          if (rx_char == '\n')
	  	      	          {
	  	      	              rx_buffer[idx] = '\0'; // 문자열의 끝을 표시합니다.

	  	      	              int speed = 0;
	  	      	              int direction = 0;

	  	      	              // 3. 수신된 문자열("S,speed,direction")을 파싱합니다.
	  	      	              if (sscanf(rx_buffer, "S,%d,%d", &speed, &direction) == 2)
	  	      	              {
	  	      	                  // 파싱에 성공하면 아래 모든 작업을 순서대로 실행합니다.

	  	      	                  // 디버깅용으로 PC에 수신된 값을 출력합니다.
	  	      	                  printf("Received: S,%d,%d (Direction Ignored)\r\n", speed, direction);

	  	      	                  // 4. speed 값(0~100)을 실제 모터 PWM 값(0~9000)으로 변환합니다.
	  	      	                  uint16_t pwm_value = (uint16_t)(speed * (MAX_PWM_SPEED / 100.0f));
	  	      	                  if (pwm_value > MAX_PWM_SPEED) {
	  	      	                      pwm_value = MAX_PWM_SPEED;
	  	      	                  }

	  	      	                  // 5. 계산된 PWM 값으로 양쪽 모터를 동일하게 구동합니다.
	  	      	                  Motor_Right_Forward(pwm_value);
	  	      	                  Motor_Left_Forward(pwm_value);

	  	      	                  // === OLED 계기판 모양 출력 코드 시작 ===

	  	      	              char oled_buffer[20];
	  	      	              	      	              ssd1306_Fill(Black);

	  	      	              	      	              // 계기판 중심 좌표와 반지름 설정
	  	      	              	      	              int center_x = 64;
	  	      	              	      	              int center_y = 60;
	  	      	              	      	              int radius_outer = 30; // 바깥쪽 반지름
	  	      	              	      	              int radius_inner = 25; // 눈금용 안쪽 반지름
	  	      	              	      	              int radius_text = 35;  // 텍스트용 반지름

	  	      	              	      	              // 1. 외곽선 및 눈금 그리기 (개선)

	  	      	              	      	              // 전체 180도 구간을 10단계(0~100%)로 나누어 눈금을 그립니다.
	  	      	              	      	              for (int i = 0; i <= 10; i++) {
	  	      	              	      	                  // 눈금의 각도 계산 (180도 ~ 0도)
	  	      	              	      	                  float tick_ratio = (float)i / 10.0f; // 0.0, 0.1, ..., 1.0
	  	      	              	      	                  float tick_angle_rad = (1.0f - tick_ratio) * 3.14159f;

	  	      	              	      	                  // 눈금의 바깥쪽 끝점 (외곽선 역할)
	  	      	              	      	                  int outer_x = center_x + (int)(radius_outer * cosf(tick_angle_rad));
	  	      	              	      	                  int outer_y = center_y - (int)(radius_outer * sinf(tick_angle_rad));

	  	      	              	      	                  // 눈금의 안쪽 끝점
	  	      	              	      	                  int inner_x;
	  	      	              	      	                  int inner_y;

	  	      	              	      	                  // 50% 단위 (5칸)는 긴 눈금, 나머지는 짧은 눈금
	  	      	              	      	                  if (i % 5 == 0) {
	  	      	              	      	                      // 긴 눈금 (주요 눈금)
	  	      	              	      	                      inner_x = center_x + (int)((radius_outer - 5) * cosf(tick_angle_rad));
	  	      	              	      	                      inner_y = center_y - (int)((radius_outer - 5) * sinf(tick_angle_rad));

	  	      	              	      	                      // 눈금 값(0, 50, 100) 텍스트 출력
	  	      	              	      	                      ssd1306_SetCursor(center_x + (int)(radius_text * cosf(tick_angle_rad)) - 5,
	  	      	              	      	                                        center_y - (int)(radius_text * sinf(tick_angle_rad)) - 3);
	  	      	              	      	                      if (i == 0) {
	  	      	              	      	                         ssd1306_WriteString("0%", Font_6x8, White);
	  	      	              	      	                      } else if (i == 5) {
	  	      	              	      	                         ssd1306_WriteString("50%", Font_6x8, White);
	  	      	              	      	                      } else if (i == 10) {
	  	      	              	      	                         ssd1306_WriteString("100%", Font_6x8, White);
	  	      	              	      	                      }

	  	      	              	      	                  } else {
	  	      	              	      	                      // 짧은 눈금
	  	      	              	      	                      inner_x = center_x + (int)((radius_outer - 3) * cosf(tick_angle_rad));
	  	      	              	      	                      inner_y = center_y - (int)((radius_outer - 3) * sinf(tick_angle_rad));
	  	      	              	      	                  }

	  	      	              	      	                  // 눈금 선 그리기
	  	      	              	      	                  ssd1306_Line(outer_x, outer_y, inner_x, inner_y, White);
	  	      	              	      	              }


	  	      	              	      	              // 2. 바늘 각도 계산
	  	      	              	      	              // 현재 PWM 값의 비율 (0.0 ~ 1.0)
	  	      	              	      	              float ratio = (float)pwm_value / (float)MAX_PWM_SPEED;
	  	      	              	      	              // 바늘 각도 계산 (라디안) - 1.0f-ratio로 왼쪽(MAX)에서 오른쪽(MIN)으로 움직이도록 조정
	  	      	              	      	              float angle_rad = (1.0f - ratio) * 3.14159f;

	  	      	              	      	              // 바늘 끝점 좌표 계산. 바늘 길이는 (radius_outer - 2)
	  	      	              	      	              int needle_len = radius_outer - 2;
	  	      	              	      	              int needle_end_x = center_x + (int)(needle_len * cosf(angle_rad));
	  	      	              	      	              int needle_end_y = center_y - (int)(needle_len * sinf(angle_rad));

	  	      	              	      	              // 3. 바늘 그리기 (얇게)
	  	      	              	      	              ssd1306_Line(center_x, center_y, needle_end_x, needle_end_y, White);

	  	      	              	      	              // 4. 바늘 중심에 작은 원 그리기 (좀 더 계기판처럼 보이게)
	  	      	              	      	              ssd1306_DrawCircle(center_x, center_y, 2, White); // DrawCircle 함수가 있다면 사용

	  	      	              	      	              // 5. 현재 PWM 값 텍스트 출력 (가장 큰 글씨)
	  	      	              	      	              ssd1306_SetCursor(center_x - 18, 5); // 화면 상단에 속도 표시
	  	      	              	      	              sprintf(oled_buffer, "%u", pwm_value);
	  	      	              	      	              ssd1306_WriteString(oled_buffer, Font_11x18, White);

	  	      	              	      	              ssd1306_SetCursor(center_x - 18, 25);
	  	      	              	      	              ssd1306_WriteString("PWM", Font_7x10, White);


	  	      	              	      	              ssd1306_UpdateScreen(); // 화면 업데이트
	  	      	              	      	              // === OLED 계기판 모양 출력 코드 끝 ===

	  	      	                  // 6. 계산된 PWM 값을 CAN 대신 **UART3**으로 전송합니다.

	  	                            // 전송할 메시지를 문자열로 구성 (예: "PWM_TX: 3500\n")
	  	                            char uart3_tx_buffer[64];
	  	                            int len = sprintf(uart3_tx_buffer, "PWM_TX: %u\r\n", pwm_value); // 2바이트 PWM 값을 문자열로 변환

	  	                            //  UART3 전송
	  	      	                  if (HAL_UART_Transmit(&huart3, (uint8_t*)uart3_tx_buffer, len, 100) == HAL_OK)
	  	      	                  {
	  	      	                      printf("UART3 Tx -> Sent: %s", uart3_tx_buffer);
	  	      	                  }
	  	      	                  else
	  	      	                  {
	  	      	                      printf("UART3 Tx -> FAILED\r\n");
	  	      	                  }
	  	      	              }

	  	      	              // 7. 다음 메시지를 받기 위해 버퍼 인덱스를 초기화합니다.
	  	      	              idx = 0;
	  	      	          }
	  	      	          else // 1. 아직 메시지가 끝나지 않았다면( '\n'이 아니라면) 버퍼에 글자를 추가합니다.
	  	      	          {
	  	      	              if (idx < (sizeof(rx_buffer) - 1))
	  	      	              {
	  	      	                  rx_buffer[idx++] = rx_char;
	  	      	              }
	  	      	          }
	  	      	      }


  }
}


	  //CAN TEST 코드
//	  TxHeader.StdId = 0x123;      // 테스트용 ID
//	    TxHeader.RTR = CAN_RTR_DATA;
//	    TxHeader.IDE = CAN_ID_STD;
//	    TxHeader.DLC = 2;            // 데이터 길이 2바이트
//	    TxData[0] = 0xAA;            // 보낼 데이터 1
//	    TxData[1] = 0x55;            // 보낼 데이터 2
//
//	    // 2. 메시지 전송
//	    if (HAL_CAN_AddTxMessage(&hcan1, &TxHeader, TxData, &TxMailbox) == HAL_OK)
//	    {                                                                                                                                                                                                                                                   ,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
//	        printf("CAN Tx -> Message sent successfully.\r\n");
//
//	        // 3. 메시지가 수신될 때까지 잠시 대기 (약 100ms)
//	        uint32_t tickstart = HAL_GetTick();
//	        while (HAL_CAN_GetRxFifoFillLevel(&hcan1, CAN_RX_FIFO0) == 0)
//	        {
//	            if ((HAL_GetTick() - tickstart) > 100)
//	            {
//	                printf("CAN Rx -> FAILED (Timeout).\r\n");
//	                break;
//	            }
//	        }
//
//	        // 4. 수신된 메시지가 있는지 확인
//	        if (HAL_CAN_GetRxFifoFillLevel(&hcan1, CAN_RX_FIFO0) > 0)
//	        {
//	            if (HAL_CAN_GetRxMessage(&hcan1, CAN_RX_FIFO0, &RxHeader, RxData) == HAL_OK)
//	            {
//	                // 5. 보낸 데이터와 받은 데이터가 일치하는지 확인
//	                if (RxHeader.StdId == 0x123 && RxData[0] == 0xAA && RxData[1] == 0x55)
//	                {
//	                    printf("CAN Rx -> SUCCESS! Loopback OK.\r\n\n");
//	                    HAL_GPIO_TogglePin(LD2_GPIO_Port, LD2_Pin); // 성공 시 LED 깜빡임
//	                }
//	                else
//	                {
//	                    printf("CAN Rx -> FAILED (Data mismatch).\r\n\n");
//	                }
//	            }
//	        }
//	    }
//	    else
//	    {
//	        printf("CAN Tx -> FAILED to send message.\r\n");
//	    }
//
//	    HAL_Delay(1000); // 1초마다 테스트 반복
//	  }
//











    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
  //
  //
  //	    if(HAL_UART_Receive(&huart1, &rx_char, 1, 100) == HAL_OK)
  //	    {
  //	        if(rx_char == '\n') // 명령 종료 문자
  //	        {
  //	            rx_buffer[idx] = '\0'; // 문자열 종료
  //	            idx = 0;
  //
  //	            // 예: "S,50,70"
  //	            int speed, direction;
  //	            if(sscanf(rx_buffer, "S,%d,%d", &speed, &direction) == 2)
  //	            {
  //	            	printf("Received from ROS -> Speed: %d, Direction: %d\r\n", speed, direction);
  //	                // 방향 변환
  //	                uint16_t right_pwm = speed + (direction - 50);
  //	                uint16_t left_pwm  = speed - (direction - 50);
  //
  //	                // 안전 범위 제한
  //	                if(right_pwm > MAX_PWM_SPEED) right_pwm = MAX_PWM_SPEED;
  //	                if(left_pwm > MAX_PWM_SPEED)  left_pwm  = MAX_PWM_SPEED;
  //
  //	                // 모터 구동
  //	                Motor_Right_Forward(right_pwm);
  //	                Motor_Left_Forward(left_pwm);
  //	            }
  //
  //	        }
  //	        else
  //	        {
  //	            rx_buffer[idx++] = rx_char;
  //	            if(idx >= sizeof(rx_buffer)) idx = sizeof(rx_buffer) - 1;
  //	        }
  //	    }
  //
  //
  //
  //	    HAL_GPIO_WritePin(GPIOB, GPIO_PIN_8, GPIO_PIN_SET); // PA5 핀에 HIGH 출력
  //
  //	    HAL_Delay(1000); // 1초 대기
  //
  //
  //
  //	    /* 1초 동안 부저 끄기 */
  //
  //	    HAL_GPIO_WritePin(GPIOB, GPIO_PIN_8, GPIO_PIN_RESET); // PA5 핀에 LOW 출력
  //
  //	    HAL_Delay(1000); // 1초 대기
  //      sprintf((char*)tx_buf, "Hello from STM32! Count: %d\n", count++);
  //      HAL_UART_Transmit(&huart1, tx_buf, strlen((char*)tx_buf), 100);
  //
  //      // PC 시리얼 모니터로 전송 확인
  //      printf("Sent to Pi: %s", tx_buf);
  //
  //
  //  // --- 1초마다 메시지 전송 ---
  //      TxData[0]++; // 매번 다른 값을 보내기 위해 1씩 증가
  //      TxData[1] = 0xDE;
  //      TxData[2] = 0xAD;
  //      TxData[3] = 0xBE;
  //      TxData[4] = 0xEF;
  //
  //      if (HAL_CAN_AddTxMessage(&hcan1, &TxHeader, TxData, &TxMailbox) == HAL_OK) {
  //        printf("STM32 Sent -> ID: 0x%lX, Data[0]: 0x%X\r\n", TxHeader.StdId, TxData[0]);
  //      } else {
  //        printf("STM32 Sent -> FAILED\r\n");
  //      }
  //
  //      // --- 메시지 수신 확인 (폴링) ---
  //      // 수신 버퍼(FIFO0)에 메시지가 있는지 확인
  //      if (HAL_CAN_GetRxFifoFillLevel(&hcan1, CAN_RX_FIFO0) > 0)
  //      {
  //        // 메시지가 있다면 읽어오기
  //        if (HAL_CAN_GetRxMessage(&hcan1, CAN_RX_FIFO0, &RxHeader, RxData) == HAL_OK)
  //        {
  //          printf("STM32 Received <- ID: 0x%lX, Data: ", RxHeader.StdId);
  //          for (int i = 0; i < RxHeader.DLC; i++) {
  //            printf("0x%X ", RxData[i]);
  //          }
  //          printf("\r\n\n");
  //        }


    // 10ms 마다 속도를 업데이트하여 가속/감속 테스트를 수행합니다.
//    if (HAL_GetTick() > motor_update_tick) {
//        motor_update_tick = HAL_GetTick() + 10; // 10ms 딜레이
//
//        if (is_accelerating) {
//            current_speed += 100; // 속도 증가
//            if (current_speed >= MAX_PWM_SPEED) {
//                current_speed = MAX_PWM_SPEED;
//                is_accelerating = 0; // 최대 속도 도달, 이제 감속 시작
//                HAL_Delay(500); // 0.5초 대기
//            }
//        } else {
//            current_speed -= 100; // 속도 감소
//            if (current_speed <= MIN_PWM_SPEED) {
//                current_speed = MIN_PWM_SPEED;
//                is_accelerating = 1; // 최소 속도 도달, 이제 가속 시작
//                HAL_Delay(500); // 0.5초 대기
//            }
//        }
//
//        // 모터를 현재 속도로 전진 구동
//        Motor_Forward(current_speed);
//
//        // OLED에 현재 속도 표시
//        char speed_buffer[20];
//        ssd1306_Fill(Black);
//        ssd1306_SetCursor(5, 5);
//        ssd1306_WriteString("Motor Test Mode", Font_7x10, White);
//        ssd1306_SetCursor(5, 20);
//        sprintf(speed_buffer, "Speed: %lu / 9999", (unsigned long)current_speed);
//        ssd1306_WriteString(speed_buffer, Font_7x10, White);
//        ssd1306_SetCursor(5, 35);
//        if (is_accelerating) {
//            ssd1306_WriteString("Status: ACCELERATING", Font_7x10, White);
//        } else {
//            ssd1306_WriteString("Status: DECELERATING", Font_7x10, White);
//        }
//        ssd1306_UpdateScreen();
//
//    }

  /* USER CODE END 3 */

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE3);

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL.PLLM = 8;
  RCC_OscInitStruct.PLL.PLLN = 72;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
  RCC_OscInitStruct.PLL.PLLQ = 2;
  RCC_OscInitStruct.PLL.PLLR = 2;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_2) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief I2C1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_I2C1_Init(void)
{

  /* USER CODE BEGIN I2C1_Init 0 */

  /* USER CODE END I2C1_Init 0 */

  /* USER CODE BEGIN I2C1_Init 1 */

  /* USER CODE END I2C1_Init 1 */
  hi2c1.Instance = I2C1;
  hi2c1.Init.ClockSpeed = 100000;
  hi2c1.Init.DutyCycle = I2C_DUTYCYCLE_2;
  hi2c1.Init.OwnAddress1 = 0;
  hi2c1.Init.AddressingMode = I2C_ADDRESSINGMODE_7BIT;
  hi2c1.Init.DualAddressMode = I2C_DUALADDRESS_DISABLE;
  hi2c1.Init.OwnAddress2 = 0;
  hi2c1.Init.GeneralCallMode = I2C_GENERALCALL_DISABLE;
  hi2c1.Init.NoStretchMode = I2C_NOSTRETCH_DISABLE;
  if (HAL_I2C_Init(&hi2c1) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN I2C1_Init 2 */

  /* USER CODE END I2C1_Init 2 */

}

/**
  * @brief TIM2 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM2_Init(void)
{

  /* USER CODE BEGIN TIM2_Init 0 */

  /* USER CODE END TIM2_Init 0 */

  TIM_MasterConfigTypeDef sMasterConfig = {0};
  TIM_OC_InitTypeDef sConfigOC = {0};

  /* USER CODE BEGIN TIM2_Init 1 */

  /* USER CODE END TIM2_Init 1 */
  htim2.Instance = TIM2;
  htim2.Init.Prescaler = 83;
  htim2.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim2.Init.Period = 9999;
  htim2.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim2.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_PWM_Init(&htim2) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim2, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sConfigOC.OCMode = TIM_OCMODE_PWM1;
  sConfigOC.Pulse = 0;
  sConfigOC.OCPolarity = TIM_OCPOLARITY_HIGH;
  sConfigOC.OCFastMode = TIM_OCFAST_DISABLE;
  if (HAL_TIM_PWM_ConfigChannel(&htim2, &sConfigOC, TIM_CHANNEL_1) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_TIM_PWM_ConfigChannel(&htim2, &sConfigOC, TIM_CHANNEL_2) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM2_Init 2 */

  /* USER CODE END TIM2_Init 2 */
  HAL_TIM_MspPostInit(&htim2);

}

/**
  * @brief USART1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART1_UART_Init(void)
{

  /* USER CODE BEGIN USART1_Init 0 */

  /* USER CODE END USART1_Init 0 */

  /* USER CODE BEGIN USART1_Init 1 */

  /* USER CODE END USART1_Init 1 */
  huart1.Instance = USART1;
  huart1.Init.BaudRate = 9600;
  huart1.Init.WordLength = UART_WORDLENGTH_8B;
  huart1.Init.StopBits = UART_STOPBITS_1;
  huart1.Init.Parity = UART_PARITY_NONE;
  huart1.Init.Mode = UART_MODE_TX_RX;
  huart1.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart1.Init.OverSampling = UART_OVERSAMPLING_16;
  if (HAL_UART_Init(&huart1) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART1_Init 2 */

  /* USER CODE END USART1_Init 2 */

}

/**
  * @brief USART2 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART2_UART_Init(void)
{

  /* USER CODE BEGIN USART2_Init 0 */

  /* USER CODE END USART2_Init 0 */

  /* USER CODE BEGIN USART2_Init 1 */

  /* USER CODE END USART2_Init 1 */
  huart2.Instance = USART2;
  huart2.Init.BaudRate = 115200;
  huart2.Init.WordLength = UART_WORDLENGTH_8B;
  huart2.Init.StopBits = UART_STOPBITS_1;
  huart2.Init.Parity = UART_PARITY_NONE;
  huart2.Init.Mode = UART_MODE_TX_RX;
  huart2.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart2.Init.OverSampling = UART_OVERSAMPLING_16;
  if (HAL_UART_Init(&huart2) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART2_Init 2 */

  /* USER CODE END USART2_Init 2 */

}

/**
  * @brief USART3 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART3_UART_Init(void)
{

  /* USER CODE BEGIN USART3_Init 0 */

  /* USER CODE END USART3_Init 0 */

  /* USER CODE BEGIN USART3_Init 1 */

  /* USER CODE END USART3_Init 1 */
  huart3.Instance = USART3;
  huart3.Init.BaudRate = 9600;
  huart3.Init.WordLength = UART_WORDLENGTH_8B;
  huart3.Init.StopBits = UART_STOPBITS_1;
  huart3.Init.Parity = UART_PARITY_NONE;
  huart3.Init.Mode = UART_MODE_TX_RX;
  huart3.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart3.Init.OverSampling = UART_OVERSAMPLING_16;
  if (HAL_UART_Init(&huart3) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART3_Init 2 */

  /* USER CODE END USART3_Init 2 */

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  /* USER CODE BEGIN MX_GPIO_Init_1 */

  /* USER CODE END MX_GPIO_Init_1 */

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOH_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOA, LD2_Pin|GPIO_PIN_6|GPIO_PIN_7, GPIO_PIN_RESET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOB, GPIO_PIN_4|GPIO_PIN_5|GPIO_PIN_8, GPIO_PIN_RESET);

  /*Configure GPIO pin : B1_Pin */
  GPIO_InitStruct.Pin = B1_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_IT_FALLING;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(B1_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pins : LD2_Pin PA6 PA7 */
  GPIO_InitStruct.Pin = LD2_Pin|GPIO_PIN_6|GPIO_PIN_7;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

  /*Configure GPIO pins : PB4 PB5 PB8 */
  GPIO_InitStruct.Pin = GPIO_PIN_4|GPIO_PIN_5|GPIO_PIN_8;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

  /* USER CODE BEGIN MX_GPIO_Init_2 */

  /* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */
PUTCHAR_PROTOTYPE
{
    // ST-Link V2-1 virtual com port (USART2)
    HAL_UART_Transmit(&huart2, (uint8_t *)&ch, 1, 0xFFFF);
    return ch;
}
/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
