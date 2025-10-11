/* Packet Processing */
#define PACKET_START (0x55)
#define PACKET_END (0xaa)

#define PACKET_ID_INTERRUPT (0x00)
#define PACKET_ID_MAX (199)
#define PACKET_ID_ERROR (0xff)

static byte packet[5] = { PACKET_START, 0xff, 0xff, 0xff, PACKET_END };

enum pin_functions
{
    BEACON_RED = 2,
    BEACON_ORANGE,
    BEACON_GREEN,
    BEACON_BUZZER,

    ACTUATOR_1, // 6
    ACTUATOR_2,
    CONVEYOR_EN,
    CONVEYOR_PWM,

    START_BUTTON, // 10
    STOP_BUTTON,
    PHOTOELECTRIC_SENSOR_1,
    PHOTOELECTRIC_SENSOR_2
};

#define SEND_PACKET(pin, value)               \
    do                                        \
    {                                         \
        packet[2] = (pin);                    \
        packet[3] = (value);                  \
        Serial.write(packet, sizeof(packet)); \
    } while (0)

/* Utility Functions */
#define INIT_OUTPUT_PAD(pad, val) \
    pinMode((pad), OUTPUT);       \
    digitalWrite((pad), (val))

#define INIT_PWM_PAD(pad, val) \
    pinMode((pad), OUTPUT);    \
    analogWrite((pad), (val))

#define CHECK(expr)              \
    if ((expr))                  \
    {                            \
        SEND_PACKET(0xff, 0xff); \
        return;                  \
    }

/* System Initialization */
void setup()
{
    /* Initialize Outputs */
    INIT_OUTPUT_PAD(BEACON_RED, HIGH);
    INIT_OUTPUT_PAD(BEACON_ORANGE, HIGH);
    INIT_OUTPUT_PAD(BEACON_GREEN, HIGH);

    INIT_OUTPUT_PAD(BEACON_BUZZER, HIGH);

    INIT_OUTPUT_PAD(ACTUATOR_1, HIGH);
    INIT_OUTPUT_PAD(ACTUATOR_2, HIGH);

    INIT_OUTPUT_PAD(CONVEYOR_EN, HIGH);
    INIT_PWM_PAD(CONVEYOR_PWM, 0);

    /* Initialize Inputs */
    pinMode(START_BUTTON, INPUT_PULLUP);
    pinMode(STOP_BUTTON, INPUT_PULLUP);
    pinMode(PHOTOELECTRIC_SENSOR_1, INPUT_PULLUP);
    pinMode(PHOTOELECTRIC_SENSOR_2, INPUT_PULLUP);

    /* Prepare UART port - 115200 baud rate */
    Serial.begin(115200);
    while (!Serial)
        ;
}

/* Set to 0xff to report initial state */
static byte inputStatus[4] = { 0xff, 0xff, 0xff, 0xff };

static inline compareAndReport(byte pin)
{
    static byte idx;
    static byte value;

    idx = pin - START_BUTTON;
    value = digitalRead(pin);
    if (inputStatus[idx] == value)
        return;

    inputStatus[idx] = value;
    SEND_PACKET(pin, value);
}

static inline void polling(void)
{
    packet[1] = PACKET_ID_INTERRUPT;

    compareAndReport(START_BUTTON);
    compareAndReport(STOP_BUTTON);
    compareAndReport(PHOTOELECTRIC_SENSOR_1);
    compareAndReport(PHOTOELECTRIC_SENSOR_2);
}

/* Actual Main loop */
void loop()
{
    polling();

    /* Packet Processing */
    if (Serial.available() < 5)
        return;

    static byte packetId;
    static byte pin;
    static byte value;

    packet[1] = PACKET_ID_ERROR;

    CHECK(Serial.read() != PACKET_START);

    packetId = Serial.read();
    CHECK(packetId == 0 || packetId > PACKET_ID_MAX);

    pin = Serial.read();
    CHECK(pin < BEACON_RED || pin > CONVEYOR_PWM);

    value = Serial.read();

    CHECK(Serial.read() != PACKET_END);

    /* Run command */
    packet[1] = packetId;

    if (pin == CONVEYOR_PWM)
        analogWrite(pin, value);
    else
        digitalWrite(pin, !!value);

    SEND_PACKET(pin, value);
}
