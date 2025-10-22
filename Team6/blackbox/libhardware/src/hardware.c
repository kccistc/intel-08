// hardware.c (min)
#include "hardware.h"
#include <sys/stat.h>
#include <unistd.h>

int hardware_init(void) {
    // 공용 디렉터리 정도만 보장
    mkdir("/data/records", 0775);
    return 0;
}

void hardware_close(void) {
    // 녹화 중이면 정리
    storage_stop_recording();
}
