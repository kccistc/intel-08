// storage.c
#include "hardware.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <libgen.h>
#include <limits.h>
#include <time.h>
#include <errno.h>
#include "cJSON.h"

static pid_t g_rec_pid = -1;
static char  g_rec_target[PATH_MAX] = {0};

static int   g_rec_w = 1280, g_rec_h = 720, g_rec_fps = 30, g_rec_bitrate = 4000000;
static char  g_rec_device[64] = "/dev/video2";
static char  g_rec_dir[PATH_MAX] = "/data/records";

static void load_config_record(void) {
    const char *path = "/etc/aiblackbox/config.json";
    FILE *fp = fopen(path, "rb"); if (!fp) return;

    if (fseek(fp, 0, SEEK_END) != 0) { fclose(fp); return; }
    long sz = ftell(fp); if (sz < 0) { fclose(fp); return; }
    rewind(fp);

    char *buf = (char*)malloc((size_t)sz + 1);
    if(!buf){ fclose(fp); return; }
    size_t n = fread(buf, 1, (size_t)sz, fp);
    fclose(fp);
    if (n != (size_t)sz) { free(buf); return; }
    buf[sz] = '\0';

    cJSON *root = cJSON_Parse(buf);
    if(root){
        cJSON *rec = cJSON_GetObjectItemCaseSensitive(root, "record");
        if(cJSON_IsObject(rec)){
            cJSON *j;
            if((j=cJSON_GetObjectItemCaseSensitive(rec,"device")) && cJSON_IsString(j))
                strncpy(g_rec_device,j->valuestring,sizeof(g_rec_device)-1);
            if((j=cJSON_GetObjectItemCaseSensitive(rec,"width"))  && cJSON_IsNumber(j))
                g_rec_w = j->valueint;
            if((j=cJSON_GetObjectItemCaseSensitive(rec,"height")) && cJSON_IsNumber(j))
                g_rec_h = j->valueint;
            if((j=cJSON_GetObjectItemCaseSensitive(rec,"fps"))    && cJSON_IsNumber(j))
                g_rec_fps = j->valueint;
            if((j=cJSON_GetObjectItemCaseSensitive(rec,"bitrate"))&& cJSON_IsNumber(j))
                g_rec_bitrate = j->valueint;
            if((j=cJSON_GetObjectItemCaseSensitive(rec,"dir"))    && cJSON_IsString(j))
                strncpy(g_rec_dir,j->valuestring,sizeof(g_rec_dir)-1);
        }
        cJSON_Delete(root);
    }
    free(buf);
}

static int ensure_parent_dir(const char *path){
    char tmp[PATH_MAX];
    strncpy(tmp, path, sizeof(tmp)-1);
    tmp[sizeof(tmp)-1] = '\0';
    char *d = dirname(tmp);
    if (!d) return -1;
    if (access(d, W_OK) == 0) return 0;
    if (mkdir(d, 0775) == 0)  return 0;
    return (errno == EEXIST) ? 0 : -1;
}

int storage_start_recording(const char* filename)
{
    if (g_rec_pid > 0) return -1; // already recording

    load_config_record();

    // 최종 파일 경로 결정
    if (filename && filename[0]) {
        strncpy(g_rec_target, filename, sizeof(g_rec_target)-1);
    } else {
        // 예: /data/records/YYYY-MM-DD_HH-MM-SS.mp4
        time_t t=time(NULL); struct tm tm; localtime_r(&t,&tm);
        snprintf(g_rec_target, sizeof(g_rec_target),
                 "%s/%04d-%02d-%02d_%02d-%02d-%02d.mp4",
                 g_rec_dir, tm.tm_year+1900, tm.tm_mon+1, tm.tm_mday,
                 tm.tm_hour, tm.tm_min, tm.tm_sec);
    }
    if (ensure_parent_dir(g_rec_target) < 0) return -1;

    // GStreamer 파이프라인 (하드웨어 인코더 우선)
    char cmd[2048];
    int n = snprintf(cmd, sizeof(cmd),
        "exec gst-launch-1.0 -e "
        "v4l2src device=%s io-mode=2 ! "
        "video/x-raw,width=%d,height=%d,framerate=%d/1 ! "
        "videoconvert ! "
        "v4l2h264enc extra-controls=controls,video_bitrate_mode=1,video_bitrate=%d ! "
        "h264parse ! mp4mux faststart=true ! "
        "filesink location='%s' sync=false",
        g_rec_device, g_rec_w, g_rec_h, g_rec_fps, g_rec_bitrate, g_rec_target);
    if (n < 0 || (size_t)n >= sizeof(cmd)) return -1;

    pid_t pid = fork();
    if (pid < 0) return -1;
    if (pid == 0) {
        execl("/bin/sh","sh","-lc",cmd,(char*)NULL);
        _exit(127);
    }
    g_rec_pid = pid;
    return 0;
}

void storage_stop_recording(void)
{
    if (g_rec_pid > 0) {
        kill(g_rec_pid, SIGINT); // EOS 유도
        for (int i=0;i<50;i++){ // 5초 대기
            int st; pid_t r = waitpid(g_rec_pid, &st, WNOHANG);
            if (r == g_rec_pid) { g_rec_pid = -1; return; }
            usleep(100*1000);
        }
        kill(g_rec_pid, SIGTERM);
        waitpid(g_rec_pid, NULL, 0);
        g_rec_pid = -1;
    }
}

// 현재 구조에선 미지원(별도 프로세스 방식)
int storage_write_frame(const FrameBuffer* frame) { (void)frame; return -38; }
