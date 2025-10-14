
#!/usr/bin/env bash
set -euo pipefail

# ===== Settings (필요하면 여기 수정) =====
V2X_DIR="/opt/v2x"
REC_DIR="/var/rec"
ARCH_DIR="/var/archive"
DB_PATH="$V2X_DIR/v2x_index.sqlite3"
ENV_FILE="$V2X_DIR/.env"
MCAST="${MCAST:-239.20.20.20}"
PORT="${PORT:-5520}"
LOG="${LOG:-/var/log/alerts.csv}"
V2X_KEY="${V2X_KEY:-changeme_please}"
RTSP_URL="${RTSP_URL:-/dev/video0}"      # RTSP 쓰면 rtsp://... 로 바꿔줘
SEG_DUR="${SEG_DUR:-10}"                 # 녹화 세그먼트(초)
# ========================================

echo "[1/7] 패키지 설치"
sudo apt-get update -y
sudo apt-get install -y python3 python3-pip ffmpeg sqlite3 jq curl

echo "[2/7] 디렉터리/권한"
sudo mkdir -p "$V2X_DIR" "$REC_DIR" "$ARCH_DIR"
sudo chmod 755 "$V2X_DIR" "$REC_DIR" "$ARCH_DIR"
sudo touch "$LOG" || true
sudo chmod 666 "$LOG" || true

echo "[3/7] 환경파일(.env)"
sudo tee "$ENV_FILE" >/dev/null <<EOT
V2X_KEY=$V2X_KEY
MCAST=$MCAST
PORT=$PORT
LOG=$LOG
RTSP_URL=$RTSP_URL
SEG_DIR=$REC_DIR
ARCHIVE_DIR=$ARCH_DIR
DB_PATH=$DB_PATH
EOT
sudo chmod 640 "$ENV_FILE"

echo "[4/7] systemd 유닛 배포(서버측)"
# 4-1) 상태 서버
sudo tee /etc/systemd/system/v2x-status.service >/dev/null <<'UNIT'
[Unit]
Description=V2X Status HTTP (port 8088)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
ExecStart=/usr/bin/python3 -u /opt/v2x/status_server.py
Restart=always
RestartSec=2

[Install]
WantedBy=multi-user.target
UNIT

# 4-2) 이벤트 브로드캐스터(사고 알림)
sudo tee /etc/systemd/system/v2x-alert-server.service >/dev/null <<'UNIT'
[Unit]
Description=V2X Accident Alert Broadcaster
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
EnvironmentFile=/opt/v2x/.env
ExecStart=/usr/bin/python3 -u /opt/v2x/server.py --mcast ${MCAST} --port ${PORT} --hmac-key ${V2X_KEY}
Restart=always
RestartSec=2

[Install]
WantedBy=multi-user.target
UNIT

# 4-3) 하트비트 송신기
sudo tee /etc/systemd/system/v2x-heartbeat.service >/dev/null <<'UNIT'
[Unit]
Description=V2X Heartbeat Broadcaster
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
EnvironmentFile=/opt/v2x/.env
ExecStart=/usr/bin/python3 -u /opt/v2x/heartbeat_tx.py --mcast ${MCAST} --port ${PORT}
Restart=always
RestartSec=2

[Install]
WantedBy=multi-user.target
UNIT

# 4-4) 감지기→이벤트 드롭 연계
sudo tee /etc/systemd/system/v2x-watcher.service >/dev/null <<'UNIT'
[Unit]
Description=V2X Event JSON watcher -> DB indexer + broadcaster trigger
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
EnvironmentFile=/opt/v2x/.env
ExecStart=/usr/bin/python3 -u /opt/v2x/db_index.py --db ${DB_PATH}
Restart=always
RestartSec=2

[Install]
WantedBy=multi-user.target
UNIT

# 4-5) 순환 녹화기(세그먼트)
sudo tee /etc/systemd/system/v2x-recorder.service >/dev/null <<UNIT
[Unit]
Description=V2X Circular Recorder (FFmpeg ${SEG_DUR}s segments)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
EnvironmentFile=/opt/v2x/.env
ExecStart=/usr/bin/ffmpeg -hide_banner -loglevel error -y \\
 -i \${RTSP_URL} -an -c:v libx264 -preset veryfast -crf 23 \\
 -f segment -segment_time ${SEG_DUR} -reset_timestamps 1 \\
 \${SEG_DIR}/seg_%s.mp4
Restart=always
RestartSec=2

[Install]
WantedBy=multi-user.target
UNIT

echo "[5/7] 로그/아카이브 청소 크론"
sudo tee /usr/local/sbin/v2x_clean.sh >/dev/null <<'CLEAN'
#!/usr/bin/env bash
set -e
find /var/rec -type f -name 'seg_*.mp4' -mmin +120 -delete 2>/dev/null
LIMIT=$((3*1024*1024*1024))
USED=$(du -sb /var/archive 2>/dev/null | awk '{print $1}')
if [ -n "$USED" ] && [ "$USED" -gt "$LIMIT" ]; then
  ls -1t /var/archive/accident_*.mp4 2>/dev/null | tail -n +50 | xargs -r rm -f
fi
CLEAN
sudo chmod +x /usr/local/sbin/v2x_clean.sh
( sudo crontab -l 2>/dev/null; echo "15 * * * * /usr/local/sbin/v2x_clean.sh" ) | sudo crontab -

echo "[6/7] systemd 리로드/활성화"
sudo systemctl daemon-reload
sudo systemctl enable v2x-status.service v2x-alert-server.service v2x-heartbeat.service v2x-watcher.service v2x-recorder.service

echo "[7/7] 서비스 시작"
sudo systemctl restart v2x-status.service v2x-alert-server.service v2x-heartbeat.service v2x-watcher.service v2x-recorder.service

echo "✅ Server install done."
echo " - status:   curl -s http://127.0.0.1:8088/status | jq ."
echo " - events:   ls -1 /opt/v2x/events | tail"
echo " - segments: ls -lh /var/rec | tail"
