#!/usr/bin/env bash
set -euo pipefail

V2X_DIR="/opt/v2x"
ENV_FILE="$V2X_DIR/.env"
MCAST="${MCAST:-239.20.20.20}"
PORT="${PORT:-5520}"
LOG="${LOG:-/var/log/alerts.csv}"
V2X_KEY="${V2X_KEY:-changeme_please}"

echo "[1/4] package"
sudo apt-get update -y
sudo apt-get install -y python3 python3-pip alsa-utils sox libsox-fmt-all

echo "[2/4] directory/log"
sudo mkdir -p "$V2X_DIR"
sudo touch "$LOG" || true
sudo chmod 666 "$LOG" || true

echo "[3/4] Evorinment file(.env)"
sudo tee "$ENV_FILE" >/dev/null <<EOT
V2X_KEY=$V2X_KEY
MCAST=$MCAST
PORT=$PORT
LOG=$LOG
EOT
sudo chmod 640 "$ENV_FILE"

echo "[4/4] Client Unit"
sudo tee /etc/systemd/system/v2x-alert-client.service >/dev/null <<'UNIT'
[Unit]
Description=V2X Alert Client (UDP Multicast)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
EnvironmentFile=/opt/v2x/.env
ExecStart=/usr/bin/python3 -u /opt/v2x/client.py --mcast ${MCAST} --port ${PORT} --hmack-key ${V2X_KEY} --log ${LOG} --require-sig --beep
Restart=always
RestarSec=2
User=pi
SupplementaryGroups=audio
Environment=PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:sbin:/bin

[Install]
WantedBy=multi-user.target
UNIT

sudo systemctl daemon-reload
sudo systemctl enable v2x-alert-client.service
sudo systemctl restart v2x-alert-client.service

echo "** Client install done."
echo " - follow: journalctl -u v2x-alert-client.service -f"
