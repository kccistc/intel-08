#!/usr/bin/env bash
set -euo pipefail
echo "[SELFTEST] emit one event"
python3 /opt/v2x/emit_event.py --type collision --severity medium --distance-m 300 --clip-hint
echo "[SELFTEST] wait 5s"
sleep 5
echo "[SELFTEST] check db"
python3 /opt/v2x/v2x_db_cli.py --recent 3 | head -n 3
echo "[SELFTEST] check status"
curl -s http://127.0.0.1:8088/status || true
