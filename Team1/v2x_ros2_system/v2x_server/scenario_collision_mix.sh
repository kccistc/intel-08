
#!/usr/bin/env bash
set -euo pipefail
export V2X_KEY="${V2X_KEY:-mysecret}"

echo "[1/3] start collision repeater (30s)"
python3 /opt/v2x/server.py --repeat --hz 1 \
  --hmac-key "$V2X_KEY" \
  --type collision --severity high --distance-m 400 --suggest slow_down \
  --ttl-s 30 &
PID=$!

sleep 30
echo "[2/3] stop repeater"
kill $PID || true
sleep 2
pkill -f "python3 .*server.py" || true

echo "[3/3] emit two single events"
python3 /opt/v2x/emit_event.py --type collision --severity medium --distance-m 350 --clip-hint
sleep 3
python3 /opt/v2x/emit_event.py --type fire --severity high --distance-m 300 --clip-hint
echo "done."
