import socket, json, time, hmac, hashlib

MCAST_GROUP = '239.20.20.20'
MCAST_PORT = 5520
HMAC_KEY = 'your-secret-hmac-key-here'

MSG_TYPE_EMERGENCY_BRAKE = 0

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
print("V2X Test Sender started (v2). Press Ctrl+C to exit.")

try:
    while True:
        print("Sending EMERGENCY_BRAKE alert in 10 seconds...")
        time.sleep(10)
        
        message = {
            "msg_type": MSG_TYPE_EMERGENCY_BRAKE,
            "vehicle_id": "Vehicle_C3PO",
            "coordinateX": 200.5,
            "coordinateY": -55.2,
            "confidence": 0.98,
            "timestamp": time.time()
        }
        
        raw_wo_sig = json.dumps(message, separators=(',',':')).encode('utf-8')
        sig = hmac.new(HMAC_KEY.encode("utf-8"), raw_wo_sig, hashlib.sha256).hexdigest()
        message_with_sig = dict(message)
        message_with_sig["sig"] = {"type": "HMAC-SHA256", "value": sig}
        final_payload = json.dumps(message_with_sig).encode('utf-8')
        sock.sendto(final_payload, (MCAST_GROUP, MCAST_PORT))
        print(f"Sent alert: {final_payload.decode()}")
except KeyboardInterrupt:
    print("\nSender stopped.")
finally:
    sock.close()