#!/usr/bin/env python3

import socket, json, time, os, struct

MCAST = os.environ.get("MCAST","239.20.20.20")

PORT  = int(os.environ.get("PORT","5520"))

SRC   = os.environ.get("SRC_ID","rsu_heartbeat")

IFACE = os.environ.get("IFACE","")  # 송신에 사용할 로컬 IPv4 (예: 192.168.0.10)

sock=socket.socket(socket.AF_INET,socket.SOCK_DGRAM,socket.IPPROTO_UDP)

sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 1)

# 지정된 경우 멀티캐스트 송신 인터페이스 설정
if IFACE:
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_IF, socket.inet_aton(IFACE))

seq=0

while True:

    seq+=1

    msg={"hdr":{"ver":1,"src":SRC,"seq":seq,"ts":time.time()},

         "accident":{"type":"heartbeat","severity":"low","distance_m":0,"road":"","lat":0,"lon":0},

         "advice":{"message":"","suggest":"caution"},"ttl_s":5}

    sock.sendto(json.dumps(msg, separators=(',',':')).encode(), (MCAST, PORT))

    print(f"[HEARTBEAT] seq={seq} sent at {time.strftime('%X')}")

    time.sleep(10)

