#!/usr/bin/env python
# -*- coding: utf-8 -*-

import streamlit as st
import requests
import time
import socket # 💡 IP 주소를 자동으로 찾기 위해 socket 라이브러리를 추가합니다.

# --- 페이지 설정 ---
st.set_page_config(
    page_title="CARLA CCTV Monitor",
    layout="wide",
)

# --- 커스텀 CSS ---
st.markdown("""
    <style>
    /* ... (CSS 스타일 코드는 길어서 생략) ... */
    .stApp {
        background-color: #0e1117;
    }
    .st-emotion-cache-1y4p8pa {
        max-width: 95%;
    }
    </style>
""", unsafe_allow_html=True)

# 💡💡💡 [수정된 부분 START] 💡💡💡
def get_local_ip():
    """서버의 내부 IP 주소를 자동으로 찾아 반환하는 함수"""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # 이 IP는 실제 연결되지 않아도 됩니다.
        # 내 컴퓨터가 이 주소로 나가기 위해 사용하는 IP를 찾으려는 목적입니다.
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1' # 실패 시 localhost를 반환합니다.
    finally:
        s.close()
    return IP

# 서버의 실제 네트워크 IP 주소를 가져옵니다.
SERVER_IP = get_local_ip()
# Flask 서버의 URL을 'localhost'가 아닌 위에서 찾은 실제 IP로 설정합니다.
FLASK_SERVER = f"http://{SERVER_IP}:5000"
# 💡💡💡 [수정된 부분 END] 💡💡💡


# --- 세션 상태 초기화 ---
if 'toast_sent_times' not in st.session_state:
    st.session_state.toast_sent_times = {'cctv1': 0, 'cctv2': 0}

# --- 함수 정의 ---
def check_server_status():
    """Flask 서버 상태 확인"""
    try:
        response = requests.get(f"{FLASK_SERVER}/health", timeout=1)
        return response.status_code == 200
    except requests.ConnectionError:
        return False

# --- 메인 UI ---
st.title("CARLA CCTV 사고 감지 시스템")
st.markdown("---")

with st.sidebar:
    st.header("시스템 제어")
    
    # 서버 상태 체크 및 표시
    server_online = check_server_status()
    if server_online:
        st.success(f"✅ 서버 연결됨 ({SERVER_IP})") # 💡 여기에 IP 주소를 표시해줍니다.
    else:
        st.error("⚠️ 서버 연결 끊김")
        st.warning("Flask 서버를 먼저 실행해주세요.")
        st.stop() # 서버가 없으면 앱 실행 중지

    st.divider()
    st.subheader("실시간 사고 감지 현황")
    # 사고 상태를 표시할 공간
    status_placeholder = st.empty()

# 메인 화면 레이아웃
col1, col2 = st.columns(2)
with col1:
    st.subheader("CCTV 1")
    # ✅ 수정된 FLASK_SERVER 변수를 사용합니다.
    cctv1_placeholder = st.image(f"{FLASK_SERVER}/video_feed/cctv1")
with col2:
    st.subheader("CCTV 2")
    # ✅ 수정된 FLASK_SERVER 변수를 사용합니다.
    cctv2_placeholder = st.image(f"{FLASK_SERVER}/video_feed/cctv2")

status_url = f"{FLASK_SERVER}/api/status"
toast_cooldown = 10 # 텍스트 경고는 10초 간격으로

try:
    # 서버에서 사고 상태 정보 가져오기
    response = requests.get(status_url, timeout=1)
    if response.status_code == 200:
        status = response.json()
        
        # 사이드바에 현재 상태 텍스트로 표시
        cctv1_status = "🔴 감지됨" if status.get('cctv1') else "🟢 정상"
        cctv2_status = "🔴 감지됨" if status.get('cctv2') else "🟢 정상"
        status_placeholder.markdown(f"**CCTV 1:** {cctv1_status}\n\n**CCTV 2:** {cctv2_status}")
        
        # 각 CCTV 상태 확인 후 텍스트 경고(Toast) 표시
        if status.get('cctv1') and (time.time() - st.session_state.toast_sent_times['cctv1'] > toast_cooldown):
            st.toast("🚨 CCTV 1에서 사고 의심!", icon="🚗")
            st.session_state.toast_sent_times['cctv1'] = time.time()
            
        if status.get('cctv2') and (time.time() - st.session_state.toast_sent_times['cctv2'] > toast_cooldown):
            st.toast("🚨 CCTV 2에서 사고 의심!", icon="🚗")
            st.session_state.toast_sent_times['cctv2'] = time.time()

except requests.RequestException:
    # 서버가 응답 없을 때 사이드바 상태 업데이트
    status_placeholder.warning("상태 정보를 가져올 수 없습니다.")

# Streamlit이 주기적으로 재실행하도록 약간의 딜레이
time.sleep(1)
st.rerun() # 스크립트를 처음부터 다시 실행하여 루프 효과 생성

