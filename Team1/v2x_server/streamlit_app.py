#!/usr/bin/env python
# -*- coding: utf-8 -*-

import streamlit as st
import requests
import time

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

# --- 서버 URL ---
FLASK_SERVER = "http://localhost:5000"

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

# --- 💡 수정: 사이드바 UI 복구 ---
with st.sidebar:
    st.header("시스템 제어")
    
    # 서버 상태 체크 및 표시
    server_online = check_server_status()
    if server_online:
        st.success("✅ 서버 연결됨")
    else:
        st.error("⚠️ 서버 연결 끊김")
        st.warning("Flask 서버를 먼저 실행해주세요.")
        st.stop() # 서버가 없으면 앱 실행 중지

    st.divider()
    st.subheader("실시간 사고 감지 현황")
    # 사고 상태를 표시할 공간
    status_placeholder = st.empty()
# --------------------------------

# 메인 화면 레이아웃
col1, col2 = st.columns(2)
with col1:
    st.subheader("CCTV 1")
    cctv1_placeholder = st.image(f"{FLASK_SERVER}/video_feed/cctv1", width='stretch')
with col2:
    st.subheader("CCTV 2")
    cctv2_placeholder = st.image(f"{FLASK_SERVER}/video_feed/cctv2", width='stretch')

# --- 💡 수정: while True 루프를 제거하고 Streamlit의 실행 흐름 활용 ---
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
# -------------------------------------------------------------