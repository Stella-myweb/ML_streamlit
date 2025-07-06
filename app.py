# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px

st.set_page_config(page_title="🏠 주민대피시설 통계 대시보드", layout="wide")

# 🔐 API 키 설정 (decoding 일반 인증키 사용)
service_key = "여기에_복호화된_인증키를_입력하세요"

@st.cache_data
def load_shelter_data(year="2019"):
    """주민대피시설 통계 API 데이터 로드"""
    url = "http://apis.data.go.kr/1741000/ShelterInfoOpenApi/getShelterInfo"
    params = {
        "serviceKey": service_key,
        "pageNo": 1,
        "numOfRows": 500,
        "type": "json",
        "bas_yy": year,
    }
    response = requests.get(url, params=params)
    if response.status_code != 200:
        st.error(f"API 요청 실패: {response.status_code}")
        return None

    data = response.json()
    items = data.get("response", {}).get("body", {}).get("items", [])
    df = pd.DataFrame(items)

    # 정제
    df["accept_rt"] = pd.to_numeric(df["accept_rt"].str.replace(",", ""), errors="coerce")  # 수용률
    df["target_popl"] = pd.to_numeric(df["target_popl"].str.replace(",", ""), errors="coerce")  # 대상인구
    df["shelt_abl_popl_smry"] = pd.to_numeric(df["shelt_abl_popl_smry"].str.replace(",", ""), errors="coerce")  # 수용 가능 인구
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")

    df.dropna(subset=["lat", "lon", "accept_rt"], inplace=True)
    return df

# 🎯 데이터 불러오기
st.title("🏠 주민대피시설 통계 대시보드")
with st.spinner("대피시설 통계 데이터를 불러오는 중입니다..."):
    shelter_df = load_shelter_data()

if shelter_df is None:
    st.stop()

# 📊 요약 지표
st.subheader("📌 전국 통계 요약")
col1, col2, col3 = st.columns(3)
col1.metric("📍 총 대피시설 수", len(shelter_df))
col2.metric("👥 대상 인구 총합", f"{int(shelter_df['target_popl'].sum()):,} 명")
col3.metric("📈 평균 수용률", f"{shelter_df['accept_rt'].mean():.2f}%")

# 📍 지역 필터
st.sidebar.header("🎛️ 필터")
sido_list = shelter_df["regi"].dropna().unique()
selected_region = st.sidebar.selectbox("시도 선택", ["전체"] + list(sorted(sido_list)))
if selected_region != "전체":
    shelter_df = shelter_df[shelter_df["regi"] == selected_region]

min_rt, max_rt = st.sidebar.slider("수용률 범위 (%)", 0.0, 500.0, (0.0, 500.0))
shelter_df = shelter_df[(shelter_df["accept_rt"] >= min_rt) & (shelter_df["accept_rt"] <= max_rt)]

# 🗺️ 대피시설 지도 시각화
st.subheader("📍 대피시설 분포 지도")
map_color = shelter_df["accept_rt"].apply(lambda x: "red" if x < 100 else "yellow" if x < 300 else "green")

fig = px.scatter_mapbox(
    shelter_df,
    lat="lat",
    lon="lon",
    color=map_color,
    hover_data=["regi", "target_popl", "accept_rt", "shelt_abl_popl_smry"],
    zoom=5,
    size_max=15,
)
fig.update_layout(mapbox_style="carto-positron", height=500, margin={"r":0, "t":0, "l":0, "b":0})
st.plotly_chart(fig, use_container_width=True)

# 🔥 수용률 낮은 지역 Top 10
st.subheader("🔥 인구 대비 수용률 낮은 지역 Top 10")
top10 = shelter_df.sort_values(by="accept_rt").head(10)
st.dataframe(top10[["regi", "target_popl", "shelt_abl_popl_smry", "accept_rt"]])

# 📊 수용률 히트맵
st.subheader("🌡️ 시도별 수용률 히트맵")
pivot = shelter_df.groupby("regi")["accept_rt"].mean().reset_index()
fig2 = px.density_heatmap(pivot, x="regi", y="accept_rt", color_continuous_scale="RdYlGn", height=300)
st.plotly_chart(fig2, use_container_width=True)

# ℹ️ 비상 정보
st.subheader("ℹ️ 실용 정보")
st.markdown("""
- **내 지역 대피시설 찾기**: 지도에서 위치 확인 가능
- **가장 가까운 대피소 거리 계산**: 향후 업데이트 예정
- **비상연락처**: [행정안전부 재난안전포털](https://www.safekorea.go.kr)
- **대피요령 안내**: 지진, 화재, 풍수해 등 상황별 대피 행동요령은 [여기](https://www.safekorea.go.kr/idsiSFK/neo/main/main.html) 참고
""")
