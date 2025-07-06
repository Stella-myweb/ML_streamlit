# -*- coding: utf-8 -*-
"""
결빙 교통사고 예측 Streamlit 앱
"""
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import matplotlib.pyplot as plt
import requests
import urllib.parse

# 페이지 설정
st.set_page_config(
    page_title="결빙 교통사고 예측 대시보드",
    page_icon="❄️",
    layout="wide"
)

# 캐시 데코레이터로 데이터 로딩 최적화
@st.cache_data
def load_and_process_data():
    """데이터 로딩 및 전처리 함수"""
    
    # API 인증키
    service_key_raw = "jUxxEMTFyxsIT2rt2P8JBO9y0EmFT9mx1zNPb31XLX27rFNH12NQ%2B6%2BZLqqvW6k%2FffQ5ZOOYzzcSo0Fq4u3Lfg%3D%3D"
    #service_key_raw = "xHdZnEfS8XNovgc69B/bQIwtLas/+h2gvgmHWbC9auMwvqT1KCMZ8VwrYBiJa+jskRMBN7pI8AMoAQ7zRY1vfg=="    
    # API 요청
    url = "http://apis.data.go.kr/B552061/frequentzoneFreezing/getRestFrequentzoneFreezing"
    params = {
        "serviceKey": service_key_raw,
        "searchYearCd": "2023",
        "siDo": "",
        "guGun": "",
        "type": "json",
        "numOfRows": "100",
        "pageNo": "1"
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code != 200:
            st.error(f"API 요청 실패: 상태 코드 {response.status_code}")
            return None, None, None, None, None
            
        data = response.json()
        items = data.get("items", [])
        
        if not items:
            st.error("API에서 데이터를 받아오지 못했습니다.")
            return None, None, None, None, None
            
        df = pd.DataFrame(items)
        
        # 데이터 전처리
        df_parsed = df['item']
        df_norm = pd.DataFrame(df_parsed.tolist())
        
        # 시도명 추출
        df_norm['시도명'] = df_norm['sido_sgg_nm'].str.extract(r'^(\S+)')
        
        # 위도/경도 변환
        df_norm['위도'] = pd.to_numeric(df_norm['la_crd'], errors='coerce')
        df_norm['경도'] = pd.to_numeric(df_norm['lo_crd'], errors='coerce')
        
        # 수치형 컬럼 변환
        numeric_cols = ['occrrnc_cnt', 'caslt_cnt', 'dth_dnv_cnt', 'se_dnv_cnt', 'sl_dnv_cnt', 'wnd_dnv_cnt']
        df_norm[numeric_cols] = df_norm[numeric_cols].apply(pd.to_numeric, errors='coerce')
        
        # 결측치 제거
        df_norm.dropna(subset=['위도', '경도', 'occrrnc_cnt'], inplace=True)
        
        if len(df_norm) == 0:
            st.error("전처리 후 사용할 수 있는 데이터가 없습니다.")
            return None, None, None, None, None
        
        return df_norm, True, None, None, None
        
    except Exception as e:
        st.error(f"데이터 로딩 중 오류 발생: {str(e)}")
        return None, None, None, None, None

@st.cache_data
def train_model(df_norm):
    """모델 학습 함수"""
    
    # 원-핫 인코딩
    df_encoded = pd.get_dummies(df_norm, columns=['시도명'], drop_first=True)
    
    # 피처와 타겟 분리
    features = ['caslt_cnt', 'dth_dnv_cnt', 'se_dnv_cnt', 'sl_dnv_cnt', 'wnd_dnv_cnt', '위도', '경도']
    features += [col for col in df_encoded.columns if col.startswith('시도명_')]
    
    X = df_encoded[features]
    y = df_encoded['occrrnc_cnt']
    
    # 학습/테스트 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 모델 학습
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 예측 및 평가
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # 결과 데이터 준비
    X_test_result = X_test.copy()
    X_test_result['예측사고건수'] = y_pred
    X_test_result['위도'] = df_encoded.loc[X_test_result.index, '위도']
    X_test_result['경도'] = df_encoded.loc[X_test_result.index, '경도']
    
    return model, X_test_result, rmse, r2, X.columns

def main():
    """메인 함수"""
    
    # 제목
    st.title("❄️ 결빙 교통사고 예측 대시보드")
    st.markdown("2023년 결빙 교통사고 다발지역을 머신러닝으로 예측하여 시각화한 결과입니다.")
    
    # 데이터 로딩
    with st.spinner("데이터를 불러오는 중..."):
        df_norm, success, _, _, _ = load_and_process_data()
    
    if df_norm is None:
        st.stop()
    
    # 모델 학습
    with st.spinner("머신러닝 모델을 학습하는 중..."):
        model, X_test_result, rmse, r2, feature_names = train_model(df_norm)
    
    # 성능 지표 표시
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("데이터 건수", len(df_norm))
    with col2:
        st.metric("RMSE", f"{rmse:.2f}")
    with col3:
        st.metric("R² Score", f"{r2:.3f}")
    
    # 지도 시각화
    st.subheader("📌 예측 사고지도")
    
    if len(X_test_result) > 0:
        fig = px.scatter_mapbox(
            X_test_result,
            lat="위도",
            lon="경도",
            size="예측사고건수",
            color="예측사고건수",
            hover_data=["예측사고건수"],
            zoom=6,
            size_max=30,
            title="예측된 사고 위험 지역 (버블 크기 = 예측 사고 건수)"
        )
        
        fig.update_layout(
            mapbox_style="carto-positron",
            margin={"r":0,"t":40,"l":0,"b":0},
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 위험지역 Top 10
        st.subheader("🔥 위험지역 Top 10")
        top_n = X_test_result.nlargest(10, '예측사고건수')
        st.dataframe(top_n[['위도', '경도', '예측사고건수']], use_container_width=True)
        
        # 피처 중요도
        st.subheader("📊 중요 변수 분석")
        importances = pd.Series(model.feature_importances_, index=feature_names)
        top_features = importances.nlargest(10)
        
        fig_importance = px.bar(
            x=top_features.values,
            y=top_features.index,
            orientation='h',
            title="사고 발생에 영향을 미친 중요 변수 Top 10"
        )
        fig_importance.update_layout(
            xaxis_title="중요도",
            yaxis_title="변수",
            height=400
        )
        st.plotly_chart(fig_importance, use_container_width=True)
    
    else:
        st.warning("테스트 데이터가 충분하지 않습니다.")
    
    # 인사이트
    st.subheader("📍 인사이트 요약")
    st.markdown("""
    - **예측 사고지도**를 통해 향후 사고 다발 가능성이 높은 지역을 선제적으로 파악할 수 있습니다.
    - **Top 10 위험지역**을 기반으로 제설작업, 경고판 설치, 감시카메라 배치 등을 우선 적용할 수 있습니다.
    - 사고건수는 사망자·부상자수, 위경도 등의 복합 요인에 영향을 받으므로 지역 맞춤형 정책이 필요합니다.
    - **실시간 지도 시각화**는 정책입안자 및 현장 관리자에게 직관적인 의사결정 근거를 제공합니다.
    """)

if __name__ == "__main__":
    main()
