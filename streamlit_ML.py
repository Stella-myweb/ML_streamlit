# -*- coding: utf-8 -*-
"""
Created on Sat Jul  5 14:59:40 2025

@author: Administrator
"""
# 1. 라이브러리 불러오기
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import matplotlib.pyplot as plt
import requests
import urllib.parse


# 인증키 (Decoding 키 붙여넣기!)
service_key_raw = "xHdZnEfS8XNovgc69B/bQIwtLas/+h2gvgmHWbC9auMwvqT1KCMZ8VwrYBiJa+jskRMBN7pI8AMoAQ7zRY1vfg=="
# service_key = urllib.parse.quote(service_key_raw, safe='')

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

response = requests.get(url, params=params)

# 응답 확인
print("🔎 상태 코드:", response.status_code)
print("🔎 Content-Type:", response.headers.get("Content-Type"))
print("🔎 응답 내용:\n", response.text[:300])

# JSON 파싱
try:
    data = response.json()
    print("✅ JSON 파싱 성공!", data.keys())
except Exception as e:
    print("❌ JSON 파싱 실패:", e)
    
items = data.get("items", [])

df = pd.DataFrame(items)

#%%
import ast
import plotly.express as px

df_parsed = df['item']
df_norm = pd.DataFrame(df_parsed.tolist())

print(df_norm.head())

print(df_norm.info())
#%%
# 3. 데이터 전처리
## (1) 시도명 추출
df_norm['시도명'] = df_norm['sido_sgg_nm'].str.extract(r'^(\S+)')

print(df_norm.info())
#%%
## (2) 위도/경도 변환
df_norm['위도'] = pd.to_numeric(df_norm['la_crd'], errors='coerce')
df_norm['경도'] = pd.to_numeric(df_norm['lo_crd'], errors='coerce')

print(df_norm.info())
#%%
## (3) 수치형 컬럼 변환
numeric_cols = ['occrrnc_cnt', 'caslt_cnt', 'dth_dnv_cnt', 'se_dnv_cnt', 'sl_dnv_cnt', 'wnd_dnv_cnt']
df_norm[numeric_cols] = df_norm[numeric_cols].apply(pd.to_numeric, errors='coerce')
print(df_norm.info())
#%%
## (4) 결측치 제거
df_norm.dropna(subset=['위도', '경도', 'occrrnc_cnt'], inplace=True)
#%%
# 4. 모델 학습용 데이터 구성
## (1) 원-핫 인코딩
df_encoded = pd.get_dummies(df_norm, columns=['시도명'], drop_first=True)
#%%
## (2) 입력 피처 / 타겟 분리
features = ['caslt_cnt', 'dth_dnv_cnt', 'se_dnv_cnt', 'sl_dnv_cnt', 'wnd_dnv_cnt', '위도', '경도']
features += [col for col in df_encoded.columns if col.startswith('시도명_')]
X = df_encoded[features]
y = df_encoded['occrrnc_cnt']
#%%
# 5. 학습/테스트 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
#%%
# 6. 모델 학습
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
#%%
# 7. 예측 및 평가
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"✅ RMSE: {rmse:.2f}")
print(f"✅ R² Score: {r2:.3f}")
#%%
# 8. 중요 변수 시각화
importances = pd.Series(model.feature_importances_, index=X.columns)
importances.nlargest(10).plot(kind='barh')
plt.title("사고 발생에 영향을 미친 중요 변수 Top 10")
plt.xlabel("중요도")
plt.tight_layout()
plt.show()
#%%
# 9. 지도 시각화
X_test_result = X_test.copy()
X_test_result['예측사고건수'] = y_pred
X_test_result['위도'] = df_encoded.loc[X_test_result.index, '위도']
X_test_result['경도'] = df_encoded.loc[X_test_result.index, '경도']

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

fig.update_layout(mapbox_style="carto-positron", margin={"r":0,"t":40,"l":0,"b":0})
fig.write_html("예측사고지도.html")
fig.show()

#%%
import streamlit as st
fig.update_layout(mapbox_style="carto-positron", margin={"r":0,"t":40,"l":0,"b":0})

# 🔝 위험지역 TOP 10
top_n = X_test_result.nlargest(10, '예측사고건수')

# 📊 화면 구성
st.title("❄️ 결빙 교통사고 예측 대시보드")
st.markdown("2023년 결빙 교통사고 다발지역을 머신러닝으로 예측하여 시각화한 결과입니다.")

st.subheader("📌 예측 사고지도")
st.plotly_chart(fig, use_container_width=True)

st.subheader("🔥 위험지역 Top 10")
st.dataframe(top_n[['위도', '경도', '예측사고건수']])

st.subheader("📍 인사이트 요약")
st.markdown("""
- **예측 사고지도**를 통해 향후 사고 다발 가능성이 높은 지역을 선제적으로 파악할 수 있습니다.
- **Top 10 위험지역**을 기반으로 제설작업, 경고판 설치, 감시카메라 배치 등을 우선 적용할 수 있습니다.
- 사고건수는 사망자·부상자수, 위경도 등의 복합 요인에 영향을 받으므로 지역 맞춤형 정책이 필요합니다.
- **실시간 지도 시각화**는 정책입안자 및 현장 관리자에게 직관적인 의사결정 근거를 제공합니다.
""")
