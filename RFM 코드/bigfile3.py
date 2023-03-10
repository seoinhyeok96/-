# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 14:42:45 2023

@author: NADA
"""

import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt

df = pd.read_parquet("df.sample.parquet.gzip")

df = df.astype({'product_id': np.uint32, 'category_id': np.uint64, 'user_id': np.uint32}) # 데이터 타입 변경
df[['event_type', 'category_code', 'brand']] = df[['event_type', 'category_code', 'brand']].astype('category') # 데이터 타입 변경
df['event_time'] = pd.to_datetime(df['event_time']) # 데이터 타입 변경
df['price'] = df['price'].astype('float32')

df = df.drop('category_id', axis=1) # category_id칼럼 삭제

'''RFM을 위한 전처리'''
last_timestamp = df['event_time'].max().tz_convert('UTC') + dt.timedelta(days=1) # 가장 최근의 날짜/시간 값을 가져와 UTC 시간대로 변환
df['event_time'] = pd.to_datetime(df['event_time']).dt.tz_convert('UTC') #  'event_time' 열에 저장된 시간 정보가 UTC 시간대를 기준으로 표시
rfm = df[df['event_type'] == 'purchase'].groupby('user_id').agg(
    Recency=('event_time', lambda x: (last_timestamp - x.max()).days),
    Frequency=('event_type', 'count'),
    MonetaryValue=('price', 'sum')
)

'''라벨 정의'''
r_labels = list(range(5, 0, -1))
f_labels = list(range(1,6))
m_labels = list(range(1,6))
cut_size = 5
'''cut 정의'''
r_cut = pd.qcut(rfm['Recency'], cut_size, labels = r_labels)
f_cut = pd.qcut(rfm['Frequency'].rank(method='first'), cut_size, labels=f_labels, duplicates='drop')
m_cut = pd.qcut(rfm['MonetaryValue'], cut_size, labels = m_labels)

'''파생변수 정의'''
rfm = rfm.assign(R = r_cut, F = f_cut, M = m_cut)
rfm["RFM_segment"] = rfm['R'].astype(str) + rfm['F'].astype(str) + rfm['M'].astype(str)
rfm["RFM_score"] = rfm[['R','F','M']].sum(axis=1)

'''시각화'''
# RFM segment 값에 따라 정렬합니다.
plt.figure(figsize=(20, 4))
plt.xticks(rotation=90)
sns.countplot(data=rfm.sort_values('RFM_score'), x = 'RFM_segment')

rfm.iloc[:,:-2].astype(float).hist(figsize=(10,6), bins = 50)

# 3d projection ax.scatter3D rfm["R"], rfm["F"], rfm["M"]
ax = plt.axes(projection='3d')
ax.scatter3D(rfm['R'],rfm['F'],rfm['M'])

# rfm["Recency"], rfm["Frequency"], rfm["MonetaryValue"]
ax = plt.axes(projection='3d')
ax.scatter3D(rfm['Recency'],rfm['Frequency'],rfm['MonetaryValue'])

# 문자열의 format 함수를 사용하여 소수점 아래는 표기하지 않도록({:,.0f}) 문자열 포맷을 지정합니다.
rfm.groupby("RFM_score").agg({"Recency": "mean", 
                              "Frequency" : "mean", 
                              "MonetaryValue" : ["mean", "sum"]
                             }).style.background_gradient().format("{:,.0f}")

# qcut 을 통해 3단계로 "silver", "gold", "platinum" 고객군을 나눕니다. 
rfm["RFM_class"] = pd.qcut(rfm['RFM_score'], 3, labels = ['silver','gold','platinum'])

# "RFM_class" 별로 그룹화 하고 "RFM_score" 의 describe 값을 구합니다.
rfm.groupby('RFM_class')['RFM_score'].describe()
























