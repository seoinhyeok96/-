# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 22:53:45 2023

@author: NADA
"""

import pandas as pd
import numpy as np
import seaborn as sns
import datetime as dt
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

'''데이터 로드 및 전처리'''
df = pd.read_parquet('df_final.parquet.gzip')


# 결측치 제거
df = df.dropna()

# 중복값 제거
df = df.drop_duplicates()

df_entire = df

df.head()

df.shape

df = df.drop('category_id', axis=1)
df.shape

sns.boxplot(data=df, x='price')



df['price'].describe()

df[df['price'] <= 0]['event_type'].unique()
df[df['user_id'] == 244951053]

df = df[df['event_type'] == 'purchase']


last_timestamp = df['event_time'].max() + dt.timedelta(days=1)
last_timestamp
'''RFM'''
rfm = df.groupby('user_id').agg({'event_time' : lambda x: (last_timestamp - x.max()).days,
                                'product_id' : 'count',
                                'price' : 'sum'})

rfm.columns = ['Recency', 'Frequency', 'MonetaryValue']
cut_size=5
r_cut = pd.qcut(rfm['Recency'].rank(method='first'), cut_size, labels=list(range(5,0,-1)))
f_cut = pd.qcut(rfm['Frequency'].rank(method='first'), cut_size, labels=list(range(1,6)))
m_cut = pd.qcut(rfm['MonetaryValue'].rank(method='first'), cut_size, labels=list(range(1,6)))
rfm['R'] = r_cut
rfm['F'] = f_cut
rfm['M'] = m_cut

rfm[(rfm['Recency'] == 26) & (rfm['Frequency'] == 1)]

rfm["RFM_segment"] = rfm['R'].astype(str) + rfm['F'].astype(str) + rfm['M'].astype(str)

rfm["RFM_score"] = rfm[['R', 'F', 'M']].sum(axis=1)

'''counterplot'''
# RFM segment 값에 따라 정렬합니다.
plt.figure(figsize=(20, 4))
plt.xticks(rotation=90)
sns.countplot(data=rfm.sort_values('RFM_score'), x = 'RFM_segment')

'''histogram'''
rfm.iloc[:,:-2].astype(float).hist(figsize=(10,8), bins = 50)

'''3d scatter'''
# 3d projection ax.scatter3D rfm["R"], rfm["F"], rfm["M"]
ax = plt.axes(projection='3d')
ax.scatter3D(rfm['R'],rfm['F'],rfm['M'])

'''3d scatter2'''
# rfm["Recency"], rfm["Frequency"], rfm["MonetaryValue"]
ax = plt.axes(projection='3d')
ax.scatter3D(rfm['Recency'],rfm['Frequency'],rfm['MonetaryValue'])



rfm_score = rfm.groupby('RFM_score').agg({'Recency':'mean',
                                         'Frequency':'mean',
                                         'MonetaryValue':['mean', 'sum']})
rfm_score.style.background_gradient().format('{:,.0f}')

rfm.head()

# qcut을 RFM_score에 따라 3단계 "silver", "gold", "platinum"로 고객군을 나눕니다.
cut_size = 3
labels = ['silver', 'gold', 'platinum']
rfm['RFM_class'] = pd.qcut(rfm['RFM_score'], cut_size, labels=labels)
rfm

pd.set_option('display.max_rows', None)
rfm[rfm['Frequency'] == 1].sort_values(['Recency', 'MonetaryValue'], ascending=[True, False])['RFM_class'].head(3000)

rfm.groupby('RFM_class')['RFM_score'].describe()

rfm.groupby('RFM_class').agg({'Recency':'mean', 'Frequency':'mean',
                             'MonetaryValue':['mean', 'sum', 'count']})

'''barplot'''
# barplot으로 RFM_class 별 평균 RFM_score 나타냅니다.
sns.barplot(data=rfm, x="RFM_class", y="RFM_score")

'''pointplot'''
# pointplot으로 x=R, hue=RFM_class 별 평균 y=RFM_score 나타냅니다.
# hue 옵션을 사용하면 특정 컬럼을 지정해서 표기할 수 있습니다.
sns.pointplot(data=rfm, x='R', hue='RFM_class', y='RFM_score')

# "RFM_class" 별로 그룹화합니다.
# "Recency", "Frequency" 의 평균을 구합니다.
# "MonetaryValue"의 "mean", "sum", "count" 값을 구합니다.
rfm_class_agg = rfm.groupby("RFM_class").agg({"Recency": "mean", 
                              "Frequency" : "mean", 
                              "MonetaryValue" : ["mean", "sum",'count']})
rfm_class_agg



file_path_parquet = 'rfm.parquet.gzip'
rfm.to_parquet(file_path_parquet, compression='gzip')
pd.read_parquet(file_path_parquet).head()


'''조인해야함'''
merged_df = pd.merge(rfm, df_entire, on='user_id', how='inner')

'''집단별 view, cart, purchase 비율 확인'''
# RFM_class별로 event_type이 view, cart, purchase 인 건수
counts = merged_df.groupby(['RFM_class', 'event_type'])['user_id'].count().reset_index()

# RFM_class별로 총 건수 구하기
total_counts = merged_df.groupby(['RFM_class'])['user_id'].count().reset_index()

# RFM_class별로 event_type이 view, cart, purchase 인 정도의 비율
result = pd.merge(counts, total_counts, on='RFM_class')
result['rate'] = result['user_id_x'] / result['user_id_y']

'''view, cart, purchase 인 정도의 비율 시각화 - bar그래프'''
result2 = result.plot(kind='bar', x='event_type', y='rate', color=['blue', 'green', 'red'], legend=False)
plt.show()

'''silver, gold, platinum purchase비율만 보여주자'''
# purchase부분만 추출
result_purchase = result.loc[result['event_type'] == 'purchase']
result_purchase = result_purchase.drop(['user_id_x','user_id_y','event_type'],axis=1)
# 그래프로 표현하기
plt.rcParams['font.family'] = 'NanumGothic'
plt.bar(result_purchase['RFM_class'], result_purchase['rate'], color=['silver', 'gold', 'lightseagreen'])
plt.title('class 별 purchase 비율')
plt.xlabel('RFM_class')
plt.ylabel('rate')
plt.show()


'''user_id'''
# user_id별 view, cart, purchase 빈도수 구하기
id_count = merged_df.groupby(['user_id', 'event_type'])['event_type'].count().unstack()
id_count.columns = ['cart', 'purchase', 'view']
id_count['total'] = id_count.sum(axis=1)
# purchase 비율 계산
id_count['purchase_rate'] = id_count['purchase'] / id_count['total']

'''prod_count'''
# product_id별 view, cart, purchase 빈도수 구하기
prod_count = merged_df.groupby('product_id')['event_type'].value_counts().unstack()
prod_count.columns = ['cart', 'purchase', 'view']
prod_count['total'] = prod_count.sum(axis=1)
# purchase 비율 계산
prod_count['purchase_rate'] = prod_count['purchase'] / prod_count['total']

merged_df.category_id.nunique()


'''category_code로 프레임 만들고 전처리'''
# category_code view, cart, purchase 빈도수 구하기
cate_count = merged_df.groupby('category_code')['event_type'].value_counts().unstack()
cate_count.columns = ['cart', 'purchase', 'view']
cate_count['total'] = cate_count.sum(axis=1)
# purchase 비율 계산
cate_count['purchase_rate'] = cate_count['purchase'] / cate_count['total']

# 구매비율 상위 10개 추출
cate_top10 = cate_count.nlargest(10, 'purchase_rate') # 주로 기기제품인 것을 확인 가능
# 이름 변경
cate_top10.index = ['diaper', 'smartphone', 'headphone', 'tv', 'tonometer', 'camera', 'microwave', 'iron', 'water_heater', 'tablet']
# purchase_rate만 남기기
cate_top10 = cate_top10['purchase_rate']

cate_top10= cate_top10.reset_index()
cate_top10.rename(columns = {'index':'prod_name'}, inplace = True)

'''그래프 그리기'''
fig, ax = plt.subplots(figsize=(10,6))
ax.bar(cate_top10['prod_name'], cate_top10['purchase_rate'], color='brown')

# x축 설정
ax.set_xticks(cate_top10.index)
ax.set_xticklabels(cate_top10['prod_name'], rotation=45)

# 그래프 제목
ax.set_title('Purchase Rates of Products')
ax.set_xlabel('Product Names')
ax.set_ylabel('Purchase Rates')
plt.show()





