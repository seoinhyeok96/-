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

df = pd.read_parquet('df.sample.parquet.gzip')
df_entire = pd.read_parquet('df.sample.parquet.gzip')

df.head()

df.shape

df = df.drop('category_id', axis=1)
df.shape

sns.boxplot(data=df, x='price')

df.info()

df['price'].describe()

df[df['price'] <= 0]['event_type'].unique()
df[df['user_id'] == 244951053]

df = df[df['event_type'] == 'purchase']
df.shape

last_timestamp = df['event_time'].max() + dt.timedelta(days=1)
last_timestamp

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
rfm
rfm[(rfm['Recency'] == 26) & (rfm['Frequency'] == 1)]

rfm["RFM_segment"] = rfm['R'].astype(str) + rfm['F'].astype(str) + rfm['M'].astype(str)
rfm["RFM_segment"].value_counts()
rfm["RFM_score"] = rfm[['R', 'F', 'M']].sum(axis=1)

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
import pandas as pd
import matplotlib.pyplot as plt
result = result.plot(kind='bar', x='event_type', y='rate', color=['blue', 'green', 'red'], legend=False)
plt.show()
'''user_id와 product_id별 view, cart, purchase 빈도수 확인'''
# user_id와 event_type으로 그룹화하여 개수 세기



# user_id별 view, cart, purchase 빈도수 구하기
id_count = merged_df.groupby(['user_id', 'event_type'])['event_type'].count().unstack()
id_count.columns = ['cart', 'purchase', 'view']
id_count['total'] = id_count.sum(axis=1)
# purchase 비율 계산
id_count['purchase_rate'] = id_count['purchase'] / id_count['total']

# product_id별 view, cart, purchase 빈도수 구하기
prod_count = merged_df.groupby('product_id')['event_type'].value_counts().unstack()
prod_count.columns = ['cart', 'purchase', 'view']
prod_count['total'] = prod_count.sum(axis=1)
# purchase 비율 계산
prod_count['purchase_rate'] = prod_count['purchase'] / prod_count['total']

merged_df.category_id.nunique()





# product_id별 view, cart, purchase 빈도수 구하기
cate_count = merged_df.groupby('category_code')['event_type'].value_counts().unstack()
cate_count.columns = ['cart', 'purchase', 'view']
cate_count['total'] = cate_count.sum(axis=1)
# purchase 비율 계산
cate_count['purchase_rate'] = cate_count['purchase'] / cate_count['total']

merged_df.category_id.nunique()



