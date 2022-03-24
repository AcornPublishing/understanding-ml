#
# 프로그램 이름: association_basics.py
# 작성자: Bong Ju Kang
# 설명: 연관 분석 이해하기
#

# 필요한 패키지
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import cm
from numpy.random import RandomState

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs
from sklearn.metrics import silhouette_score, silhouette_samples

# 초기 설정
png_path = "./data/png"
os.makedirs(png_path, exist_ok=True)

# 한글출력
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

#
# 본문 예시 예제
#

# 데이터 생성
tr_data = [['milk', 'bread'],
           ['butter'],
           ['beer', 'diapers'],
           ['milk', 'bread', 'butter'],
           ['bread']]

# 데이터 전 처리
tr_encoder = TransactionEncoder()
tr_encoder.fit(tr_data)
tr_encoder.columns_
# ['beer', 'bread', 'butter', 'diapers', 'milk']

tr_encoder_ary =  tr_encoder.transform(tr_data)
tr_encoder_ary = np.where(tr_encoder_ary==True, 1, 0)
df = pd.DataFrame(tr_encoder_ary, columns=tr_encoder.columns_ )
print(df)
#    beer  bread  butter  diapers  milk
# 0     0      1       0        0     1
# 1     0      0       1        0     0
# 2     1      0       0        1     0
# 3     0      1       1        0     1
# 4     0      1       0        0     0

# 연관 모델 적합
freq_items = apriori(df, min_support=0.4,  use_colnames=True, n_jobs=-1)

freq_items['length'] = freq_items['itemsets'].apply(lambda x: len(x))

# itemsets 컬럼은 frozenset 속성을 갖고 있다. 즉, 변경이 불가능한 집합으로 구성되어 있다.
type(freq_items['itemsets'][0])
# frozenset

# 아래의 2개의 결과는 같다.
freq_items[freq_items['itemsets'] == {'bread', 'milk'}]
freq_items[freq_items['itemsets'] == {'milk', 'bread'}]

#
# 선험 알고리즘 예시
#
# 데이터 구성
tr_data = [[1, 2, 3, 4],
           [1, 2, 4],
           [1, 2],
           [2, 3, 4],
           [2, 3],
           [3, 4],
           [2, 4]]

# 데이터 전 처리
tr_encoder = TransactionEncoder()
tr_encoder.fit(tr_data)
tr_encoder.columns_

tr_encoder_ary =  tr_encoder.transform(tr_data)
tr_encoder_ary = np.where(tr_encoder_ary==True, 1, 0)
df = pd.DataFrame(tr_encoder_ary, columns=tr_encoder.columns_)
print(df)

# 연관 모델 적합
# 길이가 1개짜리 전부
freq_items = apriori(df, min_support=0, max_len=1, use_colnames=True)
print(freq_items)

# 지지도가 3/7= 0.43 이상인 경우만
freq_items = apriori(df, min_support=3/7, max_len=1, use_colnames=True)
print(freq_items)

# 길이가 2개짜리 전부
freq_items = apriori(df, min_support=0, max_len=2, use_colnames=True)
freq_items['length'] = freq_items['itemsets'].apply(lambda x: len(x))

# 길이가 2개이면서, 지지도 조건을 충족하는 경우
print(freq_items[(freq_items['length'] == 2) & (freq_items['support'] >= 3.0/7.0)])
#     support itemsets  length
# 4  0.428571   (1, 2)       2
# 7  0.428571   (2, 3)       2
# 8  0.571429   (2, 4)       2
# 9  0.428571   (3, 4)       2

# 길이가 3개짜리 전부
freq_items = apriori(df, min_support=0, max_len=3, use_colnames=True)
freq_items['length'] = freq_items['itemsets'].apply(lambda x: len(x))
print(freq_items[freq_items['length'] == 3])
#      support   itemsets  length
# 10  0.142857  (1, 2, 3)       3
# 11  0.285714  (1, 2, 4)       3
# 12  0.142857  (1, 3, 4)       3
# 13  0.285714  (2, 3, 4)       3
# 길이가 3이면서 지지도 조건을 충족하는 경우
print(freq_items[(freq_items['length'] == 3) & (freq_items['support'] >= 3.0/7.0)])
# Empty DataFrame
# Columns: [support, itemsets, length]
# Index: []
# 길이가 3인 경우는 고빈도 집합이 없으므로 길이가 2인 경우로 마무리 된다.

#
# 규칙 생성
#

# 규칙을 생성하기 위한 고빈도 집합 생성
freq_items = apriori(df, min_support=3/7, max_len=3, use_colnames=True)

# 고빈도 집합 기반으로 규칙 생성: support, confidence, lift 기준 적용 가능
asso_rules = association_rules(freq_items, metric='confidence',  min_threshold=1)
print(asso_rules.columns)
# ['antecedents', 'consequents', 'antecedent support',
#        'consequent support', 'support', 'confidence', 'lift', 'leverage',
#        'conviction']

# 결과 확인
asso_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]

#
# 예제: [RETAIL] 데이터를 이용한 예제
#

#
# 데이터 구성
#
# 데이터 불러오기
retail = pd.read_excel('https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx')
# retail.to_csv(data_path+'/retail.csv')
retail.columns
# Index(['InvoiceNo', 'StockCode', 'Description', 'Quantity', 'InvoiceDate',
#        'UnitPrice', 'CustomerID', 'Country'],
#       dtype='object')

# 소문자로 변수명를 사용하기 위하여
new_columns = retail.columns.str.lower()
retail.columns = new_columns

# 데이터 탐색
retail['stockcode'].value_counts()
# Name: stockcode, Length: 4070, dtype: int64
retail['country'].value_counts()[:5]
# United Kingdom    495478
# Germany             9495
# France              8557
# EIRE                8196
# Spain               2533
# Name: country, dtype: int64

#
# 일부 데이터만 구성: Germany에서만 팔린 데이터로 구성
#
df = retail[retail.country=='Germany']
df.stockcode.value_counts()
# Name: stockcode, Length: 1671, dtype: int64

# 데이터 전 처리: 거래 데이터 구성

# 결측값 확인
df.isna().sum()

# 거래 데이터 구성
trxs = df.groupby(['invoiceno', 'description'])['quantity'].sum().unstack(fill_value=0).reset_index().set_index('invoiceno')
trxs_df = pd.DataFrame(np.where(trxs > 1, 1, 0), columns=trxs.columns, index=trxs.index)

# 고빈도 집합 찾기
supp_cutoff = 0.05 # 5%
pd.set_option('display.max_columns', 15)
pd.set_option('display.width', 500)

freq_items = apriori(trxs_df, min_support=supp_cutoff, use_colnames=True)
freq_items.sort_values(by='support', ascending=False).head()
#      support                                        itemsets
# 12  0.442786                                       (POSTAGE)
# 18  0.185738           (ROUND SNACK BOXES SET OF4 WOODLAND )
# 34  0.139303  (ROUND SNACK BOXES SET OF4 WOODLAND , POSTAGE)
# 17  0.119403            (ROUND SNACK BOXES SET OF 4 FRUITS )
# 11  0.104478              (PLASTERS IN TIN WOODLAND ANIMALS)

# 고빈도 집합 기반의 규칙 찾기
rules = association_rules(freq_items, metric='lift', min_threshold=1)
rules.sort_values(by='lift', ascending=False, inplace=True)

# 후행 품목집합이 있는 경우
rules[rules.consequents != ''][['antecedents', 'consequents', 'support', 'confidence', 'lift']].head()
#                                        antecedents                                     consequents   support  confidence      lift
# 4               (PLASTERS IN TIN WOODLAND ANIMALS)                (PLASTERS IN TIN CIRCUS PARADE )  0.051410    0.492063  5.598383
# 5                 (PLASTERS IN TIN CIRCUS PARADE )              (PLASTERS IN TIN WOODLAND ANIMALS)  0.051410    0.584906  5.598383
# 34            (ROUND SNACK BOXES SET OF 4 FRUITS )  (ROUND SNACK BOXES SET OF4 WOODLAND , POSTAGE)  0.079602    0.666667  4.785714
# 31  (ROUND SNACK BOXES SET OF4 WOODLAND , POSTAGE)            (ROUND SNACK BOXES SET OF 4 FRUITS )  0.079602    0.571429  4.785714
# 26           (ROUND SNACK BOXES SET OF4 WOODLAND )            (ROUND SNACK BOXES SET OF 4 FRUITS )  0.099502    0.535714  4.486607

