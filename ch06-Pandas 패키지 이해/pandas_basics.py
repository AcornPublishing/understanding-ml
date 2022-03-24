#
# 프로그램 이름: pandas_basics.py
# 작성자: Bong Ju Kang
# 설명: Pandas의 기본을 예제와 함께 이해하기
#

# 필요한 패키지
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 초기 설정
png_path = "./data/png"
os.makedirs(png_path, exist_ok=True)

#
# 데이터 구조
#

# 데이터프레임 생성: 딕셔너리 사용
df = pd.DataFrame({'x': np.random.choice(['a', 'b', 'c'], size=10),
                   'y': np.random.randn(10),
                   'z': np.random.uniform(size=10)})

df # 생성된 데이터 구조
#    x         y         z
# 0  a  0.119330  0.852338
# 1  c -1.548951  0.172606
# 2  c  1.095493  0.904218
# 3  a -0.729121  0.117180
# 4  c -1.950831  0.185651
# 5  a -0.095028  0.190626
# 6  b -1.436222  0.962624
# 7  c  0.522623  0.629979
# 8  b -1.360020  0.998643
# 9  a -1.566866  0.600820

df.index # 행 레이블
# RangeIndex(start=0, stop=10, step=1)

df.columns # 열 레이블
# Index(['x', 'y', 'z'], dtype='object')

df = pd.DataFrame({'x': np.random.choice(['a', 'b', 'c'], size=10),
                   'y': np.random.randn(10),
                   'z': np.random.uniform(size=10)}, index = np.arange(1,11))
df.index
# Int64Index([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype='int64')

# 데이터프레임 생성: 리스트 이용
df2 = pd.DataFrame([[1, 2, 3],
                    [4, 5, 6]], columns=['x', 'y', 'z'])
df2.columns
# Index(['x', 'y', 'z'], dtype='object')

# 데이터 프레임 생성: 외부 파일 또는 데이터베이스
file = './data/Credit.csv'
file
# './data/Credit.csv'

df = pd.read_csv(file)
df.head()
#     Income  Limit  Rating  Cards   ...    Student  Married  Ethnicity Balance
# 0   14.891   3606     283      2   ...         No      Yes  Caucasian     333
# 1  106.025   6645     483      3   ...        Yes      Yes      Asian     903
# 2  104.593   7075     514      4   ...         No       No      Asian     580
# 3  148.924   9504     681      3   ...         No       No      Asian     964
# 4   55.882   4897     357      2   ...         No      Yes  Caucasian     331

df.columns
# Index(['Income', 'Limit', 'Rating', 'Cards', 'Age', 'Education', 'Gender',
#        'Student', 'Married', 'Ethnicity', 'Balance'],
#       dtype='object')

#
# 데이터 부분 구성
#

df = pd.DataFrame({'x': np.random.choice(['a', 'b', 'c'], size=10),
                   'y': np.random.randn(10),
                   'z': np.random.uniform(size=10)})
df.index
df.columns


df['x'].head(3) # 열 레이블을 이용
# 0    b
# 1    c
# 2    a
# Name: x, dtype: object
type(df['x']) # 열이 1개인 경우는 시리즈로 반환
# pandas.core.series.Series

df.x.head(3) # 열이 1개인 경우는 .을 사용하여 처리해도 됨


df[['x', 'y']].head(3) # 열 레이블을 이용, 변수가 2개 이상인 경우는 리스트로 처리
#    x         y
# 0  b -1.611294
# 1  c  0.731318
# 2  a  0.523759

type(df[['x', 'y']]) # 열이 2개 이상인 경우는 데이터프레임으로 반환
# pandas.core.frame.DataFrame

df.groupby('x').mean() # 열 레이블 인덱스를 이용한 예
#           y         z
# x
# a  1.098474  0.828929
# b -0.411463  0.420848
# c  0.619459  0.673797

df.loc[:3, 'x'] # 행 인덱스와 열 인덱스를 이용, 행 인덱스는 3까지!
# 0    b
# 1    c
# 2    a
# 3    a
# Name: x, dtype: object

df.loc[:2] # 열 인덱스를 생략하는 경우
#    x         y         z
# 0  b -1.611294  0.063884
# 1  c  0.731318  0.572941
# 2  a  0.523759  0.618564

df.loc[:,'x'].head(3) # 행 인덱스는 생략할 수가 없음
# 0    b
# 1    c
# 2    a

(df['x'] == 'b').head(3) # 불린 마스크
# 0    False
# 1     True
# 2    False

df.loc[df['x'] == 'b']
#    x         y         z
# 1  b -0.966825  0.821148
# 5  b  1.008843  0.392186


df.iloc[0:3,-2: ] # 행과 열의 위치를 이용
#           y         z
# 0 -1.611294  0.063884
# 1  0.731318  0.572941
# 2  0.523759  0.618564

# 행과 열의 일부 제거
df.drop('x', axis=1) # x 열 제거
df.drop(['x', 'y'], axis=1) # x, y 열 제거
df.drop([0,7],axis=0) # [0,7] 인덱스에 해당하는 행의 제거

#
# 데이터 요약
#
df = pd.DataFrame({'x': np.random.choice(['a', 'b', 'c'], size=10),
                   'y': np.random.randn(10),
                   'z': np.random.uniform(size=10),
                   'date': pd.date_range(start='2004-11-03', periods=10)})

df.head(3)
#    x         y         z       date
# 0  c -1.201629  0.365122 2004-11-03
# 1  c -0.539897  0.594471 2004-11-04
# 2  b -0.786518  0.589175 2004-11-05

df['x'].unique() # 유일한 값
# array(['c', 'a', 'b'], dtype=object)

df['x'].value_counts() # 유일한 값에 대한 빈도
# c    4
# b    4
# a    2

df.shape # 모양
# (10, 4)

df.dtypes
# x               object
# y              float64
# z              float64
# date    datetime64[ns]

df.describe() # 디폴트는 연속형 변수만 적용. 분포 확인
#                y          z
# count  10.000000  10.000000
# mean    0.259596   0.688980
# std     0.837817   0.259345
# min    -1.016186   0.108868
# 25%    -0.431274   0.558183
# 50%     0.419177   0.755178
# 75%     0.928590   0.876695
# max     1.276984   0.965005

df.describe(percentiles=np.arange(0,1, 0.1)).tail(3) # 분위수 지정
#             y         z
# 80%  1.025002  0.912173
# 90%  1.220105  0.948507
# max  1.276984  0.965005

# 요약 통계량: 축은 행을 따라 요약하므로 변수 별 요약의 의미
df[['y', 'z']].sum(axis=0)
df[['y', 'z']].median(axis=0)
df[['y', 'z']].std(axis=0)

df[['y', 'z']].apply(lambda x: x/np.max(x)).head(3) # 사용자 정의 함수 적용
#           y         z
# 0  0.611552  0.112816
# 1 -0.527487  0.747731
# 2 -0.361387  0.936314

#
# 신규 열 생성
#
df = pd.DataFrame({'x': np.random.choice(['a', 'b', 'c'], size=10),
                   'y': np.random.randn(10),
                   'z': np.random.uniform(size=10),
                   'date': pd.date_range(start='2004-11-03', periods=10)})

df['new_var'] = np.random.choice(df['date'], 10) # 1개 신규 변수(열) 직접 생성
df.head(3)
#    x         y         z       date    new_var
# 0  a -0.161018  0.842823 2004-11-03 2004-11-10
# 1  b  0.701476  0.925236 2004-11-04 2004-11-09
# 2  b  0.209900  0.315162 2004-11-05 2004-11-09

df = df.assign(ym = lambda df: df['y']-np.mean(df['y']),
               zm = lambda df: df['z']-np.mean(df['z'])) # 2개 이상의 신규 변수 생성
df.head(3)

#    x         y         z       date    new_var        ym        zm
# 0  a  0.957613  0.207489 2004-11-03 2004-11-12  1.284978 -0.297913
# 1  c -1.173657  0.267870 2004-11-04 2004-11-06 -0.846292 -0.237532
# 2  b -0.200748  0.112787 2004-11-05 2004-11-11  0.126618 -0.392615

#
# 결측값의 처리
#

df = pd.DataFrame({'x': np.random.choice(['a', 'b', 'c'], size=10),
                   'y': np.random.randn(10),
                   'z': np.random.uniform(size=10)})

df.shape
# (10, 3)

df.loc[np.random.choice(np.arange(10), 3), 'y'] = np.nan # 결측값 생성, 결측값은 NaN 으로 표시됨
df.head(3)
#    x         y         z
# 0  a       NaN  0.459980
# 1  b  1.158075  0.465997
# 2  c  0.236095  0.435350

df.isnull().head(3) # 결측값 탐지
#        x      y      z
# 0  False   True  False
# 1  False  False  False
# 2  False  False  False

df.isnull().sum(axis=0) # 모든 열에 대한 결측값 개수를 확인
# x    0
# y    3
# z    0

df.dropna() # 모든 결측값 제거, df에서 실제 제거되지는 않음.
df.shape
# (10, 3)
df.dropna().shape
# (7, 3)

# df.dropna(inplace=True) # df에서 실제 제거 됨
# df.shape
# (7, 3)

df.dropna(axis=0, how='all').head(3) # 행의 모든 값이 결측값일때만 제거
#    x         y         z
# 0  b  0.762938  0.988471
# 1  c -0.354025  0.331220
# 2  c       NaN  0.888432

df.fillna(method='ffill').head(3) # 결측값에 대하여 전진 채움 (forward fill) 방식
#    x         y         z
# 0  b  0.762938  0.988471
# 1  c -0.354025  0.331220
# 2  c -0.354025  0.888432

df.fillna(method='bfill').head(4) # 결측값에 대하여 후진 채움 (backward fill) 방식
#    x         y         z
# 0  b  0.762938  0.988471
# 1  c -0.354025  0.331220
# 2  c -0.719113  0.888432
# 3  c -0.719113  0.701478

df.fillna(value=df.mean()['y']) # 결측값을 평균값으로 대체

#
# 데이터 결합
#

df1 = pd.DataFrame({'x': np.random.randn(5),
                   'date': pd.date_range(start='2004-11-03', periods=5)})

df2 = pd.DataFrame({'y': np.random.random(5),
                   'date': pd.date_range(start='2004-11-05', periods=5)})

pd.merge(df1, df2, on='date', how='inner') # inner 조인
#           x       date         y
# 0 -0.739995 2004-11-05  0.427575
# 1 -0.009899 2004-11-06  0.079937
# 2 -0.304330 2004-11-07  0.004967

pd.merge(df1, df2, on='date', how='outer') # outer 조인
#           x       date         y
# 0  0.923241 2004-11-03       NaN
# 1 -1.108456 2004-11-04       NaN
# 2 -0.739995 2004-11-05  0.427575
# 3 -0.009899 2004-11-06  0.079937
# 4 -0.304330 2004-11-07  0.004967
# 5       NaN 2004-11-08  0.120919
# 6       NaN 2004-11-09  0.683926

pd.merge(df1, df2, on='date', how='left') # left 조인
#           x       date         y
# 0  0.923241 2004-11-03       NaN
# 1 -1.108456 2004-11-04       NaN
# 2 -0.739995 2004-11-05  0.427575
# 3 -0.009899 2004-11-06  0.079937
# 4 -0.304330 2004-11-07  0.004967

pd.merge(df1, df2, on='date', how='right') # right 조인
#           x       date         y
# 0 -0.739995 2004-11-05  0.427575
# 1 -0.009899 2004-11-06  0.079937
# 2 -0.304330 2004-11-07  0.004967
# 3       NaN 2004-11-08  0.120919
# 4       NaN 2004-11-09  0.683926

pd.concat([df1, df2]) # 행 축을 따라 그대로 연결
pd.concat([df1.set_index('date'), df2.set_index('date')], axis=1, join='inner') # inner 조인
pd.concat([df1.set_index('date'), df2.set_index('date')], axis=1, join='outer') # outer 조인

df2[df2['date'].isin(df1['date'])] # 교집합인 df2의 행
df2[~df2['date'].isin(df1['date'])] # df2-df1: 차집합

#
# 그룹화(grouping)
#

df = pd.DataFrame({'x': np.random.RandomState(123).choice(['a', 'b', 'c'], size=10),
                   'y': np.random.RandomState(123).randn(10),
                   'z': np.random.RandomState(123).uniform(size=10),
                   'date': pd.date_range(start='2004-11-03', periods=10)})

df.groupby('x').agg('mean') # 그룹화 후 각각 그룹 별 평균값
#           y         z
# x
# a -0.578600  0.719469
# b -0.099436  0.454362
# c -0.303042  0.559906

df.groupby('x').agg(['mean', 'std', 'count']) # 그룹화 후 각각 그룹 별 평균 및 표준편차
#           y                         z
#        mean       std count      mean       std count
# x
# a -0.578600       NaN     1  0.719469       NaN     1
# b -0.099436  0.974741     3  0.454362  0.206505     3
# c -0.303042  1.624512     6  0.559906  0.257574     6


df.groupby('x').describe() # 그룹화 후 각 그룹 별 분포
#   count      mean       std    ...          50%       75%       max
# x                              ...
# a   1.0 -0.578600       NaN    ...     0.719469  0.719469  0.719469
# b   3.0 -0.099436  0.974741    ...     0.392118  0.538474  0.684830
# c   6.0 -0.303042  1.624512    ...     0.516123  0.660181  0.980764

# 그룹 별로 평균 값을 구한 후 원래 값에서 빼줌, 원 데이터와 같은 모양
df.groupby('x').transform(lambda x: x - x.mean()).head(3)
#           y         z
# 0 -0.782588  0.136563
# 1  1.096781 -0.168223
# 2  0.586021 -0.333055

# 함수 정의 후 넘겨주기, 원 데이터와 같은 모양
def normalize(x):
    return (x - np.mean(x))/np.std(x)

df.groupby('x')['y', 'z'].transform(normalize).head(3)
#           y         z
# 0 -0.527717  0.580793
# 1  1.378086 -0.997700
# 2  0.395167 -1.416461


df.groupby('x').rank(ascending=False).head(3) # 그룹 별 내림 차순 등수
#      y    z  date
# 0  4.0  2.0   6.0
# 1  1.0  3.0   3.0
# 2  3.0  6.0   5.0

df.groupby('x').cumsum().head(3) # 그룹 별 누적 합
#           y         z
# 0 -1.085631  0.696469
# 1  0.997345  0.286139
# 2 -0.802652  0.923321

#
# 모양 변경(reshaping)
#
df = pd.DataFrame({'lot1': [0.1, 0.2, 0.7], 'lot2': [0.3, 0.1, 0.9], 'parameter': ['para1', 'para2', 'para3']})
df.head()
#    lot1  lot2 parameter
# 0   0.1   0.3     para1
# 1   0.2   0.1     para2
# 2   0.7   0.9     para3


pd.melt(df) # 녹여진 상태를 확인한 후 원하는 모양을 정의
#     variable  value
# 0       lot1    0.1
# 1       lot1    0.2
# 2       lot1    0.7
# 3       lot2    0.3
# 4       lot2    0.1
# 5       lot2    0.9
# 6  parameter  para1
# 7  parameter  para2
# 8  parameter  para3

# 각 파라미터와 로트 별로 값을 정의하고 함.
melt_df = pd.melt(df, id_vars=['parameter'],
                  value_vars=['lot1', 'lot2'],
                  value_name='value',
                  var_name='lot_id').sort_values('parameter')
melt_df
#   parameter lot_id  value
# 0     para1   lot1    0.1
# 3     para1   lot2    0.3
# 1     para2   lot1    0.2
# 4     para2   lot2    0.1
# 2     para3   lot1    0.7
# 5     para3   lot2    0.9

melt_df.groupby('parameter').agg(['mean', 'std'])
#           value
#            mean       std
# parameter
# para1      0.20  0.141421
# para2      0.15  0.070711
# para3      0.80  0.141421
melt_df.groupby('lot_id').boxplot()
plt.savefig(png_path + '/pandas_boxplot.png')

# pivot을 통하여 원래 데이터로 복귀
pd.pivot_table(melt_df, index=['parameter'], columns=['lot_id'], values='value', aggfunc='sum').reset_index()
#      parameter  lot1  lot2
# 0          para1   0.1   0.3
# 1          para2   0.2   0.1
# 2          para3   0.7   0.9

#
# 데이터 정렬
#

df = pd.DataFrame({'x': np.random.RandomState(123).choice(['a', 'b', 'c'], size=10),
                   'y': np.random.RandomState(123).randn(10),
                   'z': np.random.RandomState(123).uniform(size=10),
                   'date': pd.date_range(start='2004-11-03', periods=10)})
df.sort_values(by="x")










