#
# 프로그램 이름: model_assess_intro.py
# 작성자: Bong Ju Kang
# 설명: 모델 평가의 기본을 예제와 함께 이해하기
#

# 필요한 패키지
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression

# 공통 변수 및 함수 설정
png_path = "./data/png"
os.makedirs(png_path, exist_ok=True)

# 한글출력
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 과학 표기법을 제거
np.set_printoptions(suppress=True)


# 데이터 가져오기
df = pd.read_csv("http://www-bcf.usc.edu/~gareth/ISL/Income1.csv",
                 index_col=[0]) # 처음 열을 인덱스로 사용
df.head(3)
#    Education     Income
# 1  10.000000  26.658839
# 2  10.401338  27.306435
# 3  10.842809  22.132410

# 변수 구성
X = df[["Education"]].values # 넘파이 배열로 변환
Y = df["Income"].values # 넘파이 배열로 변환


# 원래 함수 가정: 서포트벡터 회귀
reg = SVR(kernel='rbf', C=30, gamma=0.015)
reg.fit(X, Y)
Y_hat = reg.predict(X)

# 원 데이터 산점도
fig = plt.figure(figsize=(12, 5))
fig.add_subplot(1,2,1)
plt.scatter(X, Y, c='r')
plt.xlabel("교육기간(년)")
plt.ylabel("소득")
plt.show()
plt.savefig(png_path+'/model_assess_scatter.png')

# 원 함수 추가
fig.add_subplot(1,2,2)
plt.scatter(X, Y, c='r')
plt.vlines(X, ymin=Y_hat, ymax=Y, colors='k', linewidth=1)
plt.plot(X, Y_hat, linestyle='-', c='blue')
plt.xlabel("교육기간(년)")
plt.ylabel("소득")
plt.subplots_adjust(bottom=0.15)

# 그래프 저장
plt.savefig(png_path + '/modelassess_png01_intro.png')
# 그래프 화면 출력
plt.show()

