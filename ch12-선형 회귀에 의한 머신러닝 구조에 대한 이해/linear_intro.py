#
# 프로그램 이름: linear_intro.py
# 작성자: Bong Ju Kang
# 설명: 선형 회귀 문제를 통한 머신러닝의 기본 구조 익히기
#

# 필요한 패키지
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston

# 공통 변수 및 함수 설정
png_path = "./data/png"
os.makedirs(png_path, exist_ok=True)

# 한글출력
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 불러오기
from sklearn.datasets import load_boston
X = load_boston()['data']
Y = load_boston()['target'] # 주택 중간 가격
load_boston()

# 단순 회귀 분석을 위하여 하나의 입력 변수 선택
X = X[:,5] # average number of rooms per dwelling: 주거 당 평균 방 개수
df = pd.DataFrame(np.column_stack((X.reshape(-1, 1), Y.reshape(-1,1))))
df.columns = ['RM', 'MEDV']
df.head()
#       RM  MEDV
# 0  6.575  24.0
# 1  6.421  21.6
# 2  7.185  34.7
# 3  6.998  33.4
# 4  7.147  36.2

# Attribute Information (in order):
#    - CRIM     per capita crime rate by town
#    - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
#    - INDUS    proportion of non-retail business acres per town
#    - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
#    - NOX      nitric oxides concentration (parts per 10 million)
#    - RM       average number of rooms per dwelling
#    - AGE      proportion of owner-occupied units built prior to 1940
#    - DIS      weighted distances to five Boston employment centres
#    - RAD      index of accessibility to radial highways
#    - TAX      full-value property-tax rate per $10,000
#    - PTRATIO  pupil-teacher ratio by town
#    - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
#    - LSTAT    % lower status of the population
#    - MEDV     Median value of owner-occupied homes in $1000's

# 입력 특징과 목표 변수와의 산점도
plt.figure(figsize=(6, 5))
plt.scatter(X, Y, s=12)
plt.xlabel("방 개수")
plt.ylabel("주택 가격")
plt.title("방 개수 대 주택 가격")
plt.savefig(png_path + '/lr_xy_scatter.png')
plt.show()

# 입력 특징에 대한 표준화
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0)
X_scale = (X - X_mean) / X_std

# 편의(bias) 변수 추가
# X_bias = np.ones(X.shape[0]).reshape(-1, 1) # 하나의 컬럼으로 변환
X_aug = np.column_stack((np.ones(X_scale.shape[0]).reshape(-1,1), X_scale.reshape(-1, 1))) # 편의 추가된 데이터

# 비용 함수 정의
def cost(X_aug, Y, Theta):
    hypothesis = X_aug.dot(Theta.transpose())
    return 0.5 * (np.square(hypothesis - Y.reshape(-1,1)).mean(axis=0))

#
# 기울기 하강법 알고리즘
#
def gradientDescent(X_aug, Y, Theta, iterations, alpha):
    count = 1
    cost_log = []
    Theta_log = []
    while count <= iterations:
        hypothesis = X_aug @ Theta.transpose() # 행렬 곱
        Theta[0, 0] = Theta[0, 0] - alpha * ((hypothesis - Y.reshape(-1, 1)) * (X_aug[:, 0:1])).mean(axis=0)
        Theta[0, 1] = Theta[0, 1] - alpha * ((hypothesis - Y.reshape(-1, 1)) * (X_aug[:, 1:])).mean(axis=0)
        print(Theta) # 진척 사항 확인
        cost_log.append(cost(X_aug, Y, Theta))
        Theta_log.append(Theta.copy())
        count = count + 1
    return cost_log, Theta, Theta_log

# 적용
Theta = np.random.RandomState(123).rand(1, 2) # 초기값
alpha = 0.3
iterations = 20
cost_log, Theta, Theta_log = gradientDescent(X_aug, Y, Theta, iterations, alpha)
min_index = np.argmin(np.array(cost_log), axis=0) # 비용 최소 값을 주는 인덱스 가져오기
Theta = np.array(Theta_log)[min_index] # 인덱스에 해당하는 모수 추정값 가져오기

# 반복 수에 따른 비용 함수값의 변화 그래프
fig = plt.figure()
plt.plot(np.arange(len(cost_log)), np.array(cost_log))
plt.title("반복수 대 비용함수값")
plt.xlabel("반복수")
plt.ylabel("비용함수값")
plt.savefig(png_path + '/lr_gradient_descent_cost.png')
plt.show()

# 추정된 모수에 의한 그래프
predicted = X_aug @ Theta.reshape(X_aug.shape[1], -1)
plt.figure(figsize=(10, 6))
plt.scatter(X, Y, label='실제 값')
plt.plot(X, predicted, color='orange', label='회귀 직선')
plt.xlabel("방 개수")
plt.ylabel("가격")
plt.title("방 개수 대 가격 산점도 및 회귀 직선")
plt.legend()
plt.savefig(png_path + '/lr_fitted_with_scatter.png')
plt.show()

#
# 비용함수 값의 등고선
#
theta0_list = np.linspace(5, 40, 101)
theta1_list = np.linspace(-10, 20, 101)
theta0_mesh, theta1_mesh = np.meshgrid(theta0_list, theta1_list)

Theta_flat = np.dstack((theta0_mesh, theta1_mesh)).reshape(-1, 2)
cost_mesh = cost(X_aug, Y, Theta_flat).reshape(theta0_mesh.shape)
theta_list = np.array(Theta_log).reshape(-1, 2)

fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(1,1,1)
# contours = ax.contourf(theta0_mesh, theta1_mesh, cost_mesh, 15, cmap=cm.coolwarm)
contours = ax.contour(theta0_mesh, theta1_mesh, cost_mesh, 15)
ax.clabel(contours, fmt="%.0f")
for j in np.arange(1, len(theta_list)):
    ax.annotate('', xy=theta_list[j], xytext=theta_list[j - 1],
                arrowprops={'arrowstyle': '->', 'color': 'r', 'lw': 1},
                va='center', ha='center')
ax.scatter(theta_list[:, 0], theta_list[:, 1], s=40, lw=0)
ax.scatter(theta_list[-1, 0], theta_list[-1, 1], s=50, color='k', marker='*')
ax.scatter(theta_list[-1, 0], theta_list[-1, 1], s=10, color='w', marker='*')
ax.set_xlabel(r'$\theta_0$')
ax.set_ylabel(r'$\theta_1$')
ax.set_title('비용함수 값의 등고선')
plt.savefig(png_path + '/lr_gradient_descent_contour.png')


