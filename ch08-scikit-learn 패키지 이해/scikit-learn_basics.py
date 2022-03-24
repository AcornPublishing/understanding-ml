#
# 프로그램 이름: scikit-learn_basics.py
# 작성자: Bong Ju Kang
# 설명: scikit-learn의 기본을 예제와 함께 이해하기
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
# 데이터 불러오기: 사이킷런 내장 데이터
#
from sklearn.datasets import load_iris # 데이터 지정

skd = load_iris()
type(skd) # 사이킷런 데이터 형식
# sklearn.utils.Bunch

skd
# {'data': array([[5.1, 3.5, 1.4, 0.2],
#         [4.9, 3. , 1.4, 0.2],
#         [4.7, 3.2, 1.3, 0.2],
#                 ...
#         [6.5, 3., 5.2, 2.],
#         [6.2, 3.4, 5.4, 2.3],
#         [5.9, 3., 5.1, 1.8]]),
#  'target': array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#                   2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]),
#  'target_names': array(['setosa', 'versicolor', 'virginica'], dtype='<U10'),
#  'DESCR': 'Iris Plants Database\n====================\n\nNotes\n-----\nData Set Characteristics:\n',
#   ...
#  'feature_names': ['sepal length (cm)',  'sepal width (cm)',
#                    # 'petal length (cm)',  'petal width (cm)']}


# 손글씨 데이터 불러오기 (8x8)
from sklearn.datasets import load_digits
load_digits().keys()
# dict_keys(['data', 'target', 'target_names', 'images', 'DESCR'])

load_digits().data.shape # 모양 확인
# (1797, 64)

load_digits().target.shape # 모양 확인
# (1797,)

#
# 데이터 분할
#
bunch = load_digits() # 데이터 불러오기

y = bunch.target
np.unique(y)
# array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
np.bincount(y)
# array([178, 182, 177, 183, 181, 182, 181, 179, 174, 180], dtype=int64)

fig = plt.figure(figsize=(5,5))
plt.imshow(bunch.images[0], cmap='gray_r')
fig.savefig(png_path +'/scikit_image0.png')
plt.show()

# 데이터 불러오기
from sklearn.datasets import load_digits

bunch = load_digits()
X, y = bunch.data, bunch.target # 특징 데이터, 목표 데이터

from sklearn.model_selection import train_test_split
# X,y의 샘플의 수는 반드시 일치하여야 함

# 데이터 분할
train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=123)

type(train_X) # 결과는 모두 넘파이 배열임
# numpy.ndarray

train_X.shape
# (1257, 64)
test_X.shape
# (540, 64)
train_y.shape
# (1257,)

#
# 모델 적합 및 평가 (소프트맥스 분류)
#
# 소프트맥스 회귀 분석 적용
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(random_state=123, solver='lbfgs', multi_class='multinomial') # 모델 구성
modelfit = model.fit(train_X, train_y) # 모델 적합

train_predicted = modelfit.predict(train_X) # 모델 예측
train_predicted_prob = modelfit.predict_proba(train_X) # 확률 예측

test_predicted = modelfit.predict(test_X)

modelfit.score(train_X, train_y) # 훈련 데이터 정확도 계산
# 1.0
np.sum(train_predicted == train_y)/np.shape(train_y) # 훈련 데이터 정확도 수작업 계산

modelfit.score(test_X, test_y) # 평가 데이터 정확도 계산
# 0.9666666666666667
np.sum(test_predicted == test_y)/np.shape(test_y) # 평가 데이터 정확도 수작업 계산

# random forest 적용
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=123, n_estimators=100) # 모델 구성
modelfit = model.fit(train_X, train_y) # 모델 적합

train_predicted = modelfit.predict(train_X) # 모델 예측
train_predicted_prob = modelfit.predict_proba(train_X) # 확률 예측

test_predicted = modelfit.predict(test_X)

modelfit.score(train_X, train_y) # 훈련 데이터 정확도 계산
# 1.0
np.sum(train_predicted == train_y)/np.shape(train_y) # 훈련 데이터 정확도 수작업 계산

modelfit.score(test_X, test_y) # 평가 데이터 정확도 계산
# 0.96851852

# 특징 중요도
np.argsort(modelfit.feature_importances_)[::-1]
# array([21, 43, 26, 20, 42, 36, 30, 28, 33, 13, 60, 34, 10, 19,  2, 61, 18,
#        38, 51, 53, 27, 44, 46, 29, 54, 62,  5, 35, 37, 50, 58, 45, 25, 52,

# 다층 신경망 적용
from sklearn.neural_network import MLPClassifier
# 100개, 50개의 노드를 갖는 2개의 은닉 층 정의, 목표 변수의 변수 형은 자동 감지
model = MLPClassifier(hidden_layer_sizes=(128,64), activation='relu', solver='adam',
                      max_iter=500,  validation_fraction=0.15, random_state=123)
modelfit = model.fit(train_X, train_y)

modelfit.score(test_X, test_y) # 평가 데이터 정확도 계산
#  0.9703703703703703

modelfit.__dict__['loss_'] # 선택된 손실
# 0.002682754077655879


np.argmin(np.array(modelfit.__dict__['loss_curve_'])) # 선택된 손실을 주는 반복 수
# 79
# np.array(modelfit.__dict__['loss_curve_'])[79]

# 손실 함수 그래프
fig= plt.figure(figsize=(5,5))
plt.plot(modelfit.__dict__['loss_curve_'])
plt.xlabel('iteration')
plt.ylabel('loss')
plt.title('iteration vs. loss')
fig.savefig(png_path +'/scikit_mlp_loss.png')
plt.show()

#
# 모수 추정값
#

modelfit.__dict__.keys() # 분석 방법마다 키 값이 다르므로 주의!
# dict_keys(['activation', 'solver', 'alpha', 'batch_size', 'learning_rate', 'learning_rate_init',
#            'power_t', 'max_iter', 'loss', 'hidden_layer_sizes', 'shuffle', 'random_state', 'tol',
#            'verbose', 'warm_start', 'momentum', 'nesterovs_momentum', 'early_stopping',
#            'validation_fraction', 'beta_1', 'beta_2', 'epsilon', '_label_binarizer', 'classes_',
#            'n_outputs_', '_random_state', 'n_iter_', 't_', 'n_layers_', 'out_activation_',
#            'coefs_', 'intercepts_', 'loss_curve_', '_no_improvement_count', 'best_loss_', '_optimizer', 'loss_'])

modelfit.__dict__['loss_curve_'] # 반복에 따른 손실 함수 값
modelfit.__dict__['loss_'] # 가장 낮을 손실 함수 값
# 0.002682754077655879

modelfit.__dict__['coefs_'] # 사용된 모수 값 (신경망인 경우는 가중치)

np.array(modelfit.__dict__['coefs_'][0]).shape # 입력 층(64개의 노드)과 첫번째 은닉층(100개의 노드)과의 가중치
# (64, 100)

#
# 비 지도 학습: 주성분 분석 및 시각화
#

# 데이터 분할
bunch = load_digits() # 데이터 불러오기
X, y = bunch.data, bunch.target # 특징 데이터, 목표 데이터

from sklearn.model_selection import train_test_split
train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=123)

# 데이터 표준화
train_X_scaled = train_X - np.mean(train_X, axis=0) # 평균 값만 0 으로 표준화

# 주성분 분석
from sklearn.decomposition import PCA
pca = PCA(random_state=123) # 모델 생성
pcafit = pca.fit(train_X_scaled) # 모델 적합

pcafit.__dict__.keys()
# dict_keys(['n_components', 'copy', 'whiten', 'svd_solver', 'tol', 'iterated_power', 'random_state',
#            'mean_', 'noise_variance_', 'n_samples_', 'n_features_', 'components_', 'n_components_',
#            'explained_variance_', 'explained_variance_ratio_', 'singular_values_'])

pcafit.explained_variance_ratio_ # 분산 설명률
# array([1.46423015e-01, 1.35448133e-01, 1.17006362e-01, 8.37887743e-02,
#        5.95821983e-02, 4.90242938e-02, 4.33704006e-02, 3.62610826e-02,

np.cumsum(pcafit.explained_variance_ratio_) #  누적 분산 설명률
# array([0.14642301, 0.28187115, 0.39887751, 0.48266628, 0.54224848,
#        0.59127278, 0.63464318, 0.67090426, 0.70500975, 0.73681969,

pcafit.singular_values_
# array([4.69280447e+02, 4.51350885e+02, 4.19500541e+02, 3.54993749e+02,
#        2.99354677e+02, 2.71539779e+02, 2.55402124e+02, 2.33532953e+02,


# 한글 출력
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 주성분 분산 설명률 그래프
fig = plt.figure(figsize=(5,5))
plt.plot(pcafit.explained_variance_ratio_, label='분산 설명률')
plt.xlabel('주성분')
plt.ylabel('분산 설명률')
plt.title('주성분과 분산 설명률')
fig.savefig(png_path +'/scikit_pca_variance_ratio.png')
plt.show()

# 주성분 누적 분산 설명률 그래프
fig = plt.figure(figsize=(5,5))
plt.plot(np.cumsum(pcafit.explained_variance_ratio_), label='누적 분산 설명률')
plt.xlabel('주성분')
plt.ylabel('누적 분산 설명률')
plt.title('주성분과 누적 분산 설명률')
fig.savefig(png_path +'/scikit_pca_cusum_variance_ratio.png')
plt.show()


pcafit.transform(train_X_scaled).shape # 주성분 점수 구하기, 64개의 각 주성분별 점수
# (1257, 64)

# 주성분 점수 그래프
fig = plt.figure(figsize=(7,5))
plt.scatter(pcafit.transform(train_X_scaled)[:, 0], pcafit.transform(train_X_scaled)[:,1], c=train_y, s=10) # 제1, 2 주성분
plt.colorbar()
plt.xlabel('주성분1')
plt.ylabel('주성분2')
plt.title('주성분 간 산점도')
fig.savefig(png_path +'/scikit_pca_score.png')
plt.show()

# 참고
first_evec = pcafit.__dict__["components_"][0].reshape((-1,1)) # 첫번째 주성분 벡터
train_X_scaled@first_evec # 첫번째 주성분 벡터에 프로젝션: 주성분 점수, pca.transform(train_X)[:, 0]와 동일

# t-SNE: t-distributed Stochastic Neighbor Embedding
from sklearn.manifold import TSNE
tsne = TSNE(perplexity=30,random_state=123)
tsnefit = tsne.fit(X=train_X_scaled)
tsnefit.__dict__.keys()
# dict_keys(['n_components', 'perplexity', 'early_exaggeration', 'learning_rate', 'n_iter',
#            'n_iter_without_progress', 'min_grad_norm', 'metric', 'init', 'verbose', 'random_state',
#            'method', 'angle', 'n_iter_', 'kl_divergence_', 'embedding_'])
tsneout = tsnefit.fit_transform(X=train_X_scaled) # 2차원
fig = plt.figure(figsize=(7,5))
plt.scatter(tsneout[:, 0], tsneout[:, 1], c=train_y, s=3)
plt.colorbar()
plt.xlabel(r'$y_1$') # 저차원 벡터
plt.ylabel(r'$y_2$') # 저차원 벡터
plt.title('t-SNE plot')
fig.savefig(png_path +'/scikit_tsne_score.png')
plt.show()
