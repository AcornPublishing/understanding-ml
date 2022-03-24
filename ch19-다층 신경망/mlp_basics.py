#
# 프로그램 이름: mlp_basics.py
# 작성자: Bong Ju Kang
# 설명: 다층 신경망 모델 이해하기
#

# 필요한 패키지
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests, zipfile, io

from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn.model_selection import GridSearchCV, KFold
from sklearn.svm import SVC
from sklearn.datasets.samples_generator import make_blobs

from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, InputLayer, Dropout
from keras.optimizers import Adam, Optimizer
from keras import backend as K
from keras.utils import to_categorical
from keras.regularizers import l2
from keras.wrappers.scikit_learn import KerasClassifier

# 3차원 그래프
from mpl_toolkits.mplot3d import Axes3D

# 초기 설정
png_path = "./data/png"
os.makedirs(png_path, exist_ok=True)

# 한글출력
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# GPU 메모리를 독점적으로 사용하지 말고 공유하여 사용하도록 설정
config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
config.gpu_options.allow_growth = True
K.set_session(tf.Session(config=config))

#
# 본문 수록 신경망 예시(전진 패스, 후진 패스)
#

#
# 예시를 위한 사전 값 정의
#
# 데이터, 가중치, 편의 정의
alpha = 0.1

# 하나의 예(입력 데이터)는 하나의 컬럼으로 정의
x = np.array([1., 2.]).reshape(-1, 1)
w_2 = np.array([[0.1, 0.2],
               [0.1, 0.1]])
b_2 = np.array([1., 1.]).reshape(-1,1)

# 하나의 행이 다음 층의 하나의 노드가 받는 가중치
w_3 = np.array([0.1, 0.1]).reshape(1, -1)
b_3 = np.array([1.]).reshape(-1,1)
y = np.array([0.5]).reshape(-1,1)

#
# 전진 패스
#
# 2 층: 1차 은닉층
z_2 = w_2 @ x + b_2
print(z_2)
# [[1.5]
#  [1.3]]

def sigmoid(z):
    return 1/(1+np.exp(-z))

a_2 = sigmoid((z_2))
print(a_2)
# [[0.81757448]
#  [0.78583498]]

# 3 층
z_3 = w_3 @ a_2 + b_3
print(z_3)
# [[1.16034095]]
a_3 = z_3
print(a_3)
# [[1.16034095]]

# 손실 계산
def loss_mse(y_true, act_value):
    return 0.5*np.mean((y_true-act_value)**2)

loss = loss_mse(y, a_3)
print('loss={:0.4f}'.format(loss))
# loss=0.2180

#
# 후진 패스
#

def sigmoid_pdiff(z):
    return sigmoid(z)*(1-sigmoid(z))

# linear
delta_3 = 1.0 * (-(y-a_3))
print(delta_3)
# [[0.66034095]]

b_3_grad = delta_3
print(b_3_grad)
# [[0.66034095]]

w_3_grad = delta_3 @ a_2.T
print(w_3_grad)
# array([[0.5398779 , 0.51891902]])

delta_2 = sigmoid_pdiff(z_2) * (w_3.T @ delta_3)
print(delta_2)
# [[0.00984875]
#  [0.01111343]]
b_2_grad = delta_2
print(b_2_grad)
# [[0.00984875]
#  [0.01111343]]

w_2_grad = delta_2 @ x.T
print(w_2_grad)
# [[0.00984875 0.0196975 ]
#  [0.01111343 0.02222686]]

# 모수 갱신
b_3_new = b_3 - alpha*b_3_grad
# array([[0.93396591]])
w_3_new = w_3 - alpha*w_3_grad
# array([[0.04601221, 0.0481081 ]])
b_2_new = b_2 - alpha*b_2_grad
# array([[0.99901512],
#        [0.99888866]])
w_2_new = w_2 - alpha*w_2_grad
# array([[0.09901512, 0.19803025],
#        [0.09888866, 0.09777731]])


#
# 예제: [DIGITS] 데이터
#

# 데이터 구성
bunch = load_digits()
dir(bunch)
# ['DESCR', 'data', 'images', 'target', 'target_names']

# 이미지 파일로 데이터 구성 형식 이해하기
plt.figure(figsize=(5,5))
plt.imshow(bunch['images'][0], cmap='gray')
plt.savefig(png_path + '/mlp_data_digits_image.png')
plt.show()

X = bunch['data']
y = bunch['target']
X.shape
# (1797, 64)
y.shape
# (1797,)

# 데이터 전 처리 및 분할
max_value = np.max(X)
X_scaled = X/max_value
y_onehot = to_categorical(y)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_onehot, test_size=0.3)

# 모델 구성
# 입력 차원 정의
input_dims = X_train.shape[1]

# 아키텍처 정의
model=Sequential()
model.add(InputLayer(input_shape=(input_dims,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()

# 모델 정의를 위한 추가 변수 정의
model.compile(optimizer=Adam(lr=0.1), loss='categorical_crossentropy', metrics=['accuracy'])

# 모델 적합
hist = model.fit(X_train, y_train, validation_split=0.1, batch_size=50, epochs=200)

# 모델 평가
scores = model.evaluate(X_test, y_test, batch_size=100)
model.metrics_names
# ['loss', 'acc']
print('손실함수값=', scores[0], '\n정확도=', scores[1])
# 손실함수값= 0.22069841531036352
# 정확도= 0.9703703655136956

#
# 초 모수 결정 (은닉 층 개수와 학습률 조정)
#

# 병렬 처리를 위하여 모델은 외부에 저장
try:
    import mlp_basics_defs as defs
except:
    import os, sys
    curr_path = os.path.abspath("./ch19-다층 신경망")
    if curr_path not in sys.path:
        sys.path.append(curr_path)
    import mlp_basics_defs as defs

# 모델 등록
model = KerasClassifier(build_fn=defs.grid_base_model, verbose=0, epochs=200)

param_grid = dict(learning_rate=[0.01, 0.1],
                  batch_size=[10, 50])

# 초 모수 값의 조합에 의한 모델 적합
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
grid_search.fit(X_train, y_train)

# 선택된 초 모수 값
print(grid_search.best_params_)
# {'batch_size': 10, 'learning_rate': 0.01}

#
# 예제: [BANK] 데이터 적용
#

#
# 데이터 가져오기
#
# 데이터 경로
path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00222/'
zip_url = path + 'bank.zip'

# 짚 파일 풀기
z = zipfile.ZipFile(io.BytesIO(requests.get(zip_url).content))
# 짚 파일 내의 구성 요소 보기
z.infolist()
# 특정 파일 가져오기
df = pd.read_csv(z.open('bank.csv'), sep=';')
df.shape
# (4521, 17)

#
# 데이터 전 처리
#

# 목표 변수 분포 확인
df.y.value_counts()
# no     4000
# yes     521
base_dist = df.y.value_counts() / df.shape[0]
# no     0.88476
# yes    0.11524

# 변수 정의
feature_list = [name for name in df.columns if name != 'y']
categorical_variables = df.columns[(df.dtypes == 'object') & (df.columns != 'y')]
num_variables = [name for name in feature_list if name not in categorical_variables]

# 범주형 데이터를 숫자형 데이터로 전환
df_X_onehot = pd.get_dummies(df[categorical_variables], prefix_sep='_')
df_y_onehot = pd.get_dummies(df['y'], drop_first=True)


# 범주형 데이터와 숫자형 데이터 결합
X = np.c_[df[num_variables].values, df_X_onehot.values].astype(np.float64)
y = df_y_onehot.values.ravel()

# 모든 특징의 이름 리스트
feature_names = num_variables + df_X_onehot.columns.tolist()

# 데이터 표준화
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=123)

#
# 초 모수 결정: 학습률, 드롭아웃 여부 그리고 배치 크기 결정
#

# 그리드 기초 모델 정의
def grid_base_model(learning_rate=0.01, dropout_flag=True):
    input_dims = 51
    model = Sequential()
    model.add(InputLayer(input_shape=(input_dims,), name='input'))

    # 과적합 방지를 위한 가중치와 편의에 대한 벌점 항 추가
    model.add(Dense(100, activation='relu',
                    kernel_regularizer=l2(0.01),
                    bias_regularizer=l2(0.01),
                    name='hidden-1'))

    # 과적합 방지를 위한 dropout 추가
    if dropout_flag :
        model.add(Dropout(0.1, name='hidden-1-drop'))
    model.add(Dense(1, activation='sigmoid', name='output'))
    model.summary()

    # 최적화 방법, 손실, 평가 방법 등을 정의
    optimizer = Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 모델 구성
model = KerasClassifier(build_fn=grid_base_model, epochs=50, verbose=0)

# 초 모수 그리드 정의
param_grid = dict(learning_rate=[0.01, 0.1], dropout_flag = [True, False], batch_size=[50, 100])
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, return_train_score=True)


# 초 모수 그리드 적합
grid_search.fit(X_train, y_train)

# 초 모수 결정
grid_search.best_params_
# {'batch_size': 100, 'dropout_flag': True, 'learning_rate': 0.01}

# 초 모수 결정을 위한 근거
grid_search.cv_results_

#
# 모델 적합 및 평가
#

# 초 모수 정의
# {'batch_size': 100, 'dropout_flag': True, 'learning_rate': 0.01}

# 모델 적합
input_dims =X_train.shape[1]
model = Sequential()
model.add(InputLayer(input_shape=(input_dims,), name='input'))

# 과적합 방지를 위한 가중치와 편의에 대한 벌점 항 추가
model.add(Dense(100, activation='relu',
                kernel_regularizer=l2(0.01),
                bias_regularizer=l2(0.01),
                name='hidden-1'))
# 과적합 방지를 dropout 추가
model.add(Dropout(0.1, name='hidden-1-drop'))
model.add(Dense(1, activation='sigmoid', name='output'))
model.summary()

# 최적화 방법, 손실, 평가 방법 등을 정의
optimizer = Adam(lr=0.01)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# 모델 적합
hist = model.fit(X_train, y_train, validation_split=0.1, batch_size=100, epochs=200)

# 모델 평가
scores = model.evaluate(X_test, y_test, batch_size=100)
model.metrics_names
# ['loss', 'acc']
print('손실함수값=', scores[0], '\n정확도=', scores[1])
# 손실함수값= 0.32632701504116296
# 정확도= 0.8909358897890866

# 층 순서대로 가중치와 절편 값임. 하나의 층의 각 가중치는 행이고 이포크마다 갱신되는 값이다.
model.weights
# [<tf.Variable 'hidden-1/kernel:0' shape=(51, 100) dtype=float32_ref>,
#  <tf.Variable 'hidden-1/bias:0' shape=(100,) dtype=float32_ref>,
#  <tf.Variable 'output/kernel:0' shape=(100, 1) dtype=float32_ref>,
#  <tf.Variable 'output/bias:0' shape=(1,) dtype=float32_ref>]

# 입력 층의 1번째 변수에 대한 1번째 이포크 시의 가중값
model.get_weights()[0][0,0]
# 입력 층의 1번째 변수에 대한 2번째 이포크 시의 가중값
model.get_weights()[0][0,1]

# 각 가중치 값의 변화
plt.plot(model.get_weights()[0][0,:])
plt.plot(model.get_weights()[0][1,:])
plt.plot(model.get_weights()[0][2,:])

# 손실 값의 변화를 살펴봄
hist.history.keys()
# dict_keys(['val_loss', 'val_acc', 'loss', 'acc'])

# 손실 및 정확도 그래프
# 최소 손실을 주는 이포크
opt_epoch = np.argmin(hist.history['val_loss'])+1
min_value = np.min(hist.history['val_loss'])
max_acc = hist.history['val_acc'][opt_epoch]

fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(1, 2,1)
ax1 = plt.plot(hist.history['loss'], label='훈련 데이터 손실')
ax1 = plt.legend()
ax2 = fig.add_subplot(1, 2, 2)
ax2 = plt.plot(hist.history['val_loss'], c='orange', label='검증 데이터 손실')
ax2 = plt.annotate("이포크 수: %d" % opt_epoch, xy=(opt_epoch, min_value),
             xytext=(35, 35), textcoords='offset points',
             arrowprops={'arrowstyle': '->', 'color': 'r', 'lw': 1})
ax2 = plt.legend()
fig.suptitle('훈련 데이터 대 검증 데이터 손실', weight='bold')
plt.savefig(png_path + '/mlp_loss_train_validate.png')
plt.show()

#
# 모델 평가: ROC 그래프
#
# 모델에 의한 예측 확률 계산
y_pred_proba = model.predict_proba(X_test)

# fpr: 1-특이도, tpr: 민감도, auc 계산
fpr, tpr, _ = metrics.roc_curve(y_true=y_test, y_score=y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)

# ROC 그래프 출력
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label="신경망\n곡선밑 면적(AUC)=" + "%.4f" % auc)
plt.plot([-0.02, 1.02], [-0.02, 1.02], color='gray', linestyle=':', label='무작위 모델')
plt.margins(0)
plt.legend(loc=4)
plt.xlabel('fpr: 1-Specificity')
plt.ylabel('tpr: Sensitivity')
plt.title("ROC Curve", weight='bold')
plt.legend()
plt.savefig(png_path + '/mlp_ROC.png')
plt.show()