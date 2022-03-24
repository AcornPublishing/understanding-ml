#
# 프로그램 이름: convolution_basics.py
# 작성자: Bong Ju Kang
# 설명: 합성곱 신경망 모델 이해하기
#

# 필요한 패키지
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests, zipfile, io
from keras.models import load_model

from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn.model_selection import GridSearchCV, KFold

from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, InputLayer, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from keras.optimizers import Adam, Optimizer
from keras import backend as K
from keras.utils import to_categorical
from keras.regularizers import l2
from keras.wrappers.scikit_learn import KerasClassifier

from scipy.signal import correlate, convolve2d
from numpy.random import RandomState

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
# 합성곱 신경망 이해를 위한 기본 연산 (교재 수록)
#

#
# 교차 상관 연산과 합성곱 연산
#
# 데이터 구성
image = RandomState(0).choice(np.arange(0, 4), size=25).reshape(5, 5)
# array([[0, 3, 1, 0, 3],
#        [3, 3, 3, 1, 3],
#        [1, 2, 0, 3, 2],
#        [0, 0, 0, 2, 1],
#        [2, 3, 3, 2, 0]])

filter = RandomState(0).choice(np.arange(3), size=9).reshape(-1, 3)
# array([[0, 1, 0],
#        [1, 1, 2],
#        [0, 2, 0]])

# 교차 상관
ccor = correlate(image, filter)
# array([[ 0,  0,  6,  2,  0,  6,  0],
#        [ 0, 12, 11, 10,  9,  9,  3],
#        [ 6, 11, 19,  9, 16, 11,  3],
#        [ 2,  8,  6, 11, 12, 10,  2],
#        [ 0,  5,  8, 10, 11,  5,  1],
#        [ 4,  8, 11, 10,  7,  3,  0],
#        [ 0,  2,  3,  3,  2,  0,  0]])

ccor_valid = correlate(image, filter, 'valid')
# array([[19,  9, 16],
#        [ 6, 11, 12],
#        [ 8, 10, 11]])

stride = 1
filter_size = 3

# 1번째 값 계산
np.sum(image[:3, :3] * filter)
# 19

# 2번째 값 계산
np.sum(image[:filter_size, stride:filter_size + stride] * filter)
# 9

# correlate 계산 방식은 원래 이미지 데이터에 0 값을 행과 열에 2개씩 padding 한 후 계산한 것임
# 아래는 실제로 padding 후 계산한 결과임
padding = filter_size - 1
adj_shape = np.array(image.shape) + padding * 2
image_with_padding = np.zeros(shape=adj_shape)
image_with_padding[padding:image.shape[0] + padding, padding:image.shape[1] + padding] = image
edge = np.zeros(shape=(image_with_padding.shape[0] - padding, image_with_padding.shape[1] - padding))
for i in np.arange(image_with_padding.shape[0] - padding):
    for j in np.arange(image_with_padding.shape[1] - padding):
        edge[i, j] = np.sum(image_with_padding[i:i + padding + 1, j:j + padding + 1] * filter)
# array([[ 0.,  0.,  6.,  2.,  0.,  6.,  0.],
#        [ 0., 12., 11., 10.,  9.,  9.,  3.],
#        [ 6., 11., 19.,  9., 16., 11.,  3.],
#        [ 2.,  8.,  6., 11., 12., 10.,  2.],
#        [ 0.,  5.,  8., 10., 11.,  5.,  1.],
#        [ 4.,  8., 11., 10.,  7.,  3.,  0.],
#        [ 0.,  2.,  3.,  3.,  2.,  0.,  0.]])

# 합성곱 연산
convolve2d(image, filter)
# array([[ 0,  0,  3,  1,  0,  3,  0],
#        [ 0,  6,  7, 10,  6,  6,  6],
#        [ 3,  7, 20, 12, 13, 13,  6],
#        [ 1,  9, 10, 13,  9, 15,  4],
#        [ 0,  4,  7,  5, 11,  9,  2],
#        [ 2,  5, 10, 11, 12,  6,  0],
#        [ 0,  4,  6,  6,  4,  0,  0]])

# 행으로 한번, 열로 한번: 0축으로 한번, 1축으로 한번
filter_180 = np.rot90(filter, k=2)
# filter180 = np.rot90(np.rot90(filter))
filter_row_90 = np.rot90(filter)

np.flip(np.flip(filter, axis=0), axis=1)

convolve2d(image, filter_180)
convolve2d(filter_180, image)
# array([[ 0,  0,  6,  2,  0,  6,  0],
#        [ 0, 12, 11, 10,  9,  9,  3],
#        [ 6, 11, 19,  9, 16, 11,  3],
#        [ 2,  8,  6, 11, 12, 10,  2],
#        [ 0,  5,  8, 10, 11,  5,  1],
#        [ 4,  8, 11, 10,  7,  3,  0],
#        [ 0,  2,  3,  3,  2,  0,  0]])

fpass_z = convolve2d(image, filter_180, 'same')
bias = 0.1
fpass_z = fpass_z + bias
act_z = np.where(fpass_z >= 0, fpass_z, 0)
# array([[12, 11, 10,  9,  9],
#        [11, 19,  9, 16, 11],
#        [ 8,  6, 11, 12, 10],
#        [ 5,  8, 10, 11,  5],
#        [ 8, 11, 10,  7,  3]])

i = padding + 1
j = padding + 1
np.sum(image_with_padding[i:i + padding + 1, j:j + padding + 1] * filter_180.T)

test_image = image[:3, :3]
np.sum(test_image * filter_180[::-1, ::-1])
np.rot90(filter_180, 2)

#
# 모수의 추정: 전진패스와 후진패스의 예
#
image_matrix = RandomState(2).choice(np.arange(0, 3), size=9).reshape(3, 3)
# array([[0, 3, 1, 0, 3],
#        [3, 3, 3, 1, 3],
#        [1, 2, 0, 3, 2],
#        [0, 0, 0, 2, 1],
#        [2, 3, 3, 2, 0]])


filter = RandomState(0).choice(np.arange(3), size=4).reshape(-1, 2)
# array([[0, 1, 0],
#        [1, 1, 2],
#        [0, 2, 0]])

# 전진 패스
filter_flip = np.rot90(filter, k=2)
fpass_z = convolve2d(image_matrix, filter_flip, 'valid') + 0.1
act_z = np.where(fpass_z >= 0, fpass_z, 0)

# 후진 패스
delta_matrix = RandomState(0).randn(2, 2)
delta_matrix_flip = np.rot90(delta_matrix, k=2)
weight_grad = convolve2d(image_matrix, delta_matrix_flip, 'valid')

#
# 예제: [DIGITS] 데이터 적용 (다층 신경망)
#

# 데이터 구성
bunch = load_digits()
dir(bunch)
# ['DESCR', 'data', 'images', 'target', 'target_names']

# Data Set Characteristics:
#     :Number of Instances: 5620
#     :Number of Attributes: 64
#     :Attribute Information: 8x8 image of integer pixels in the range 0..16.
#     :Missing Attribute Values: None
#     :Creator: E. Alpaydin (alpaydin '@' boun.edu.tr)
#     :Date: July; 1998
#
# This is a copy of the test set of the UCI ML hand-written digits datasets
# http://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits

# 이미지 파일로 데이터 구성 형식 이해하기
plt.figure(figsize=(5, 5))
plt.imshow(bunch['images'][0], cmap='binary')
plt.grid()
plt.savefig(png_path + '/convnet_data_digits_image.png')
plt.show()

# 입력 특징 구성
X = bunch['data']
X.shape
# (1797, 64)

# 목표 변수
y = bunch['target']
y.shape
# (1797,)

# 데이터 전 처리 및 분할
max_value = np.max(X)
X_scaled = X / max_value
y_onehot = to_categorical(y)
y_onehot.shape
# (1797, 10)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_onehot, test_size=0.3, random_state=123)

#
# 모델 구성
#
# 입력 차원 정의
input_dims = X_train.shape[1]

# 아키텍처 정의
model = Sequential()
model.add(InputLayer(input_shape=(input_dims,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()

# 모델 정의를 위한 추가 변수 정의
model.compile(optimizer=Adam(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])

# 모델 적합
hist = model.fit(X_train, y_train, validation_split=0.1, batch_size=10, epochs=100)

# 모델 평가
scores = model.evaluate(X_test, y_test, batch_size=10)
print('손실함수값=', scores[0], '\n정확도=', scores[1])
# 손실함수값= 0.12081231161361058
# 정확도= 0.9833333315672698

#
# 초 모수 결정 (은닉 층 개수와 학습률 조정)
#

# 병렬 처리를 위하여 모델은 외부에 저장
# 같은 프로젝트 디렉토리에 해당 파일을 저장한 경우에
# 1) sys.path를 실행하여 해당 파일의 위치가 경로에 있는 지 확인
# 2) 없으면, 가령, 프로젝트가 여러개인 경우에는
# 파이참의 File > Settings > Project > Project Dependenceis... 에 가서 각 프로젝트의 체크 박스를 활성화

# 해당 파일 호출
try:
    import convolution_basics_defs as defs
except:
    import os, sys

    curr_path = os.path.abspath("./ch20-합성곱 신경망")
    if curr_path not in sys.path:
        sys.path.append(curr_path)
    import convolution_basics_defs as defs

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
# 예제: [DIGITS] 데이터 적용 (합성곱 신경망)
#

# 데이터 구성
bunch = load_digits()

# 입력 특징
X = bunch['data']
X.shape
# (1797, 64)

# 목표 변수
y = bunch['target']
y.shape
# (1797,)

# 데이터 전 처리 및 분할
max_value = np.max(X)
X_scaled = X / max_value
y_onehot = to_categorical(y)

# 입력 차원 정의: 이미지 형식으로 적용 (높이, 폭, 채널(깊이))
X_conv = X_scaled.reshape(-1, 8, 8, 1)
y_conv = y_onehot

# 데이터 분할
X_conv_train, X_conv_test, y_conv_train, y_conv_test = train_test_split(X_conv, y_conv,
                                                                        test_size=0.3, random_state=1234)

#
# 모델 구성
#
# 입력 차원 지정
input_shape = X_conv_train.shape[1:]

# 아키텍처 정의
model = Sequential()
model.add(InputLayer(input_shape=input_shape))
model.add(Conv2D(64, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
# model.add(BatchNormalization())
# model.add(Dropout(0.1))
model.add(Dense(10, activation='softmax'))
model.summary()

# 모델 정의를 위한 추가 변수 정의
model.compile(optimizer=Adam(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])

# 모델 적합
hist = model.fit(X_conv_train, y_conv_train, validation_split=0.1, batch_size=10, epochs=100, verbose=2)

# model.save(data_path+'/convmodel.h5')
# saved_model = load_model(data_path+'/convmodel.h5')

hist.history.keys()
plt.plot(hist.history['val_loss'])

scores = model.evaluate(X_conv_test, y_conv_test, batch_size=10)
# scores = saved_model.evaluate(X_conv_test, y_conv_test, batch_size=10)
print('손실함수값=', scores[0], '\n정확도=', scores[1])
# 손실함수값= 0.04426065459343465
# 정확도= 0.9870370339464258
