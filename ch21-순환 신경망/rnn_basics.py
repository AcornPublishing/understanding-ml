#
# 프로그램 이름: rnn_basics.py
# 작성자: Bong Ju Kang
# 설명: 순환 신경망 모델 이해하기
#

# 필요한 패키지
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.utils import to_categorical
from numpy.core.multiarray import ndarray
from numpy.random import RandomState


# 초기 설정
png_path = "./data/png"
os.makedirs(png_path, exist_ok=True)

# 한글출력
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

#
# 모수의 추정: 전진패스, 후진패스의 예시
#

#
# 데이터 구성
#
data = "hello"
chars = ['h', 'e', 'l', 'o']
data_size, vocab_size = len(data), len(chars)

# 문자를 숫자로
char_to_idx = {ch: i for i, ch in enumerate(chars)}

# 마지막 문자 데이터 생략하기 위하여
x_str = data[:-1]

# 처음 문자 데이터를 생략하기 위하여
y_str = data[1:]

# 숫자형 데이터 구성
x_train = [char_to_idx[c] for c in x_str]
y_train = [char_to_idx[c] for c in y_str]

# 문자마다 가변수 만들기(one-hot)
x_train = to_categorical(x_train, num_classes=vocab_size)
y_train = to_categorical(y_train, num_classes=vocab_size)

# 입력 모양 만들기 (batch, seq_length, features)
x_train = x_train.reshape(-1, len(x_train), vocab_size)
y_train = y_train.reshape(-1, len(y_train), vocab_size)

print(x_train)
# [[[1. 0. 0. 0.]
#   [0. 1. 0. 0.]
#   [0. 0. 1. 0.]
#   [0. 0. 1. 0.]]]

#
# 전진 패스
#

# 아키텍처 초 모수 정의
hidden_units = 3
input_units = 4
seq_length = 4

# 하나의 입력도 열벡터, 출력도 열벡터로 전환
y_true = y_train[0].T
input = x_train[0].T

# 모수 초기값 정의
Wx = RandomState(0).randn(3,4)
Wh = RandomState(0).randn(3,3)
Wy = RandomState(0).randn(4,3)
bh = np.full(shape=(3,1), fill_value=RandomState(0).rand())
by = np.full(shape=(4,1), fill_value=RandomState(1).rand())

# 출력층 값 계산을 위한 소프트맥스 함수 정의
def softmax(x):
    exps = np.exp(x)
    return exps / np.sum(exps)

# 은닉층의 상태값 초기화
h = np.zeros(shape=(hidden_units, seq_length+1))

# 출력층의 출력값 초기화
y_pred_prob = np.zeros(shape=(input_units, seq_length))

#
# 전진 패스
#
for j in np.arange(seq_length):
    h[:, [j]] = np.tanh(Wx @ input[:, [j]] + Wh @ h[:, [j-1]] + bh)
    y_pred_prob[:, [j]] = softmax(Wy @ h[:, [j]] + by)

# 출력층 값을 이용한 예측
x_index = np.argmax(x_train[0], axis=1)
x_str = [chars[i] for i in x_index]
y_index = np.argmax(y_pred_prob, axis=0)
y_str = [chars[i] for i in y_index]

# 입력값과 예측값 출력
print(x_index, ''.join(x_str), "---> ",
      y_index, ''.join(y_str))
# [0 1 2 2] hell --->  [1 1 1 1] eeee

#
# 후진 패스
#

# 각 가중치(편의 포함)의 편미분 값을 받는 행렬 초기화
dWx, dWh, dWy = np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(Wy)
dbh, dby = np.zeros_like(bh), np.zeros_like(by)

# 은닉층의 오차를 역으로 전파하기 위한 변수
dh_next = np.zeros((hidden_units, 1))

# 후진 패스
# 시퀀스의 역으로 진행
for j in reversed(np.arange(seq_length)):
    # 출력층 미분 (교차엔트로피)
    dy = - y_true[:, [j]] / y_pred_prob[:, [j]]

    # 출력층 입력값 미분 (소프트맥스)
    dsm = np.diag(y_pred_prob[:, [j]].ravel())- (y_pred_prob[:, [j]]@y_pred_prob[:, [j]].T)

    # 출력층의 오차
    dz = dsm @ dy
    # dz = y_pred_prob[:, [j]] - y_true[:, [j]]

    # 출력층의 그래디언트
    dWy += dz @ h[:, [j]].T
    dby += dz

    # 은닉층의 오차: 출력층과 시퀀스의 다음 시간의 오차의 합
    dh_act = Wy.T @ dz + dh_next
    dh = (1 - h[:, [j]] ** 2) * dh_act  # tanh  미분 값

    # 은닉층의 그래디언트
    dWx += dh @ input[:, [j]].T
    dWh += dh @ h[:, [j - 1]].T
    dbh += dh
    dh_next = Wh.T @ dh

# 각 그래디언트의 평균
dWx /= seq_length
dWh /= seq_length
dbh /= seq_length
dWy /= seq_length
dby /= seq_length

# 후진 패스 결과
print('\n###### gradients ######')
print('--- dWx ---\n', dWx)
print('--- dWh ---\n', dWh)
print('--- dbh ---\n', dbh)
print('--- dWy ---\n', dWy)
print('--- dby ---\n', dby)

# --- dWx ---
#  [[-0.0020301   0.00082249  0.00021054  0.        ]
#  [-0.00389519  0.00189845  0.00017362  0.        ]
#  [ 0.12560695  0.0072327  -0.06291784  0.        ]]
# --- dWh ---
#  [[ 0.00101693  0.00101981  0.00053308]
#  [ 0.00203502  0.0020417   0.00095297]
#  [-0.05583219 -0.0558249  -0.05200547]]
# --- dbh ---
#  [[-0.00099707]
#  [-0.00182312]
#  [ 0.06992181]]
# --- dWy ---
#  [[ 0.34395243  0.34402328  0.28400027]
#  [ 0.26577406  0.26533651  0.27711726]
#  [-0.46244094 -0.46206665 -0.42580961]
#  [-0.14728554 -0.14729315 -0.13530792]]
# --- dby ---
#  [[ 0.34528129]
#  [ 0.26447028]
#  [-0.46275877]
#  [-0.1469928 ]]

#
# LSTM 예시
#
# 참조 문헌: https://github.com/keras-team/keras/blob/master/examples/addition_rnn.py

from keras.layers import LSTM, RepeatVector, TimeDistributed, Dense
from keras.models import Sequential
import numpy as np
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

#
# 데이터 구성
#

DIGITS = 2
N_SAMPLES = 5000
np.random.seed(1234)

#
# 데이터 생성
#
# 예 [('64+78', '142'), ('78+58', '136'), ('84+29', '113'), ('26+13', '39 '), ('12+16', '28 ')]
def make_addition_dataset():
    x_dict = dict()
    while len(x_dict) < N_SAMPLES:
        a = np.random.randint(1, np.power(10, DIGITS))
        b = np.random.randint(1, np.power(10, DIGITS))
        key = ("%d+%d" % (a, b)).rjust(DIGITS + 1 + DIGITS)  # 오른쪽 정렬
        value = str(a + b).ljust(1 + DIGITS)  # 왼쪽 정렬
        x_dict[key] = value
    return list(x_dict.items())

rawdata = make_addition_dataset()

# 데이터 확인
print(rawdata[:5])
# [('48+84', '132'), ('39+54', '93 '), ('77+25', '102'), ('16+50', '66 '), ('24+27', '51 ')]

# 입력값과 출력값으로 분리: 결과는 tuple
x_rawdata, y_rawdata = zip(*rawdata)

# 문자별 인덱스 생성
chars = list('0123456789+ ')
char_to_idx = {ch: i for i, ch in enumerate(chars)}
vocab_size = len(chars)
data_size = len(x_rawdata)
print('data_size =', data_size)
print('vocab_size =', vocab_size)
# data_size = 5000
# vocab_size = 12

# 문자열을 인덱스 벡터로 변환
def conv_vector(rawdata):
    # 48 + 84 -> [4, 8, 10, 8, 4]
    list = []
    for data in rawdata:
        list.append([char_to_idx[c] for c in data])
    return list

x_data_index = conv_vector(x_rawdata)
y_data_index = conv_vector(y_rawdata)
print(x_rawdata[0], '->', x_data_index[0])
print(y_rawdata[0], '->', y_data_index[0])
# 48+84 -> [4, 8, 10, 8, 4]
# 132 -> [1, 3, 2]

# one-hot 인코딩
x_data = np_utils.to_categorical(x_data_index, num_classes=vocab_size)
y_data = np_utils.to_categorical(y_data_index, num_classes=vocab_size)
print(x_data[0])
# [[0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
#  [0. 0. 0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
#  [0. 0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]]
print(y_data[0])
# [[0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]]
print(x_data.shape)
# (5000, 5, 12)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3)

#
# 모델 구성 및 적합
#

# 초 모수 정의
batch_size = 50
seq_length = X_train.shape[1]
cell_units = 100  # 은닉층 노드의 개수
out_length = y_train.shape[1]

# 모델 생성
model = Sequential()
model.add(LSTM(cell_units, input_shape=(seq_length, vocab_size)))
model.add(RepeatVector(out_length))
model.add(LSTM(cell_units, return_sequences=True))
model.add(TimeDistributed(Dense(vocab_size, activation='softmax')))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# lstm_16 (LSTM)               (None, 100)               45200
# _________________________________________________________________
# repeat_vector_3 (RepeatVecto (None, 3, 100)            0
# _________________________________________________________________
# lstm_17 (LSTM)               (None, 3, 100)            80400
# _________________________________________________________________
# time_distributed_12 (TimeDis (None, 3, 12)             1212
# =================================================================
# Total params: 126,812
# Trainable params: 126,812

# 훈련
hist = model.fit(X_train, y_train, batch_size=batch_size, epochs=100, validation_split=0.1)

# 평가
model.evaluate(X_test, y_test, batch_size=batch_size)
# [0.13061178972323736, 0.9733333388964335]

# 예측
y_pred = model.predict_classes(X_test[:5])
x_test = np.argmax(X_test[:5], axis=-1)

# 예측된 결과를 문자로 전환 또는 숫자를 문자로 전환
def conv_string(rawdata):
    list = []
    for data in rawdata:
        list.append(''.join([chars[c] for c in data]))
    return list

# 예측된 결과를 확인
for i in range(5):
    x = np.argmax(X_test[[i]], axis=-1)
    print(conv_string(x), '->', conv_string(y_pred[[i]]))
# ['94+21'] -> ['115']
# [' 93+8'] -> ['101']
# ['55+47'] -> ['102']
# ['42+76'] -> ['118']
# [' 6+73'] -> ['79 ']






