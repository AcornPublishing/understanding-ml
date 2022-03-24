#
# 프로그램 이름: keras_basics.py
# 작성자: Bong Ju Kang
# 설명: keras의 기본을 예제와 함께 이해하기
#

# 필요한 패키지
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import tensorflow as tf

#
# 데이터 불러오기
#
# 손글씨 데이터 불러오기 (8x8)
load_digits().keys()
# dict_keys(['data', 'target', 'target_names', 'images', 'DESCR'])

load_digits().data.shape  # 모양 확인
# (1797, 64)

load_digits().target.shape  # 모양 확인
# (1797,)

# 데이터 분할
bunch = load_digits()  # 데이터 불러오기
X, y = bunch.data, bunch.target  # 특징 데이터, 목표 데이터
train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=123)

# 목표 데이터에 대한 가변수 생성 (one-hot encoding)
num_classes = 10
train_y = keras.utils.to_categorical(train_y, num_classes)
test_y = keras.utils.to_categorical(test_y, num_classes)

# 가변수 생성의 의미
keras.utils.to_categorical(np.array([0,1,2]), 3)
# array([[1., 0., 0.],
#        [0., 1., 0.],
#        [0., 0., 1.]], dtype=float32)
keras.utils.to_categorical(np.array([0,1,3]), 4)
# array([[1., 0., 0., 0.],
#        [0., 1., 0., 0.],
#        [0., 0., 0., 1.]], dtype=float32)


#
# keras MLP 적용
#
keras.backend.clear_session() # 세션 초기화

# 똑 같은 결과를 가져오기 위하여 1개의 쓰레드 사용
tf_config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
keras.backend.set_session(tf.Session(config=tf_config))
np.random.seed(101)
tf.set_random_seed(101)

# 신경망 모델 구성
input_shape = (train_X.shape[1],)
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=input_shape))
model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss="categorical_crossentropy",
              optimizer='adam', metrics=['accuracy'])
model.summary()
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# dense_1 (Dense)              (None, 128)               8320
# _________________________________________________________________
# dense_2 (Dense)              (None, 64)                8256
# _________________________________________________________________
# dense_3 (Dense)              (None, 10)                650
# =================================================================
# Total params: 17,226
# Trainable params: 17,226
# Non-trainable params: 0
# _________________________________________________________________


# 신경망 모델 적합
model.fit(train_X, train_y,
          batch_size=100,
          epochs=500,
          verbose=1,
          validation_split=0.15)

# 신경망 모델 평가
score = model.evaluate(test_X, test_y, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
# Test loss: 0.1712413256188113
# Test accuracy: 0.9703703703703703

#
# 합성곱 신경망 적합(CNN)
#
keras.backend.clear_session()

# 재현성을 위한 코드
keras.backend.set_session(tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)))
np.random.seed(101)
tf.set_random_seed(101)

# 데이터 재 정렬 (배치, 행(높이), 열(폭), 깊이(채널))
train_X = train_X.reshape(-1, 8, 8, 1)
test_X = test_X.reshape(-1, 8, 8, 1)

# 모델 구성
input_shape = train_X.shape[1:]
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape)) # 3x3 커널이 32개
model.add(Conv2D(64, (3, 3), activation='relu')) # 3x3 커널이 64개
model.add(MaxPooling2D(pool_size=(2, 2))) # 2x2 커널을 이용한 풀링
model.add(Dropout(0.25)) # 25% dropout
model.add(Flatten()) # 평탄화
model.add(Dense(128, activation='relu')) # 포화연결층
model.add(Dropout(0.25)) # 25% dropout
model.add(Dense(num_classes, activation='softmax')) # 출력층

model.compile(loss="categorical_crossentropy",
              optimizer='adam',
              metrics=['accuracy'])
model.summary()
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# conv2d_1 (Conv2D)            (None, 6, 6, 32)          320
# _________________________________________________________________
# conv2d_2 (Conv2D)            (None, 4, 4, 64)          18496
# _________________________________________________________________
# max_pooling2d_1 (MaxPooling2 (None, 2, 2, 64)          0
# _________________________________________________________________
# dropout_1 (Dropout)          (None, 2, 2, 64)          0
# _________________________________________________________________
# flatten_1 (Flatten)          (None, 256)               0
# _________________________________________________________________
# dense_1 (Dense)              (None, 128)               32896
# _________________________________________________________________
# dropout_2 (Dropout)          (None, 128)               0
# _________________________________________________________________
# dense_2 (Dense)              (None, 10)                1290
# =================================================================
# Total params: 53,002
# Trainable params: 53,002
# Non-trainable params: 0
# _________________________________________________________________

model.fit(train_X, train_y,
          batch_size=100,
          epochs=200,
          verbose=1,
          validation_split=0.15)
score = model.evaluate(test_X, test_y, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
# Test loss: 0.05672662238918107
# Test accuracy: 0.9851851851851852
