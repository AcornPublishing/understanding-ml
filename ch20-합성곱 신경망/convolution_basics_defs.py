#
# 프로그램 이름: mlp_basics_defs.py
# 작성자: Bong Ju Kang
# 설명: 신경망을 병렬로 처리하기
#

import tensorflow as tf
import keras.backend as K
from keras import Sequential
from keras.engine import InputLayer
from keras.layers import Dense
from keras.optimizers import Adam

# CPU만 사용하도록
# K.set_session(tf.Session(config=tf.ConfigProto(device_count={'GPU': 0})))

# GPU 메모리를 독점적으로 사용하지 말고 공유하여 사용하도록 설정
config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
config.gpu_options.allow_growth = True
K.set_session(tf.Session(config=config))


# 모델 정의
def grid_base_model(learning_rate=0.1):
    input_dims = 64
    model = Sequential()
    model.add(InputLayer(input_shape=(input_dims,)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    optimizer = Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
