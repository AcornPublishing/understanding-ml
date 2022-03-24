#
# 프로그램 이름: svd_basics.py
# 작성자: Bong Ju Kang
# 설명: 특이값 분해 이해하기
#

# 필요한 패키지
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from numpy.random import RandomState
from numpy.linalg import svd, matrix_rank
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import sparse_random_matrix
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA

# 초기 설정
png_path = "./data/png"
os.makedirs(png_path, exist_ok=True)

# 한글출력
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

#
# 특이값 분해 계산
#
input = RandomState(0).randint(0, 9, 4*3).reshape((4,3))
print(input)
# [[5 0 3]
#  [3 7 3]
#  [5 2 4]
#  [7 6 8]]

# compact SVD
# 여기서 주의할 점은 리턴되는 값 중 v는 v의 전치행렬을 의미한다.
u, d, v = svd(input, full_matrices=False)


# 원래 행렬 복원
u@np.diag(d)@v

# 차원 축소 및 축소된 행렬의 프로베니어스 노름
reduced = input.shape[1]-1
approxi_input = u[:, :reduced]@np.diag(d)[:reduced, :reduced]@v[:reduced, :]
diff = input - approxi_input
# diff = input - u@np.diag(d)@v.T
frobenius_norm = np.sqrt(np.trace(diff.T@diff))
print(frobenius_norm)


# 희귀 행렬인 경우
input = RandomState(0).randint(0, 2, 3*5).reshape((3,5))
input = np.array([[0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 1, 0, 0],
                  [1, 0, 1, 0, 1, 0],
                  [0, 1, 0, 0, 0, 1]])
svd(input, full_matrices=False)

#
# 특이값 분해 예제: 추천 시스템
#

# 데이터 구성
# (Matrix, Alien, Serenity, Casablanca, Amelie)
X = np.array([[1,1,1,0,0],
              [3,3,3,0,0],
              [4,4,4,0,0],
              [5,5,5,0,0],
              [0,2,0,4,4],
              [0,0,0,5,5],
              [0,1,0,2,2]])

# 특이값 분해
u, d, v = svd(X, full_matrices=False)

# 계수 재 조정(rank=3)
rank = matrix_rank(X)
u = u[:, :rank]
d = d[:rank]
v = v[:rank, :]


# 고유값 벡터 또는 개념 축에 대한 이해
# V행렬에 대하여 가중치라고 할 수 있는 D 행렬을 곱한 후 코사인 유사도 구함
np.around((np.diag(d)@v).T, 2)
cosine_similarity((np.diag(d)@v).T)
print(np.around(cosine_similarity((np.diag(d)@v).T), 2))
# [[ 1.    0.95  1.    0.    0.  ]
#  [ 0.95  1.    0.95  0.2   0.2 ]
#  [ 1.    0.95  1.   -0.   -0.  ]
#  [ 0.    0.2  -0.    1.    1.  ]
#  [ 0.    0.2  -0.    1.    1.  ]]

print(np.around(cosine_similarity((np.diag(d)@v).T), 2))
# [[ 1.    0.95  1.    0.    0.  ]
#  [ 0.95  1.    0.95  0.2   0.2 ]
#  [ 1.    0.95  1.   -0.   -0.  ]
#  [ 0.    0.2  -0.    1.    1.  ]
#  [ 0.    0.2  -0.    1.    1.  ]]

# 차원 축소
reduced = d.shape[0]-1
u_reduced = u[:, :reduced]
d_reduced = d[:reduced]
v_reduced = v[:reduced, :]

# 축소된 행렬의 곱
X_2 = u_reduced @ np.diag(d_reduced) @ v_reduced
print(np.around(X_2, 2))
# [[ 0.99  1.01  0.99 -0.   -0.  ]
#  [ 2.98  3.04  2.98 -0.   -0.  ]
#  [ 3.98  4.05  3.98 -0.01 -0.01]
#  [ 4.97  5.06  4.97 -0.01 -0.01]
#  [ 0.36  1.29  0.36  4.08  4.08]
#  [-0.37  0.73 -0.37  4.92  4.92]
#  [ 0.18  0.65  0.18  2.04  2.04]]
diff = X- X_2
frobenius_norm = np.sqrt(np.trace(diff.T@diff))
print(frobenius_norm)
# 1.3455597127440255

#
# 추천 알고리즘 예시
#
q1 = np.array([5, 0, 0, 0, 0]).reshape(-1,1)
q2 = np.array([0, 4, 5, 0, 0]).reshape(-1,1)

# 각 벡터(이용자)가 개념축에 정사영
q1_proj = v_reduced @ q1
q2_proj = v_reduced @ q2
X_proj = v_reduced @ X.T

cosine_similarity(np.c_[q1_proj, q2_proj].T)
np.around(cosine_similarity(np.c_[q1_proj, X_proj].T), 2)

q1_proj_scaled = np.diag(d_reduced) @ v_reduced @ q1
q2_proj_scaled = np.diag(d_reduced) @ v_reduced @ q2
X_proj_scaled = np.diag(d_reduced) @v_reduced @ X.T
cosine_similarity(np.c_[q1_proj_scaled, q2_proj_scaled].T)

# q1 이용자와 기존 이용자 간의 코사인 유사도
q_X_cosine = cosine_similarity(np.c_[q1_proj_scaled, X_proj_scaled].T)[0]
np.around(q_X_cosine, 3)

# q2 이용자와의 코사인 유사도
q2_proj_scaled
q1_proj_scaled
q1_q2_cosine = cosine_similarity(np.c_[q1_proj_scaled,q2_proj_scaled].T)
np.around(q1_q2_cosine[0,1], 3)
# 0.996

#
# 개념 축에 정사영된 이용자 그래프
#

user_to_movie_proj = X_proj_scaled.T
user_to_movie = X_proj.T

plt.figure(figsize=(6,5))
plt.scatter(user_to_movie_proj[:, 0], user_to_movie_proj[:, 1], label='기존 이용자')
for j in np.arange(X.shape[0]):
    plt.text(user_to_movie_proj[j, 0]+0.2, user_to_movie_proj[j, 1]+0.3, j+1,
             horizontalalignment='left', verticalalignment='bottom', fontsize=9)
plt.xlabel('SF')
plt.ylabel('Romance')
plt.scatter(q1_proj_scaled[0], q1_proj_scaled[1], label='추천 대상자')
plt.scatter(q2_proj_scaled[0], q2_proj_scaled[1], label='비슷한 대상자')
plt.legend(loc='lower left')
plt.savefig(png_path + '/svd_concept_projection.png')
plt.show()



