#
# 프로그램 이름: pca_basics.py
# 작성자: Bong Ju Kang
# 설명: 주성분 분석 이해하기
#

# 필요한 패키지
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_mldata, load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

from numpy.random import RandomState
from numpy.linalg import svd, matrix_rank
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import sparse_random_matrix
from scipy.sparse import csr_matrix

# 초기 설정
png_path = "./data/png"
os.makedirs(png_path, exist_ok=True)

# 한글출력
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

#
# 주성분의 계산 예제
#

# 데이터 구성
X = RandomState(0).randint(0, 9, 5*4).reshape((5,4))
print(X)
# [[5 0 3 3]
#  [7 3 5 2]
#  [4 7 6 8]
#  [8 1 6 7]
#  [7 8 1 5]]

#
# 주성분 분석 적합
#

# 데이터 표준화
ss = StandardScaler()
ss.fit(X.astype(float))
X_scaled = ss.transform(X.astype(float))
print(np.around(X_scaled, 2))
# [[-0.82 -1.19 -0.62 -0.88]
#  [ 0.54 -0.25  0.41 -1.32]
#  [-1.5   1.    0.93  1.32]
#  [ 1.22 -0.88  0.93  0.88]
#  [ 0.54  1.32 -1.65  0.  ]]

# 주성분 적합
pca = PCA()
pca.fit(X_scaled)

# 주성분(V 행렬의 열벡터): 여기서는 V의 전치 행렬이 출력되므로 행 벡터가 된다.
print(np.around(pca.components_, 2))
# [[-0.44  0.47  0.33  0.69]
#  [-0.13  0.63 -0.76 -0.14]
#  [-0.89 -0.27 -0.   -0.37]
#  [ 0.08  0.56  0.57 -0.6 ]]


# 1st 주성분
pca.components_[0, :]

# 고유값: X행렬의 특이값의 제곱 또는 X^T*X 행렬의 고유값
e_values = pca.singular_values_**2

# 주성분의 설명력: 주성분의 분산
pca.singular_values_**2 / (X_scaled.shape[0]-1)
print(np.around(pca.explained_variance_, 2))
# [1.96 1.62 1.08 0.34]

# 주성분의 전체 분산에 대한 설명률
pca.explained_variance_ / np.sum(pca.explained_variance_)
print(np.around(pca.explained_variance_ratio_, 2))
# [0.39 0.32 0.22 0.07]

# 누적 설명률 그래프
plt.figure(figsize=(6,5))
plt.plot(pca.explained_variance_ratio_, label='설명률')
plt.plot(np.cumsum(pca.explained_variance_ratio_), label='누적설명률')
plt.xlabel('주성분')
plt.legend()
plt.show()
plt.savefig(png_path+'/pca_variance_ratio.png')

# 일부 주성분만을 이용한 차원 축소
# 4개중 3개만 이용
V = pca.components_[:3].T
X_reduced = (V@V.T@X_scaled.T).T
diff = X_scaled - X_reduced
# 프로베니어스 노름
np.sqrt(np.trace(diff.T@diff))
# 1.1738148567357103



#
# 주성분을 이용한 분석 예: MNIST 손글씨 숫자 데이터
#

#
# 데이터 구성
#
mnist = fetch_mldata("MNIST original")

# 표준화
max = np.max(mnist.data)
X = mnist.data / max
y = mnist.target
X.shape
# (70000, 784)

# 데이터 이해
rndperm = np.random.permutation(X.shape[0])

fig = plt.figure( figsize=(16,7) )
for i in np.arange(0,30):
    ax = fig.add_subplot(3, 10, i + 1)
    ax.matshow(X[rndperm[i]].reshape((28, 28)), cmap='binary')
    if i != 0:
        plt.xticks([], [])
        plt.yticks([], [])
plt.tight_layout()
plt.show()
plt.savefig(png_path + '/pca_mnist_image_plot.png')

#
# 모델 구성 및 적합
#
pca = PCA()
pca.fit(X)

# 고유값, 고유값 벡터
pca.singular_values_**2
pca.components_  # 고유 벡터(주성분)

# 주성분 점수
pca_scores = pca.transform(X)
pca_scores[:, 0]  #1st 주성분 점수
pca_scores[:, 1]  #2nd 주성분 점수
pca_scores.shape
# (70000, 784)

# 10,000개의 무작위 데이터에 대한 주성분 점수들에 대한 산점도
subset_index = RandomState(0).randint(0, X.shape[0], 10000)
plt.scatter(pca_scores[subset_index, 0], pca_scores[subset_index, 1], c=y[subset_index].astype(object), s=5)
plt.colorbar()
plt.xlabel('주성분 1')
plt.ylabel('주성분 2')
plt.show()
plt.savefig(png_path + '/pca_mnist_principal_components_plot.png')

# 주성분 분산 그래프
plt.plot(pca.explained_variance_ratio_)
plt.plot(np.cumsum(pca.explained_variance_ratio_))

# 차원 축소: 설명률이 주어진 경우 몇개의 주성분이 필요한지 파악
cutoff_explained_cumulative_ratio = 0.9
num_pcomp = np.argmin(np.cumsum(pca.explained_variance_ratio_) <= cutoff_explained_cumulative_ratio)
print(num_pcomp)
# 86
np.max(np.cumsum(pca.explained_variance_ratio_[:num_pcomp]))
# 0.8992309994465242

# 차원 축소 결과
V = pca.components_[:num_pcomp].T
X_reduced = (V@V.T@X.T).T
diff = X - X_reduced
# 프로베니어스 노름
np.sqrt(np.trace(diff.T@diff))
# 612.7129791631348

#
# 주성분 의미 해석
#
# 8 숫자 데이터만 추출
digit_8_index = y==8.0
X_8 = X[digit_8_index]
X_8.shape
# (6825, 784)

# 주성분 점수
X_8_scores = pca.transform(X_8)

# 전체 데이터 중에 10%씩에 해당하는 인덱스를 가져옴
percentile_index = (X_8.shape[0]*np.arange(0, 1, 0.1)).astype(int)
# 주성분 점수를 오름 차순으로 정렬한 후 각 10%씩에 해당하는 인덱스를 가져옴
raw_index = np.argsort(X_8_scores[:, 0])[percentile_index]
# 원 데이터에서 해당 데이터를 가져옴
X_8[raw_index]

# 10% 분위수에 해당하는 데이터에 대한 그래프
size = len(raw_index)
fig = plt.figure( figsize=(16,2) )
for i, j in enumerate(raw_index):
    ax = fig.add_subplot(1, size, i + 1)
    ax.imshow(X_8[j].reshape((28, 28)), cmap='binary')
    if i != 0:
        plt.xticks([], [])
        plt.yticks([], [])
plt.tight_layout()
plt.show()
plt.savefig(png_path + '/pca_mnist_scores_image_plot.png')


#
# 주성분 분석 예제 (DIGITS)
#

#
# 데이터 구성
#
bunch = load_digits()

X = bunch['data']
y = bunch['target']

X.shape
# (1797, 64)

# 표준화
max = np.max(X)
X_scaled = X / max

# 데이터 이해
rndperm = np.random.permutation(X.shape[0])

fig = plt.figure(figsize=(12, 8))
for i in np.arange(0,30):
    # ax = fig.add_subplot(3,10,i+1, title='Digit: ' + str(df.loc[rndperm[i],'label']) )
    ax = fig.add_subplot(3, 10, i + 1)
    ax.matshow(X[rndperm[i]].reshape((8, 8)), cmap='binary')
    # ax.matshow(df.loc[rndperm[i],feature_names].values.reshape((28,28)).astype(float))
    # ax.matshow(df.loc[df.label=='8.0', feature_names].values[i].reshape(28,28).astype(float))
    if i != 0:
        plt.xticks([], [])
        plt.yticks([], [])
plt.tight_layout()
plt.show()
plt.savefig(png_path + '/pca_digits_image_plot.png')

#
# 모델 구성 및 적합
#
pca = PCA()
pca.fit(X_scaled)

# 고유값, 고유값 벡터
pca.singular_values_**2
pca.components_
# 1st 주성분
pca.components_[0, :]


# 주성분 점수
pca_scores = pca.transform(X_scaled)
pca_scores[:, 0] #1st 주성분 점수
pca_scores[:, 1] #2nd 주성분 점수
pca_scores.shape
# (70000, 784)

# 10,000개의 무작위 데이터에 대한 주성분 점수들에 대한 산점도
subset_index = RandomState(0).randint(0, X.shape[0], 10000)
plt.scatter(pca_scores[subset_index, 0], pca_scores[subset_index, 1], c=y[subset_index].astype(object), s=5)
plt.colorbar()
plt.xlabel('주성분 1')
plt.ylabel('주성분 2')
plt.show()
plt.savefig(png_path + '/pca_digits_principal_components_plot.png')

#
# for t-SNE
#
n_sne = 1000
time_start = os.times()
tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=1000)
tsne_results = tsne.fit_transform(X_scaled[rndperm[:n_sne]])

tsne_label = y[rndperm[:n_sne]]
x_tsne = tsne_results[:,0]
y_tsne = tsne_results[:,1]

lblist = np.unique(y)
colors = [plt.cm.jet(float(i)/len(lblist)) for i in lblist]
for i, j in enumerate(lblist):
    x_tsne_class = x_tsne[tsne_label == i]
    y_tsne_class = y_tsne[tsne_label == i]
    # colorlist = df_tsne[df.label == j]['label']
    plt.scatter(x_tsne_class, y_tsne_class, c=colors[i], s=7, label=str(lblist[i]), alpha=0.8)
plt.legend(loc='best').draggable()
plt.title('DIGITS: t-SNE 그래프')
plt.show()
plt.savefig(png_path + '/tsne_digits_plot.png')

# 주성분 분산 그래프
plt.plot(pca.explained_variance_ratio_)
plt.plot(np.cumsum(pca.explained_variance_ratio_))

# 차원 축소: 설명률이 주어진 경우 몇개의 주성분이 필요한지 파악
cutoff_explained_cumulative_ratio = 0.9
num_pcomp = np.argmin(np.cumsum(pca.explained_variance_ratio_) <= cutoff_explained_cumulative_ratio)
print(num_pcomp)
# 20
np.max(np.cumsum(pca.explained_variance_ratio_[:num_pcomp]))
# 0.8943031165985265










