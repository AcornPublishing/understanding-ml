#
# 프로그램 이름: cluster_basics.py
# 작성자: Bong Ju Kang
# 설명: 군집 분석 이해하기
#

# 필요한 패키지
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import cm
from numpy.random import RandomState

from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs
from sklearn.metrics import silhouette_score, silhouette_samples

# 초기 설정
png_path = "./data/png"
os.makedirs(png_path, exist_ok=True)

# 한글출력
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

#
# 격차 통계량 계산 예제
#

# 데이터 구성
X, y = make_blobs(n_samples=1000, n_features=2, centers=10, random_state=7)
X.shape
# (1000, 2)

# 데이터 이해
plt.figure(figsize=(7,6))
plt.scatter(X[:, 0], X[:,1], c=y,  s=9)
plt.title("산점도와 군집(10개) 분포")
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.colorbar()
plt.show()
plt.savefig(png_path+'/cluster_blob_scatter.png')

#
# 격차 통계량 계산
#

# 계산을 위한 초 모수 정의

# 최대 군집의 개수
max_clusters = 20

# 참조 분포의 개수
num_ref_dists = 10

# 참조 분포의 차원: (샘플 개수, 특징 개수)
num_features = 2
B = 100
num_ref_data_shape = (B, num_features)

# 격차 통계량 자리 지킴이(placeholder)
gap_stat = np.zeros(shape=(max_clusters,))

# 각 군집의 개수에 대하여
for index, clusters in enumerate(np.arange(1, max_clusters+1)):
    # 참조 분포의 wcss  자리 확보
    ref_wcss = np.zeros(num_ref_dists)
    # 각 참조 분포에 대하여
    for j in np.arange(num_ref_dists):
        # 참조 분포의 생성 (b-a)*uniform() + a: 유계 상자
        random_dist = (np.max(X, axis=0) - np.min(X, axis=0)) * \
                      RandomState(j).random_sample(num_ref_data_shape) + \
                      np.min(X, axis=0).reshape(1, 2)
        # 적합
        km = KMeans(clusters)
        km.fit(random_dist)
        # WCSS
        ref_wcss[j] = km.inertia_
    # 원 데이터 적합
    km = KMeans(clusters)
    km.fit(X)
    # 원 데이터 WCSS
    wcss = km.inertia_
    # 격차 통계량 계산
    gap_stat[index] = np.mean(np.log(ref_wcss)) - np.log(wcss)
print(gap_stat)
# [-2.19660846 -1.9590175  -1.90628713 -1.95859784 -1.72469402 -1.33368902
#  -1.16974462 -0.94765692 -0.94574371 -0.9539331  -1.036548   -1.10812855
#  -1.18153949 -1.21712557 -1.27528754 -1.33497447 -1.34521287 -1.36448381
#  -1.42248713 -1.41095365]

# 격차 통계량 그래프
plt.figure(figsize=(7, 7))
plt.plot(np.arange(max_clusters), gap_stat)
plt.xticks(np.arange(max_clusters),np.arange(1, max_clusters+1) )
plt.grid()
plt.title('군집개수에 따른 격차통계량의 값')
plt.xlabel('군집 개수')
plt.ylabel('격차통계량 값')
plt.show()
plt.savefig(png_path+'/cluster_blob_gap.png')

# 원 데이터와 참조 분포 그래프
random_dist = (np.max(X, axis=0) - np.min(X, axis=0)) * \
                      RandomState(0).random_sample(num_ref_data_shape) + \
                      np.min(X, axis=0).reshape(1, 2)
plt.figure(figsize=(7,6))
plt.scatter(X[:, 0], X[:,1],  s=9, label='데이터')
plt.scatter(random_dist[:, 0], random_dist[:,1], c='orange', s=7, label='무작위분포')
plt.title('원 데이터와 참조 분포 데이터')
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.legend()
plt.show()
plt.savefig(png_path+'/cluster_blob_with_random_scatter.png')

#
# 실루엣 통계량 계산
#

# 추가 필요한 패키지
from sklearn.metrics import silhouette_score

# 실루엣 결과 자리 지킴이
sil_avg = np.zeros(shape=(max_clusters-1,))

# 각 군집에 대하여
for index, clusters in enumerate(np.arange(2, max_clusters+1)):
    km = KMeans(clusters)
    km.fit(X)
    cluster_label = km.predict(X=X)
    sil_avg[index] = silhouette_score(X, cluster_label)

print(sil_avg)
# [0.50187747 0.53630729 0.56448052 0.57390226 0.64018763 0.64664969
#  0.68289145 0.6187224  0.56237004 0.52021199 0.51151902 0.50893581
#  0.41552878 0.4317837  0.41651814 0.37374767 0.3740731  0.35672327
#  0.34760964]

# 군집 개수에 따른 실루엣 평균값 그래프
plt.figure(figsize=(7, 7))
plt.plot(np.arange(max_clusters-1), sil_avg)
plt.xticks(np.arange(max_clusters-1), np.arange(2, max_clusters+1) )
plt.grid()
plt.title('군집 개수에 따른 실루엣 평균값')
plt.xlabel('군집 개수')
plt.ylabel('실루엣 평균값')
plt.show()
plt.savefig(png_path+'/cluster_blob_silhouette.png')

# 최적 군집에 대한 평가: 각 군집별 실루엣 계수값 비교
opt_clusters = 8
km = KMeans(opt_clusters)
km.fit(X)
cluster_label = km.predict(X=X)
silhouette_samples(X, cluster_label)

# 최적 산점도
colors = cm.nipy_spectral(cluster_label.astype(float) / max_clusters)
fig = plt.figure(figsize=(6,5))
ax = fig.add_subplot(1,1,1)
ax.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7, c=colors, edgecolor='k')
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.show()
plt.savefig(png_path+'/cluster_optimal_scatter.png')

# 실루엣 대 산점도
colors = cm.nipy_spectral(cluster_label.astype(float) / max_clusters)
fig = plt.figure(figsize=(12,5))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
ax1 = ax1.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7, c=colors, edgecolor='k')
for i in np.arange(2, opt_clusters+1):
    ith_sil_score = np.sort(silhouette_samples(X, cluster_label)[cluster_label==i])[::-1]
    pcolor = cm.nipy_spectral(np.float(i) / max_clusters)
    ax2 = plt.plot(ith_sil_score, c=pcolor,  label=np.str(i))
ax2 = plt.xlabel('군집 내 표본 번호')
ax2 = plt.ylabel('군집 별 실루엣 계수값')
ax2 = plt.legend()







