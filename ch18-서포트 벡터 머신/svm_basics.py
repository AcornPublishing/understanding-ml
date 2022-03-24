#
# 프로그램 이름: svm_basics.py
# 작성자: Bong Ju Kang
# 설명: 서포트벡터 머신 이해하기
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

# 3차원 그래프
from mpl_toolkits.mplot3d import Axes3D

# 초기 설정
png_path = "./data/png"
os.makedirs(png_path, exist_ok=True)

# 한글출력
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

#
# 예제: 커널 함수로 분리 가능한 경우(하드 마진)
#

# 데이터 만들기
X, y = make_blobs(n_samples=100, centers=2, cluster_std=0.7, random_state=0)

# 데이터 구성 확인
plt.figure(figsize=(6,6))
plt.scatter(X[:,0], X[:,1], c=y, cmap='rainbow')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('산점도', weight='bold')
plt.savefig(png_path + '/svm_scatter.png')
plt.show()

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

#
# 디폴트 SVM
#
clf = SVC()
print(clf)
# SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
#   decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
#   max_iter=-1, probability=False, random_state=None, shrinking=True,
#   tol=0.001, verbose=False)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
# 1.0

#
# SVM 적합: 선형 커널 적합
#
# 하드 마진인 경우에 벌점 상수 C를 크게 해야 함
clf = SVC(kernel='linear', C=1000)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
# 1.0

# 적합된 결정 함수
clf.intercept_
# array([6.09562784])
clf.coef_
# array([[ 0.41403294, -2.48273044]])

# 서포트 벡터
clf.support_vectors_
# array([[2.56509832, 3.28573136],
#        [0.35482006, 2.9172298 ],
#        [1.23408114, 2.25819849]])

# 결정 경계선 그래프
# 그리드 생성
x_min = np.min(X)
x_max = np.max(X)
x_space = np.linspace(x_min, x_max, 50)
y_space = np.linspace(x_min, x_max, 50)
X_grid, Y_grid = np.meshgrid(x_space, y_space)
xy = np.c_[X_grid.ravel(), Y_grid.ravel()]

# 그리드에 의한 결정함수 값
dvalue = clf.decision_function(xy).reshape(X_grid.shape)

# 마진과 결정경계선 그래프
plt.figure(figsize=(6,6))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='rainbow')
plt.contour(X_grid, Y_grid, dvalue, colors='k',
           levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
            s=200,  linewidths=1, facecolors='none', edgecolors='k',
            label='서포트벡터')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('마진과 결정경계선', weight='bold')
plt.legend()
plt.savefig(png_path + '/svm_ex_hard_margin.png')
plt.show()

#
# 18.13. 예제: 커널 함수로 분리가 불 가능한 경우(소프트 마진)
#

# 데이터 만들기
X, y = make_blobs(n_samples=100, centers=2, cluster_std=1.2, random_state=0)

# 데이터 구성 확인
plt.figure(figsize=(6,6))
plt.scatter(X[:,0], X[:,1], c=y, cmap='rainbow')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('산점도', weight='bold')
plt.savefig(png_path + '/svm_scatter_soft.png')
plt.show()

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

# SVM 적합
clf = SVC(kernel='linear', C=1000)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
# 0.9333333333333333

# 서포트 벡터의 수 (클래스별)
clf.n_support_
# array([8, 8])

# 결정 경계선 그래프
# 그리드 생성
x_min = np.min(X)
x_max = np.max(X)
x_space = np.linspace(x_min, x_max, 50)
y_space = np.linspace(x_min, x_max, 50)
X_grid, Y_grid = np.meshgrid(x_space, y_space)
xy = np.c_[X_grid.ravel(), Y_grid.ravel()]

# 그리드에 의한 결정함수 값
dvalue = clf.decision_function(xy).reshape(X_grid.shape)

# 마진과 결정경계선 그래프
plt.figure(figsize=(6,6))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='rainbow')
plt.contour(X_grid, Y_grid, dvalue, colors='k',
           levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
            s=200,  linewidths=1, facecolors='none', edgecolors='k',
            label='서포트벡터')

plt.xlabel('x1')
plt.ylabel('x2')
plt.title('마진과 결정경계선(소프트마진)', weight='bold')
plt.legend()
plt.savefig(png_path + '/svm_ex_soft_margin.png')
plt.show()

#
# 여유 변수의 의미
#

# 데이터 만들기
X, y = make_blobs(n_samples=300, centers=2, cluster_std=1.5, random_state=0)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)

# SVM 모델 적합
clf = SVC(kernel='linear', C=10000)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
# 0.9333333333333333

# 결정 경계선 그래프
# 그리드 생성
x_min = np.min(X)
x_max = np.max(X)
x_space = np.linspace(x_min, x_max, 50)
y_space = np.linspace(x_min, x_max, 50)
X_grid, Y_grid = np.meshgrid(x_space, y_space)
xy = np.c_[X_grid.ravel(), Y_grid.ravel()]

# 그리드에 의한 결정함수 값
dvalue = clf.decision_function(xy).reshape(X_grid.shape)


# 여유 변수의 정의
slack_shape = X_train.shape[0]
y_train_svm = np.where(y_train==0, -1, 1)
slack = np.max([np.zeros(shape=slack_shape), 1-y_train_svm *clf.decision_function(X_train)], axis=0)
slack_index = (slack != 0)
seq_num = np.arange(np.sum(slack_index))

# 산점도와 여유변수 그래프
plt.figure(figsize=(6,6))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='rainbow', s=13)

# 여유 변수값 처리
for j in seq_num:
    plt.text(X_train[slack_index][j,0], X_train[slack_index][j,1], j, fontsize=9)

plt.contour(X_grid, Y_grid, dvalue, colors='k',
           levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])
plt.xlabel('x1')
plt.ylabel('x2')
plt.title(r'산점도와 여유변수값$(\xi_i)$', weight='bold')
# plt.legend()
plt.savefig(png_path + '/svm_ex_soft_margin_slack.png')
plt.show()

#
# 예제: 선형 이외의 커널 함수 적용 (가우시안 커널)
#
# 데이터 만들기
X, y = make_blobs(n_samples=100, centers=2, cluster_std=1.2, random_state=0)

# 데이터 구성 확인
plt.figure(figsize=(6,6))
plt.scatter(X[:,0], X[:,1], c=y, cmap='rainbow')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('산점도', weight='bold')
plt.savefig(png_path + '/svm_scatter_soft.png')
plt.show()

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

# SVM 적합
clf = SVC(kernel='rbf', C=1000)
clf.fit(X_train, y_train)
clf.score(X_train, y_train)
# 1.0
clf.score(X_test, y_test)
# 0.8333333333333334

# 서포트 벡터의 수 (클래스별)
clf.n_support_
# array([17, 18])

# 결정 경계선 그래프
# 그리드 생성
x_min = np.min(X)
x_max = np.max(X)
x_space = np.linspace(x_min, x_max, 50)
y_space = np.linspace(x_min, x_max, 50)
X_grid, Y_grid = np.meshgrid(x_space, y_space)
xy = np.c_[X_grid.ravel(), Y_grid.ravel()]

# 그리드에 의한 결정함수 값
dvalue = clf.decision_function(xy).reshape(X_grid.shape)

# 마진과 결정경계선 그래프
plt.figure(figsize=(6,6))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='rainbow')
plt.contour(X_grid, Y_grid, dvalue, colors='k',
           levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
            s=200,  linewidths=1, facecolors='none', edgecolors='k',
            label='서포트벡터')

plt.xlabel('x1')
plt.ylabel('x2')
plt.title('마진과 결정경계선(가우시안 커널)', weight='bold')
plt.legend()
plt.savefig(png_path + '/svm_ex_soft_margin_rbf.png')
plt.show()


# 마진과 결정경계선 그래프 (평가 데이터 이용)
plt.figure(figsize=(6,6))
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='rainbow')
plt.contour(X_grid, Y_grid, dvalue, colors='k',
           levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('마진과 결정경계선(가우시안 커널, 평가데이터)', weight='bold')
plt.savefig(png_path + '/svm_ex_soft_margin_rbf_test_data.png')
plt.show()

#
# 서포트벡터 머신의 초 모수의 결정: 그리드 방식
#

# 데이터 만들기
X, y = make_blobs(n_samples=100, centers=2, cluster_std=1.2, random_state=0)

# 데이터 구성 확인
plt.figure(figsize=(6,6))
plt.scatter(X[:,0], X[:,1], c=y, cmap='rainbow')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('산점도', weight='bold')
plt.savefig(png_path + '/svm_scatter_soft.png')
plt.show()

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)

# 초 모수 집합 정의
param_grid = {'kernel': ['linear', 'poly', 'rbf'],
              'C': [1, 10, 100]}

# 초 모수 값의 조합에 의한 모델 적합
sv = SVC(random_state=0)
grid_search = GridSearchCV(estimator=sv, param_grid=param_grid, cv=3, n_jobs=-1)
grid_search.fit(X_train, y_train)

# 선택된 초 모수 값
print(grid_search.best_params_)
# {'C': 10, 'kernel': 'linear'}

# 선택된 초 모수에 의한 적합
opt_param = grid_search.best_params_
opt_param.update({'random_state': 0})

clf = SVC(**opt_param)
clf.fit(X_train, y_train)
acc = clf.score(X_test, y_test)
print("모델 정확도: {:.4f}".format(acc))
# 모델 정확도: 0.8667

#
# 18.16. 예제: [BANK] 데이터 적용
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
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=1234)

# 초 모수 집합 정의
param_grid = {'kernel': ['linear', 'poly', 'rbf'],
              'C': [1, 10, 100]}

# 초 모수 값의 조합에 의한 모델 적합
sv = SVC(random_state=0)
grid_search = GridSearchCV(estimator=sv, param_grid=param_grid, cv=3, n_jobs=-1)
grid_search.fit(X_train, y_train)

# 선택된 초 모수 값
print(grid_search.best_params_)
# {'C': 10, 'kernel': 'linear'}

# 선택된 초 모수에 의한 적합
opt_param = grid_search.best_params_
opt_param.update({'random_state': 0})

clf = SVC(**opt_param)
clf.fit(X_train, y_train)
acc = clf.score(X_test, y_test)
print("모델 정확도: {:.4f}".format(acc))
# 모델 정확도: 0.8961

#
# 모델 평가: ROC 그래프
#
# 최적 모델 적합
param_grid = {'kernel': 'linear',
              'C': 10,
              'probability': True,
              'random_state': 0}
clf = SVC(**param_grid)
clf.fit(X_train, y_train)
acc = clf.score(X_test, y_test)
print("모델 정확도: {:.4f}".format(acc))
# 모델 정확도: 0.8961

# 모델에 의한 예측 확률 계산
y_pred_proba = clf.predict_proba(X_test)[::, 1]

# fpr: 1-특이도, tpr: 민감도, auc 계산
fpr, tpr, _ = metrics.roc_curve(y_true=y_test, y_score=y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)

# ROC 그래프 출력
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label="서포트벡터머신\n곡선밑 면적(AUC)=" + "%.4f" % auc)
plt.plot([-0.02, 1.02], [-0.02, 1.02], color='gray', linestyle=':', label='무작위 모델')
plt.margins(0)
plt.legend(loc=4)
plt.xlabel('fpr: 1-Specificity')
plt.ylabel('tpr: Sensitivity')
plt.title("ROC Curve", weight='bold')
plt.legend()
plt.savefig(png_path + '/svm_ROC.png')
plt.show()
