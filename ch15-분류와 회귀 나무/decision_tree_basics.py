#
# 프로그램 이름: decision_tree_basics.py
# 작성자: Bong Ju Kang
# 설명: 결정 나무 기법을 이용한 모델링 이해하기
#

# 필요한 패키지
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests, zipfile, io

from sklearn.model_selection import train_test_split, KFold
from sklearn import metrics

from sklearn.datasets import load_boston
import pydot
from PIL import Image
from sklearn.tree import DecisionTreeRegressor, export_graphviz, DecisionTreeClassifier

# 초기 설정
png_path = "./data/png"
os.makedirs(png_path, exist_ok=True)

# 한글출력
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(1234)  # for reproducibility

#
# 데이터 가져오기
#
# UCI Machine Learning Repository: 원래 여기에 있던 데이터 임
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data", header=None)
# Attribute Information:
#
#     1. CRIM      per capita crime rate by town
#     2. ZN        proportion of residential land zoned for lots over
#                  25,000 sq.ft.
#     3. INDUS     proportion of non-retail business acres per town
#     4. CHAS      Charles River dummy variable (= 1 if tract bounds
#                  river; 0 otherwise)
#     5. NOX       nitric oxides concentration (parts per 10 million)
#     6. RM        average number of rooms per dwelling
#     7. AGE       proportion of owner-occupied units built prior to 1940
#     8. DIS       weighted distances to five Boston employment centres
#     9. RAD       index of accessibility to radial highways
#     10. TAX      full-value property-tax rate per $10,000
#     11. PTRATIO  pupil-teacher ratio by town
#     12. B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks
#                  by town
#     13. LSTAT    % lower status of the population
#     14. MEDV     Median value of owner-occupied homes in $1000's

# 데이터 불러오기
X = load_boston()['data']
y = load_boston()['target']
X.shape
# (506, 13)

# 변수 명 보기
features = load_boston()['feature_names']
# array(['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
#        'TAX', 'PTRATIO', 'B', 'LSTAT'], dtype='<U7')

# 입력 변수 일부 보기
X[:1]
# array([[6.320e-03, 1.800e+01, 2.310e+00, 0.000e+00, 5.380e-01, 6.575e+00,
#         6.520e+01, 4.090e+00, 1.000e+00, 2.960e+02, 1.530e+01, 3.969e+02,
#         4.980e+00]])

# 결정 나무를 그리기 위한 doe.exe가 있는 경로를 추가
os.environ["PATH"] += os.pathsep + "C:\\ProgramData\\Anaconda3\\Library\\bin\\graphviz"

#
# 회귀 결정 나무
#
# 변수 중 방의 개수와 중심가와의 거리 만 선택
sel_index = np.argwhere((features == 'RM') | (features == 'DIS')).ravel()
sel_X = X[:, sel_index]

# 회귀 결정 나무 적합
regr = DecisionTreeRegressor(max_leaf_nodes=5)
regr.fit(sel_X, y)
regr.score(sel_X, y)
# 0.6466349514463494

#  결정 나무 출력
# 문자열 데이터 정의
dot_data = io.StringIO()

# 문자열 데이터에 결정 나무 보내기
export_graphviz(regr, out_file=dot_data, feature_names=['RM', 'DIS'], class_names=None, filled=True)

# 문자열 데이터로부터 그래프 생성
graph, = pydot.graph_from_dot_data(dot_data.getvalue())

# 그래프 보여주기
plt.figure(figsize=(6, 5))
plt.axis("off")
graph.set('dpi', 300)
graph.set('pad', 0.2)
plt.title('개수(RM)와 거리(DIS)에 의한 주택 가격')
plt.imshow(Image.open(io.BytesIO(graph.create_png())), interpolation='bilinear')
plt.savefig(png_path + '/cart_treeSample.png')

# 입력 공간의 분할 예시
np.min(sel_X[:, 1])
plt.figure(figsize=(8,6))
plt.scatter(sel_X[:, 0], sel_X[:, 1], c='lightblue', s=15 )
plt.title('입력 공간의 분할')
plt.xlabel('개수')
plt.ylabel('거리')
plt.xlim(3, 9)
plt.ylim(1, 12.5)
# plt.gca().yaxis.tick_right()
plt.xticks([3, 6.941, 9])
plt.yticks([1, 2.588, 12.5])
plt.vlines(6.941, ymin=1, ymax=12.5)
plt.hlines(2.588, xmin=3, xmax=6.941)
plt.annotate('R1', xy=(8, 5), fontsize='xx-large')
plt.annotate('R2', xy=(4, 1.5), fontsize='xx-large')
plt.annotate('R3', xy=(4, 5), fontsize='xx-large')
# plt.subplots_adjust(bottom=0.15, right=0.85)
plt.savefig(png_path + '/cart_inputSpace.png')
plt.show()

# 계산
252*18.183 - (199*10.976 + 53*13.258)
# 1695.2179999999998
76*79.729 - (46*41.296 + 30*36.628)
# 3060.9479999999994


# 분류 나무을 위한 분류 기준
p = np.linspace(0.001,0.999, 100)
miss_rate = 1-np.max([p, 1-p], axis=0)
gini_index = 2*p*(1-p)
cross_entropy = -p*np.log2(p)-(1-p)*np.log2(1-p)
scaled_cross_entropy = 1/2*cross_entropy

# 손실 함수 그래프
plt.figure(figsize=(7, 5))
plt.plot(p, scaled_cross_entropy, ":", label="교차엔트로피(크기조정)")
plt.plot(p, miss_rate, "-", label="오분류율")
plt.plot(p, gini_index, "--", label="지니지수")
plt.title('노드의 특정 범주 예측 확률값에 따른 손실 함수값')
plt.xlabel("p")
plt.ylabel("각 지수값")
plt.legend()
plt.savefig(png_path + '/cart_classificationLoss.png')
plt.show()

#
# 회귀 결정 나무 적합 (모든 변수 사용)
#

# 데이터 분할
X = load_boston()['data']
y = load_boston()['target']
features = load_boston()['feature_names']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# 회귀 결정 나무 적합
regr = DecisionTreeRegressor(min_samples_leaf=5)
regr.fit(X_train, y_train)
regr.score(X_test, y_test)
# 0.6212616301853735

#  결정 나무 출력
# 문자열 데이터 정의
dot_data = io.StringIO()

# 문자열 데이터에 결정 나무 보내기
export_graphviz(regr, out_file=dot_data, feature_names=features, class_names=None, filled=True)

# 문자열 데이터로부터 그래프 생성
graph, = pydot.graph_from_dot_data(dot_data.getvalue())

# 그래프 보여주기
plt.figure(figsize=(15, 7))
plt.axis("off")
graph.set('dpi', 300)
graph.set('pad', 0.1)
plt.tight_layout()
plt.imshow(Image.open(io.BytesIO(graph.create_png())), interpolation='bilinear')
plt.savefig(png_path + '/cart_tree.png')

#
# 예제: [BANK] 데이터의 결정 나무 적합
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
df = pd.read_csv(z.open('bank.csv'),sep=';')
df.shape
# (4521, 17)

df.to_csv

#
# 데이터 전 처리
#
df.y.value_counts() # 목표 변수 분포 확인
# no     4000
# yes     521
# df.y.value_counts()/df.shape[0]

# 변수 정의
feature_list = [name for name in df.columns if name !='y']
categorical_variables = df.columns[(df.dtypes == 'object') & (df.columns != 'y')]
num_variables = [name for name in feature_list if name not in categorical_variables]

# 범주형 데이터를 숫자형 데이터로 전환
df_X_onehot = pd.get_dummies(df[categorical_variables], prefix_sep='_')
df_y_onehot = pd.get_dummies(df['y'], drop_first=True)

# 범주형 데이터와 숫자형 데이터 결합
X = np.c_[df[num_variables].values, df_X_onehot.values]
y = df_y_onehot.values

#
# 데이터 분할
#
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#
# 결정 나무 적합
#
# 모델 구성
clf = DecisionTreeClassifier(random_state=123, min_samples_leaf=5,  max_depth=5)
# DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=5,
#             max_features=None, max_leaf_nodes=None,
#             min_impurity_decrease=0.0, min_impurity_split=None,
#             min_samples_leaf=5, min_samples_split=2,
#             min_weight_fraction_leaf=0.0, presort=False, random_state=123,
#             splitter='best')
clf = DecisionTreeClassifier(random_state=123, max_leaf_nodes=11)

# 모델 적합
clf.fit(X_train, y_train)

# 모델 평가: 정확도
clf.score(X_test, y_test)
# 0.9027266028002948


#
# 최적의 결정 나무 생성: 교차 검증을 통한 최적의 나무 잎의 개수를 결정
#

# 오분류율 벡터 초기화
miscl_rate = np.zeros(shape=(198,))

# 교차 검증을 위한 초기 세팅
n_splits = 3
kf =KFold(n_splits=n_splits, random_state=0)

# 잎의 개수에 따른 평가 데이터의 오 분류율 계산
for i in np.arange(2, 200):
    clf = DecisionTreeClassifier(random_state=123, max_leaf_nodes=i, min_samples_leaf=5)
    cv_values =np.zeros(shape=(n_splits,))
    # 교차 검증 루틴
    j=0
    for train_idx, test_idx in kf.split(X_train):
        clf.fit(X_train[train_idx], y_train[train_idx])
        cv_values[j] = 1 - clf.score(X_train[test_idx], y_train[test_idx])
        j += 1
    # 평균 오류율 계산
    miscl_rate[i-2] = np.mean(cv_values)


# 오 분류율의 최소 값을 주는 결정 나무 잎의 개수
min_index = np.argmin(miscl_rate)
num_leaves = min_index + 2
# 28
min_value = miscl_rate[min_index]
# 0.10240414159854432

# 잎의 개수에 따른 오 분류율 그래프
plt.figure(figsize=(6, 6))
plt.plot(miscl_rate)
plt.xlabel('잎의 개수')
plt.ylabel('오분류율')
plt.title("잎의 개수 대 오분류율(교차 검증)", weight='bold')
plt.annotate("나무잎의 개수: %d" % num_leaves , xy=(min_index, min_value),
             xytext=(35, 15), textcoords='offset points',
             arrowprops={'arrowstyle': '->', 'color': 'r', 'lw': 1})
plt.savefig(png_path + '/cart_pruning_cv.png')
plt.show()

#
# 결정 나무 출력
#

# 최적의 나무 적합
clf = DecisionTreeClassifier(random_state=123, max_leaf_nodes=28, min_samples_leaf=5)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
# 0.8975681650700074

# 문자열 데이터 정의: 메모리 저장
dot_data = io.StringIO()

# 문자열 데이터에 결정 나무 보내기
feature_names = num_variables + df_X_onehot.columns.tolist()
export_graphviz(clf, out_file=dot_data, feature_names=feature_names, class_names=None, filled=True)

# 문자열 데이터로부터 그래프 생성
graph, = pydot.graph_from_dot_data(dot_data.getvalue())

# 결정 나무 출력
plt.figure(figsize=(15, 7))
plt.axis("off")
graph.set('dpi', 300)
graph.set('pad', 0.1)
plt.tight_layout()
plt.imshow(Image.open(io.BytesIO(graph.create_png())), interpolation='bilinear')
plt.savefig(png_path + '/cart_tree_example.png')

#
# ROC 그래프 그리기
#
# 모델에 의한 예측 확률 계산
y_pred_proba = clf.predict_proba(X_test)[::, 1]

# fpr: 1-특이도, tpr: 민감도, auc 계산
fpr, tpr, _ = metrics.roc_curve(y_true=y_test,  y_score=y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)

# ROC 그래프 출력
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label="결정나무\n곡선밑 면적(AUC)=" + "%.4f" % auc)
plt.plot([-0.02, 1.02], [-0.02, 1.02], color='gray', linestyle=':', label='무작위 모델')
plt.margins(0)
plt.legend(loc=4)
plt.xlabel('fpr: 1-Specificity')
plt.ylabel('tpr: Sensitivity')
plt.title("ROC Curve", weight='bold')
plt.legend()
plt.savefig(png_path + '/cart_ROC.png')
plt.show()

