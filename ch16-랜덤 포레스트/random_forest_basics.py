#
# 프로그램 이름: random_forest_basics.py
# 작성자: Bong Ju Kang
# 설명: 랜덤 포레스트 모델링 이해하기
#

# 필요한 패키지
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests, zipfile, io

from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, KFold

# 초기 설정
png_path = "./data/png"
os.makedirs(png_path, exist_ok=True)

# 한글출력
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


#
# 예제: [BANK] 데이터의 랜덤포레스트 적합
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

#
# 데이터 전 처리
#

# 목표 변수 분포 확인
df.y.value_counts()
# no     4000
# yes     521
base_dist = df.y.value_counts()/df.shape[0]
# no     0.88476
# yes    0.11524

# 변수 정의
feature_list = [name for name in df.columns if name !='y']
categorical_variables = df.columns[(df.dtypes =='object') & (df.columns != 'y')]
num_variables = [name for name in feature_list if name not in  categorical_variables]

# 범주형 데이터를 숫자형 데이터로 전환
df_X_onehot = pd.get_dummies(df[categorical_variables], prefix_sep='_')
df_y_onehot = pd.get_dummies(df['y'], drop_first=True)

# 범주형 데이터와 숫자형 데이터 결합
X = np.c_[df[num_variables].values, df_X_onehot.values]
y = df_y_onehot.values.ravel()

# 모든 특징의 이름 리스트
feature_names = num_variables + df_X_onehot.columns.tolist()

#
# 데이터 분할
#
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

#
# 랜덤 포레스트 적합
#
# 모델 구성: 나무는 100개 사용하며, 분기 변수의 개수는 자동
clf = RandomForestClassifier(random_state=123, n_estimators=100, min_samples_leaf=5)
print(clf)
# RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#             max_depth=None, max_features='auto', max_leaf_nodes=None,
#             min_impurity_decrease=0.0, min_impurity_split=None,
#             min_samples_leaf=5, min_samples_split=2,
#             min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
#             oob_score=False, random_state=123, verbose=0, warm_start=False)

# 모델 적합
clf.fit(X_train, y_train)

# 모델 평가: 정확도
clf.score(X_test, y_test)
# 0.9019896831245394

#
# 초 모수의 선택
#
# 초 모수 집합 정의
param_grid = {'n_estimators': [100, 200, 300, 400, 500],
              'max_features':[2, 3, 5, 7, 10],
              'min_samples_leaf': [3,5]}

# 초 모수 값의 조합에 의한 모델 적합
rf = RandomForestClassifier()
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1)
grid_search.fit(X_train, y_train)

# 선택된 초 모수 값
print(grid_search.best_params_)
# {'max_features': 10, 'min_samples_leaf': 3, 'n_estimators': 500}

# 선택된 초 모수에 의한 적합
clf = RandomForestClassifier(random_state=123, n_estimators=500,
                             min_samples_leaf=3, max_features=10, n_jobs=-1)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
# 0.9034635224760501

#
# 최적의 랜덤 포레스트 생성(교차 검증)
#
# 오분류율 벡터 초기화
num_trees = 200
miscl_rate = np.zeros(shape=(num_trees,))

# 교차 검증을 위한 초기 세팅
n_splits = 3
kf = KFold(n_splits=n_splits, random_state=0)

# 나무 개수에 따른 교차 검증 평가 데이터의 오 분류율 계산
for i in np.arange(1, num_trees+1):
    clf = RandomForestClassifier(random_state=123, n_estimators=i,
                                 min_samples_leaf=3, max_features=10, n_jobs=-1)
    # 교차 검증 루틴
    cv_values = np.zeros(shape=(n_splits,))
    j=0
    for train_idx, test_idx in kf.split(X_train):
        clf.fit(X_train[train_idx], y_train[train_idx])
        cv_values[j] = 1 - clf.score(X_train[test_idx], y_train[test_idx])
        j += 1
    # 평균 오류율 계산
    miscl_rate[i-1] = np.mean(cv_values)

# 오 분류율의 최소 값을 주는 나무의 개수
min_index = np.argmin(miscl_rate)
opt_trees = min_index + 1
# 61
min_value = miscl_rate[min_index]
# 0.10366346813912845

# 나무의 개수에 따른 오 분류율 그래프
plt.figure(figsize=(6, 6))
plt.plot(miscl_rate)
plt.xlabel('나무의 개수')
plt.ylabel('오분류율')
plt.title("나무의 개수 대 오분류율", weight='bold')
plt.annotate("나무의 개수: %d" % opt_trees , xy=(min_index, min_value),
             xytext=(35, 35), textcoords='offset points',
             arrowprops={'arrowstyle': '->', 'color': 'r', 'lw': 1})
plt.savefig(png_path + '/random_forest_pruning.png')
plt.show()

#
# 특징 중요도
#

# 최적 모델 적합
clf = RandomForestClassifier(random_state=123, n_estimators=7,
                             min_samples_leaf=3, max_features=10, n_jobs=-1)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
# 0.89240972733972

# 중요도 벡터
fimp = clf.feature_importances_

# 중요도 벡터의 오름 차순 정렬 후 상위 20개 및 가장 중요한 변수의 중요도 값으로 나눔
ix = np.argsort(fimp)[::-1][:20]
imp_values = fimp[ix]/fimp[np.argmax(fimp)]

# 중요도 벡터의 각 요소 이름
imp_names = np.array(feature_names)[ix]

# 중요도 막대 그래프
y_pos = np.arange(len(imp_names))
fig = plt.figure(figsize=(6,6))
fig.subplots_adjust(left=0.3)
ax = fig.add_subplot(1,1,1)
ax.barh(y_pos, imp_values, align='center', color='orange', ecolor='black', tick_label=imp_names)
ax.set_xlim(0,1)
ax.yaxis.label.set_size(40)
ax.set_yticks(y_pos)
ax.set_yticklabels(imp_names)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('중요도')
ax.set_title('상위 20개 중요한 입력 변수')
plt.savefig(png_path + '/random_forest_feature_importance.png')
plt.show()

#
# 모델 평가: ROC 그래프
#
# 최적 모델 적합
clf = RandomForestClassifier(random_state=123, n_estimators=7,
                             min_samples_leaf=3, max_features=10, n_jobs=-1)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
# 0.9086219602063376

# 모델에 의한 예측 확률 계산
y_pred_proba = clf.predict_proba(X_test)[::, 1]

# fpr: 1-특이도, tpr: 민감도, auc 계산
fpr, tpr, _ = metrics.roc_curve(y_true=y_test,  y_score=y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)

# ROC 그래프 출력
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label="랜덤 포레스트\n곡선밑 면적(AUC)=" + "%.4f" % auc)
plt.plot([-0.02, 1.02], [-0.02, 1.02], color='gray', linestyle=':', label='무작위 모델')
plt.margins(0)
plt.legend(loc=4)
plt.xlabel('fpr: 1-Specificity')
plt.ylabel('tpr: Sensitivity')
plt.title("ROC Curve", weight='bold')
plt.legend()
plt.savefig(png_path + '/random_forest_ROC.png')
plt.show()
