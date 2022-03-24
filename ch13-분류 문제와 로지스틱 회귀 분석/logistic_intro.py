#
# 프로그램 이름: logistic_intro.py
# 작성자: Bong Ju Kang
# 설명: 로지스틱 회귀 분석을 통한 분류 문제 이해하기
#

# 필요한 패키지
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests, zipfile, io
from sklearn.linear_model import LinearRegression, LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split

# 초기 설정
png_path = "./data/png"
os.makedirs(png_path, exist_ok=True)

# 한글출력
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 가져오기
path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00222/'
zip_url = path + 'bank.zip'

z = zipfile.ZipFile(io.BytesIO(requests.get(zip_url).content)) # 짚 파일 풀기
z.infolist() # 짚 파일 내의 구성 요소 보기
df = pd.read_csv(z.open('bank.csv'),sep=';') # 특정 요소 가져오기
df.columns

# 데이터 속성
# Input variables:
#    # bank client data:
#    1 - age (numeric)
#    2 - job : type of job (categorical: "admin.","unknown","unemployed","management","housemaid","entrepreneur","student",
#                                        "blue-collar","self-employed","retired","technician","services")
#    3 - marital : marital status (categorical: "married","divorced","single"; note: "divorced" means divorced or widowed)
#    4 - education (categorical: "unknown","secondary","primary","tertiary")
#    5 - default: has credit in default? (binary: "yes","no")
#    6 - balance: average yearly balance, in euros (numeric)
#    7 - housing: has housing loan? (binary: "yes","no")
#    8 - loan: has personal loan? (binary: "yes","no")
#    # related with the last contact of the current campaign:
#    9 - contact: contact communication type (categorical: "unknown","telephone","cellular")
#   10 - day: last contact day of the month (numeric)
#   11 - month: last contact month of year (categorical: "jan", "feb", "mar", ..., "nov", "dec")
#   12 - duration: last contact duration, in seconds (numeric)
#    # other attributes:
#   13 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
#   14 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric, -1 means client was not previously contacted)
#   15 - previous: number of contacts performed before this campaign and for this client (numeric)
#   16 - poutcome: outcome of the previous marketing campaign (categorical: "unknown","other","failure","success")
#
#   Output variable (desired target):
#   17 - y - has the client subscribed a term deposit? (binary: "yes","no")

df.info()
df.y.value_counts() # 목표 변수 분포 확인
# no     4000
# yes     521
# Name: y, dtype: int64

# 문자 변수를 숫자 변수로 치환하기
df['num_y'] = pd.get_dummies(df['y'], drop_first=True)

# 변수 정의
x = df['duration']
y = df['num_y']

#
# 선형 회귀 모형 적합
#
# 접촉 시간(duration) vs 정기예금 가입여부(num_y) 산점도
plt.figure(figsize=(6,4))
plt.scatter(x,y, s=1, label="산점도")
plt.xlabel("접촉시간")
plt.ylabel('정기예금가입여부')
plt.title('접촉시간 vs. 정기예금가입여부 산점도 및 선형회귀 직선')
# 선형 회귀 모형
regmodel = LinearRegression(fit_intercept=True)
regmodel.fit(x.values.reshape((-1,1)), y.values.reshape((-1,1)))
h = regmodel.predict(x.values.reshape((-1,1)))
plt.plot(x,h, color='orange', label="선형회귀")
plt.ylim((-0.1,1.1))
plt.legend()
plt.savefig(png_path + '/logistic_linearfit.png')
plt.show()


#
# 로지스틱 함수의 적합
#

# 로지스틱 함수의 모양 살펴보기
z = np.linspace(-5,5,100)
p_z = 1/(1+np.exp(-z))
plt.figure(figsize=(6,4))
plt.plot(z,p_z, color='black', label='logistic function')
plt.xlabel('z')
plt.ylabel('p(z)')
plt.legend()
plt.savefig(png_path + '/logistic_logisticCurve.png')
plt.show()

# 로지스틱 함수 적합
logisticModel = LogisticRegression(random_state=123)
logisticModel.fit(x,y)

# 적합 결과
logisticModel.coef_
logisticModel.intercept_

# 예측
logisticModel.predict_proba(x).shape
df['num_y'].value_counts()
predicted = logisticModel.predict(x)
prob = logisticModel.predict_proba(x)[:,1] # 'y' 확률
np.unique(predicted, return_counts=True)
score = logisticModel.score(x,y) # 정분류율
# 0.8882990488829905
# confusion matrix (분류 결과표)
metrics.confusion_matrix(y, predicted)
# array([[3913,   87],
#        [ 350,  171]], dtype=int64)

# 적합된 결과 그래프
plt.figure(figsize=(6,4))
plt.scatter(x,y, s=1)
plt.xlabel("접촉시간")
plt.ylabel('정기예금가입여부')
plt.title('접촉시간 vs. 정기예금가입여부 산점도 및 로지스틱회귀 곡선')
plt.scatter(x,prob, color='orange', label="로지스틱 회귀", s=1)
plt.axhline(y=0.5, color='red', label='결정선')
plt.ylim((-0.1,1.1))
plt.legend(loc=(0.05, 0.75))
plt.savefig(png_path + '/logistic_scatterWithlogisticCurve.png')
plt.show()

#
# 비용 함수의 의미
#
h = np.linspace(0.00001,1, 1000)
loss = -np.log(h)
plt.figure(figsize=(6,4))
plt.scatter(h, loss, s=1)
plt.xlabel("h")
plt.ylabel('cost=-log(h)')
plt.title('h vs. cost Scatter Plot with y=1')
plt.tight_layout()
plt.savefig(png_path + '/logistic_costFunction.png')
plt.show()

#
# ROC 그래프 알고리즘에 대한 이해
#

# 데이터 구성
# 예측 확률
y_score = np.array([0.3, 0.4, 0.55, 0.75,  0.97])
# 실제 목표 값
y_true = np.array([0, 1, 0, 1, 1 ])

# 예측 확률에 대한 내림 차순 정렬한 인덱스 값
ix = np.argsort(y_score)[::-1]

# 내림 차순 정렬된 목표 값의 순차적으로 누적
fps = np.cumsum(y_true[ix] == 0)
tps = np.cumsum(y_true[ix] == 1)

# (0, 0) 부터 시작하기 위하여 0 값 추가
tps = np.r_[0, tps]
fps = np.r_[0, fps]

# 전체 이벤트 개수와 비 이벤트 개수로 나눔
fpr = fps / fps[-1]
tpr = tps / tps[-1]

# fpr, tpr를 이용한 그래프
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr)
plt.plot([-0.02, 1.02], [-0.02, 1.02], color='gray', linestyle=':') # 무작위 모델
plt.margins(0) # 실제 데이터 그림과 축간의 간격
plt.xlabel('fpr: 1-Specificity')
plt.ylabel('tpr: Sensitivity')
plt.title("ROC Curve", weight='bold')
plt.savefig(png_path + '/logistic_ROC_scratch.png')
plt.show()

#
# 예제: [BANK] 데이터의 로지스틱 회귀 적합
#

# 데이터 가져오기
path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00222/'
zip_url = path + 'bank.zip'

z = zipfile.ZipFile(io.BytesIO(requests.get(zip_url).content)) # 짚 파일 풀기
z.infolist() # 짚 파일 내의 구성 요소 보기
df = pd.read_csv(z.open('bank.csv'),sep=';') # 특정 요소 가져오기
df.columns

# 가변수 구성을 위한 get_dummies 이해하기
pd.get_dummies([0,1,0,1,2])
#    0  1  2
# 0  1  0  0
# 1  0  1  0
# 2  1  0  0
# 3  0  1  0
# 4  0  0  1

pd.get_dummies([0,1,0,1,2], drop_first=True)
#    1  2
# 0  0  0
# 1  1  0
# 2  0  0
# 3  1  0
# 4  0  1

# 문자 변수를 숫자 변수로 치환하기
df['num_y'] = pd.get_dummies(df['y'], drop_first=True)

# 범주형 변수명 가져오기
categorical_vars = df.drop(['y', 'num_y'], axis=1).columns[df.drop(['y', 'num_y'], axis=1).dtypes == 'object']

# 숫자형 변수명 가져오기
num_vars = df.drop(['y', 'num_y'], axis=1).columns[df.drop(['y', 'num_y'], axis=1).dtypes != 'object']

# 범주형 변수에 대한 가변수 구성하기
dumm_data = pd.get_dummies(df[categorical_vars], prefix_sep='_', drop_first=True)

# 가변수와 숫자형 변수만을 이용한 입력 특징 데이터 구성하기
Xdf = df.join(dumm_data)[num_vars.tolist() + dumm_data.columns.tolist()]
X = Xdf.values

# 목표 변수 구성하기
y = df['num_y'].values

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# 로지스틱 함수 적합
logisticModel = LogisticRegression(random_state=123)
logisticModel.fit(X_train, y_train)

# 적합 결과
logisticModel.coef_
logisticModel.intercept_

# 예측
logisticModel.predict(X_test)
logisticModel.predict_proba(X_test)
df['num_y'].value_counts()
predicted = logisticModel.predict(X_test)
pd.Series(predicted).value_counts()
score = logisticModel.score(X_test, y_test)  # return mean accuracy, 정분류율 반환
# 0.8983050847457628
# prob = logisticModel.predict_proba(X_test)[:, 1]

# confusion matrix (분류 결과표)
metrics.confusion_matrix(y_test, predicted)
# array([[1171,   28],
#        [ 110,   48]], dtype=int64)

#
# ROC 그래프 그리기
#
# 모델에 의한 예측 확률 계산
y_pred_proba = logisticModel.predict_proba(X_test)[::, 1]

# fpr: 1-특이도, tpr: 민감도, auc 계산
fpr, tpr, _ = metrics.roc_curve(y_true=y_test,  y_score=y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)

# ROC 그래프 생성
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label="로지스틱 회귀\n곡선밑 면적(AUC)=" + "%.4f" % auc)
plt.plot([-0.02, 1.02], [-0.02, 1.02], color='gray', linestyle=':', label='무작위 모델')
plt.margins(0)
plt.legend(loc=4)
plt.xlabel('fpr: 1-Specificity')
plt.ylabel('tpr: Sensitivity')
# plt.axhline(y=0.7, color='red', label='민감도 기준선')
# plt.axvline(x=0.2, color='green', label='1-특이도 기준선')
plt.title("ROC Curve", weight='bold')
plt.legend()
plt.savefig(png_path + '/logistic_ROC2.png')
plt.show()




