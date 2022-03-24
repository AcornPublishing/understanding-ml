#
# 프로그램 이름: matplotlib_basics.py
# 작성자: Bong Ju Kang
# 설명: matplotlib의 기본을 예제와 함께 이해하기
#

# 필요한 패키지
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 초기 설정
png_path = "./data/png"
os.makedirs(png_path, exist_ok=True)

#
# 데이터 준비
#
x = np.arange(100)
x2 = pd.date_range('2004-11-03', periods=100)
y = np.random.RandomState(123).normal(0, 1, 100)
y2 = np.random.RandomState(123).uniform(0, 1, 100)

img_data =[
      [0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 1, 1, 1, 1, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 1, 0, 0],
      [0, 0, 0, 0, 0, 0, 1, 0, 0],
      [0, 1, 1, 1, 1, 1, 0, 0, 0],
      [0, 0, 1, 1, 1, 1, 1, 0, 0],
      [0, 0, 0, 0, 0, 0, 1, 0, 0],
      [0, 0, 0, 1, 1, 1, 0, 0, 0],
      [0, 0, 1, 1, 1, 0, 0, 0, 0],
      ]
#
# 그래프 준비
#

plt.rcParams # 디폴트 세팅 확인

fig = plt.figure(figsize=(6,4)) # 그래프 크기 설정: 인치 단위의 폭과 높이
ax = fig.add_subplot(2, 2, 4) # 그래프가 있어야 할 축: 행, 열 그리고 위치
plt.savefig(png_path + '/pyplot_subplot.png')

#
# 그래프 생성
#
fig = plt.figure()

ax1 = fig.add_subplot(2,2,1)
ax1.plot(x, y) # 디폴트 그림

ax2 = fig.add_subplot(2,2,2)
ax2.plot(x, y, color='green', marker='o', linestyle='dashed',
        linewidth=2, markersize=3) # 디폴트 값 변경

ax3 = fig.add_subplot(2,2,3)
ax3.scatter(x, y) # 디폴트 그림

ax4 = fig.add_subplot(2,2,4)
ax4.scatter(x, y, c='orange', s=1) # 디폴트 값 변경
plt.savefig(png_path + '/pyplot_axis.png')

plt.figure(figsize=(5,5)) # 그림 크기만 변경
plt.imshow(img_data, cmap='gray_r') # 이미지 데이터에 대한 그림
plt.savefig(png_path + '/pyplot_img.png')
plt.show()

#
# 그래프 수정
#
fig = plt.figure(figsize=(12,6))

ax = fig.add_subplot(1, 2, 1)
ax = plt.plot(x, y) # 디폴트 그래프
ax = plt.plot(x, y2) # 디폴트 그래프

ax2 = fig.add_subplot(1, 2, 2)
ax2 = plt.plot(x, y, marker='o', ms = 3, label='x vs y') # 표식 기호, 크기, 그래프 이름 수정
ax2 = plt.plot(x, y2, c='red', ls='--',label='x vs y2') # 색깔, 선 스타일, 그래프 이름 수정
ax2 = plt.title(r'y with $\sigma=1$', fontsize=10) # 제목 주기: 수학 기호 또는 식 쓰기
ax2 = plt.xlabel('x') # x 축에 레이블 주기
ax2 = plt.ylabel('y and y2') # y 축에 레이블 주기
ax2 = plt.text(np.mean(x), np.mean(y),
               'This is the (mean(x), mean(y)) point', size=10) # 특정 위치에 문자 쓰기
ax2 = plt.legend(loc='best') # 범례 보여주기
fig.savefig(png_path + '/pyplot_customization.png', transparent=True)
fig.show()
plt.close(fig)








