#
# 프로그램 이름: python_basics.py
# 작성자: Bong Ju Kang
# 설명: 파이썬 언어의 기본을 예제와 함께 이해하기
#
import os

#
# 기본 데이터 형
#
x = 100
print(type(x))

x = 0.0
print(type(x))

x = True
print(type(x))

x = 'hello!\nWelcome to ML world'
print(x)
print(type(x))

x = b'bong'
print(type(x))

#
# 컨테이너 데이터 형
#
# list
x = [1,2,3]
print(type(x))

x = ["a",2.3,3, b"bong"]
print(type(x))
print(type(x[0]))
print(type(x[1]))
print(type(x[2]))
print(type(x[3]))

x=["bong"]
print(type(x))
print(type(x[0]))

# tuple
x = (1,2,3)
print(type(x))

x = 1,2,3
print(type(x))

x = "a",1.1,2
print(type(x))
print(type(x[0]))

x = "bong",
print(type(x))

# dict
x = {"key1": 100, "key2": 200}
print(type(x))

x = dict(key1=100, key2=200)
print(type(x))

# set
x = {"key1", "key2"}
print(type(x))

x = {1, 2, 3, 3}
print(x)
print(type(x))

#
# 값의 할당
#
x = 1 + 1.2
x = y = 0 # 같은 값 할당
x, y = 1, 0.9 # 각각 해당 값 할당
x, y = y, x # 값 바꾸기

x = 0
x += 2 # x = x + 2
x -= 3 # x = x - 3
x = None
print(type(x))

del x # x 변수 제거

#
# 형 변환
#
print(type("100"))
print(type(int("100"))) # str -> int
print(int("ff", 16)) # str -> int
print(int(2.1)) # float -> int
print(float(0)) # int -> float
print(round(13.2)) # float -> int

# 다양한 bool 형 변환
print(type(bool(None))) # NoneType -> bool
print(bool(0))
print(bool(1))
print(bool(False))
print(bool(True))
print(bool(""))

print(type(str(100))) # int -> str
print(type(str(100.0))) # float -> str

print(type(chr(97))) # int -> str
print(type(ord('\n'))) # str -> int

print(type(bytes([97,98,99,100]))) # list -> bytes

print(type(list("abc"))) # str -> list
print(type(dict([(1, "abc"), (2, "cde")]))) # list -> dict

print(type(set("abcde"))) # str -> set
x = "-".join(['bong', 'ju', 'kang']) # list -> str
print(type(x))

x.split("-") # str -> list

#
# 열 컨테이너 인덱싱
#

# 인덱싱
x = [2, 4, 6, 8, 10]

len(x) # 5
x[0] # 2
x[4] # 10
x[5] # IndexError: list index out of range
x[-1] # 10
x[-len(x)] # 2

# 슬라이싱
x = [2, 4, 6, 8, 10]
x[1:2] # [4]
x[::] # [2, 4, 6, 8, 10]
x[:] # [2, 4, 6, 8, 10]
x[:3] # [2, 4, 6]
x[:-1] # [2, 4, 6, 8]
x[1:3] # [4, 6]
x[::2] # [2, 6, 10]
x[::-2] # [10, 6, 2]
del x[:2] # [6, 8, 10]

#
# 논리 연산 (boolean logic)
#
0 > 1 # False
0 == 0 # True

(0 > 1) and (0 > 1)  # false
(0 > 1) & (0 > 1)  # false, and=&
(0 > 1) or (0 == 0)  # True
(0 > 1) | (0 == 0)  # True or=|

not (0 > 1) # True


#
# 문의 구성
#
if (0 == 0):
    print((0 == 0))
    if not (0 > 1):
        print (not (0 > 1))
print((0 > 1) and (0 > 1))
# 결과: True, True, False

#
# 모듈 가져오기
#
import numpy as np # numpy의 별명으로 np를 사용
import matplotlib.pyplot as plt # 파이썬 경로의 matplotlib 디렉토리 밑의 pyplot 파일을 지칭
np.sin(np.pi/2)
dfx = np.linspace(0, np.pi)
dfy = np.sin(dfx)
plt.plot(dfx, dfy)

#
# 조건문
#
income=100
if income < 100:
    grade = "low"
elif income < 200:
    grade = "middle"
else:
    grade = "high"
print(grade)

#
# 수학 함수
#
import numpy as np

3 // 2 # 1
-3 // -2 # 1
3 // -2 # -2

# 행렬 곱: @
x = np.matrix([[1,2], [3,4]])
y = np.matrix([[0,1], [1,0]])
print(x @ y)
#  [[2 1]
#  [4 3]]
np.matmul(x, y) # 행렬 곱
np.multiply(x, y) # 원소별 곱(elementwise product)

# 수학 함수
np.sin(np.pi/2) # 1.0
np.sqrt(100) # 10.0
np.log(np.e) # e에 대한 자연 로그
np.log([1, np.e, np.e**2, 0]) # [  0.,   1.,   2., -inf]
np.ceil(9.9) # 10.0
np.floor(9.9) # 9.0

#
# 조건 반복문 (conditional loop statement): while, for
#
value = 0
index = 1
while index <= 10:
    value += index**2
    index += 1
print("value=", value, ",", "index=", index)
# value= 385 , index= 11
# index 값이 11이면 while 반복 문을 탈출 따라서 합은 index=10 까지만의 합임

str = "bong ju kang"
index = 0
for value in str:
    if value == 'g':
        index += 1
print("value=", value, "has found", index, "occurences")
# value= g has found 2 occurences

# range 함수를 이용한 for
str = "bong ju kang"
range(len(str)) # 12
str[len(str)] # IndexError: string index out of range
str[len(str)-1] # 'g'
[index for index in range(len(str))] # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

index = 0
for index in range(len(str)):
    value = str[index]
    if index > 10:
        print("value\'s index > 10 is", value) # value's index > 10 is g
print('exit index = ', index) # exit index =  11

# enumerate 함수를 이용한 값과 위치 찾아오기
[a for a, b in enumerate(str)] # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
[b for a, b in enumerate(str)] # ['b', 'o', 'n', 'g', ' ', 'j', 'u', ' ', 'k', 'a', 'n', 'g']

#
# 컨테이너 형에 대한 일반 연산
#
str = "bong ju kang"
value = "123456789"

len(str) # 12
len(value) # 9
min(value) # '1'
max(value) # '9'
max(str) # 'u'
sorted(str) # [' ', ' ', 'a', 'b', 'g', 'g', 'j', 'k', 'n', 'n', 'o', 'u']
x = 'b'
x in str # True
[{a:b} for a, b in enumerate(str)] # 위치(index)와 값(value)을 반환하는 반복자(iterator)
# [{0: 'b'},
#  {1: 'o'},
#  {2: 'n'},
#  {3: 'g'},
#  {4: ' '},
#  {5: 'j'},
#  {6: 'u'},
#  {7: ' '},
#  {8: 'k'},
#  {9: 'a'},
#  {10: 'n'},
#  {11: 'g'}]
[val for val in zip(('a', 'b', 'c'), ('c', 'b', 'd', 'e'))] # 같은 위치에 있는 값을 반환하는 반복자
[val for val in zip(str, value)]
# [('b', '1'),
#  ('o', '2'),
#  ('n', '3'),
#  ('g', '4'),
#  (' ', '5'),
#  ('j', '6'),
#  ('u', '7'),
#  (' ', '8'),
#  ('k', '9')]

[val for val in reversed(value)] # ['9', '8', '7', '6', '5', '4', '3', '2', '1']
2*value # 반복
str + "," + value # 연결
value.index('8') # 위치 찾기
str.count('g') # 빈도 구하기

#
# 리스트 연산
#
str = "bong ju kang"
value = list(str) # str -> list 형 변환
value.append(', banene') # 해당 값을 하나의 요소로 추가
# ['b', 'o', 'n', 'g', ' ', 'j', 'u', ' ', 'k', 'a', 'n', 'g', ', banene']
value.remove(', banene') # 해당 값을 제거
# ['b', 'o', 'n', 'g', ' ', 'j', 'u', ' ', 'k', 'a', 'n', 'g']
value.extend(', banene') # 해당 열을 추가
# ['b', 'o', 'n', 'g', ' ', 'j', 'u', ' ', 'k', 'a', 'n', 'g', ',', ' ', 'b', 'a', 'n', 'e', 'n', 'e']
value.pop() # 마지막 값을 제거
value.pop(0) # 해당 위치의 값을 제거
value.sort() # 오름 차순으로 정렬
value.reverse() # 역으로 정렬

#
# 딕셔너리(dictionary) 연산
#
dic = {'lang':80, 'math':90, 'his':75} # 딕셔너리 생성
dic['lang'] # 키를 이용한 딕셔너리 값 조회
dic.get("lang") # get 방법을 이용한 값 조회
dic['math'] = 100 # 키를 이용한 값 수정
dic.update({'math':100}) # update 방법을 이용한 값 수정
dic.update(dict(math=90)) # dict 함수와 update 방법을 이용한 값 수정
del dic['lang'] # 해당 키 제거 (값 포함)
dic.clear() # 딕셔너리 초기화

dic = {'lang':80, 'math':90, 'his':75} # 딕셔너리 생성
dic.keys() # 키 보기
# dict_keys(['lang', 'math', 'his'])
dic.values() # 키 순서대로 값을 보기
# dict_values([80, 90, 75])
dic.items() # 키와 값
# dict_items([('lang', 80), ('math', 90), ('his', 75)])
last_item = dic.popitem() # 마지막 항목을 제거 (키와 값을 동시에 제거)
print(last_item) # ('his', 75)

#
# 집합(set) 연산
#
s1 = {'a', 'b', 'c'}
s2 = {'b', 'c', 'd', 'e'}
s_union = s1 | s2 # 합집합
s_intersect = s1 & s2 #  교집합
s_difference = s1 - s2 # 차집합

s1 < s2 # False
s1 == s2 # False
s1 < {'a', 'b', 'c', 'd'} # True

s1.add('f') # 항목 추가
s1.update({'k'}) # 원래 있던 집합에 해당 집합을 추가
s1.remove('k') # 항목 제거
new_s = s1.copy() # 집합 복사

#
# 함수 정의
#

import numpy as np

def bnn_add(x,y):
    """
    이 함수는 피 연산자를 받아서 덧셈을 처리해준다.
    :param x: 덧셈이 가능한 값
    :param y: 덧셈이 가능한 값
    :return: 덧셈 후 결과 값을 반환
    """
    return x + y

print(bnn_add(1.0, 3)) # 4.0
retvalue = bnn_add(np.array([1,9]), np.array([2.0, 3]))
print(retvalue) # [ 3. 12.]

def vclass(x):
    if x in ['Hybrid','SUV','Sedan','Wagon']:
        return 'Family Vehicle'
    else:
        return 'Truck or Sports Vehicle'
print(vclass('Hyundai')) # Truck or Sports Vehicle

#
# 문자열(string) 연산
#
str='\n    bong ju kang'
print(str)
str.strip() # 문자열의 앞과 뒤의 화이트 스페이스(white space) 제거, 'bong ju kang'
# 화이트스페이스의 종류
# chr(9) # 수평 탭(tab), '\t'
# chr(10) # 라인피드(line feed), '\n'
# chr(11) # 수직 탭, '\x0b'
# chr(12) # 폼피드(form feed), '\x0c'
# chr(13) # 캐리지리턴(carriage return), '\r'
# chr(32) # 공백(space), ' '

str = 'bong ju kang'
str.index('ka') # 부분 문자열의 시작하는 위치 값을 반환, 8

str.upper() # 대문자로 변환 'BONG JU KANG'
str.lower() # 소문자로 변환
'bong Ju'.swapcase() # 반대로 변환, 'BONG jU'
'bong ju'.capitalize() # 첫문자만 대문자로 변환, 'Bong ju'

str.isalnum() # 숫자와 문자로만 구성되어 있는 문자열인지 확인, False
str.title() # 제목 형식으로 변환, 'Bong Ju Kang'

'bong ju'.split(' ') # 공백으로 분리, ['bong', 'ju']
'-'.join('bong') # bong을 반복하여 '-' 문자와 연결, 'b-o-n-g'
'-'.join(['bong', 'ju', 'kang']) # 'bong-ju-kang'

#
# 포맷 구성하기(formatting)
#
"${:.2f}".format(12345) # '$12345.00'
"${0:.2f}".format(12345) # '$12345.00'
"{:,.1f}원".format(12345.77) # ''12,345.8원'
"{1:>20s}".format(12345, 'bong ju kang') # '        bong ju kang'
"{1:^20s}".format(12345, 'bong ju kang') # '    bong ju kang    '

#
# 파일 읽고 쓰기
#
sample_data = """293,66,1,30,29,14,1,293,66,1,30,29,14,A,E,446,33,20,NA,A
315,81,7,24,38,39,14,3449,835,69,321,414,375,N,W,632,43,10,475,N
479,130,18,66,72,76,3,1624,457,63,224,266,263,A,W,880,82,14,480,A
496,141,20,65,78,37,11,5628,1575,225,828,838,354,N,E,200,11,3,500,N
321,87,10,39,42,30,2,396,101,12,48,46,33,N,E,805,40,4,91.5,N
594,169,4,74,51,35,11,4408,1133,19,501,336,194,A,W,282,421,25,750,A
185,37,1,23,8,21,2,214,42,1,30,9,24,N,E,76,127,7,70,A
298,73,0,24,24,7,3,509,108,0,41,37,12,A,W,121,283,9,100,A
323,81,6,26,32,8,2,341,86,6,32,34,8,N,W,143,290,19,75,N"""

# 쓰기
filename = './data/sample_data.csv'
os.makedirs(os.path.dirname(filename), exist_ok=True)
f = open(filename, "w")
f.write(sample_data)
f.close()

# 읽기 1
f = open(filename, "r")
file_data = f.read()
f.close()

# 읽기 2
f = open(filename, "r")
file_data = ""
for line in f.readlines():
    print(type(line))
    file_data += line
f.close()

# 확인
print(file_data)

