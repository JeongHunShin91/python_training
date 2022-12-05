# (분류분석/시각화)
# 2. 위스콘신 유방암 데이터셋을 대상으로 분류기법 2개를 적용하여 기법별 결과를
# 비교하고 시각화하시오. (R과 python 버전으로 모두 실행)
# -종속변수는diagnosis: Benign(양성), Malignancy(악성)
# svm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC
from mlxtend.plotting import plot_decision_regions
import re

# 데이터전처리
dataset = pd.read_csv("./wdbc_data.csv")
dataset = dataset.drop(["id"],axis=1)
y = dataset["diagnosis"]
x = dataset.drop(["diagnosis"],axis=1)
print(x)
print(y)

# 학습데이터/ 검정데이터 분류
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size = 0.3)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# 스케일링
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# svm 훈련
svm_model = SVC(kernel='rbf', C=8, gamma=0.1)
svm_model.fit(X_train_std, y_train)

# 성능확인
y_pred = svm_model.predict(X_test_std)
print(np.mean(y_pred == y_test))

# 시각화
type(x)
x1=np.array(x)
type(x1)
y1=y == "B"
plt.scatter(x[["perimeter_worst"]],x[["points_mean"]],c=y1)
plt.show()

# ================================
#랜덤포레스트
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


import warnings
warnings.filterwarnings('ignore')

dataset = pd.read_csv("./wdbc_data.csv")
dataset = dataset.drop(["id"],axis=1)
y = dataset["diagnosis"]
x = dataset.drop(["diagnosis"],axis=1)
print(x)
print(y)

# 학습데이터/ 검정데이터 분류
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size = 0.3)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# 랜덤 포레스트 학습 및 별도의 테스트 세트로 예측 성능 평가
rf_clf = RandomForestClassifier(random_state=0)
rf_clf.fit(X_train, y_train)
pred = rf_clf.predict(X_test)
accuracy = accuracy_score(y_test, pred)
print('랜덤 포레스트 정확도: {:.4f}'.format(accuracy))

# 하이퍼 파라미터 튜닝
model = RandomForestClassifier()
model
params = { 'n_estimators' : [10, 100],
           'max_depth' : [6, 8, 10, 12],
           'min_samples_leaf' : [8, 12, 18],
           'min_samples_split' : [8, 16, 20]
            }

# RandomForestClassifier 객체 생성 후 GridSearchCV 수행
rf_clf = RandomForestClassifier(random_state = 0, n_jobs = -1)
grid_cv = GridSearchCV(rf_clf, param_grid = params, cv = 3, n_jobs = -1)
grid_cv.fit(X_train, y_train)

print('최적 하이퍼 파라미터: ', grid_cv.best_params_)
print('최고 예측 정확도: {:.4f}'.format(grid_cv.best_score_))

#위의 결과로 나온 최적 하이퍼 파라미터로 다시 모델을 학습하여 테스트 세트 데이터에서 예측 성능을 측정
rf_clf1 = RandomForestClassifier(n_estimators = 100,
                                max_depth = 12,
                                min_samples_leaf = 8,
                                min_samples_split = 8,
                                random_state = 0)
rf_clf1.fit(X_train, y_train)
pred = rf_clf1.predict(X_test)
print('예측 정확도: {:.4f}'.format(accuracy_score(y_test,pred)))

# 시각화
ftr_importances_values = rf_clf1.feature_importances_
ftr_importances = pd.Series(ftr_importances_values, index = X_train.columns)
ftr_top20 = ftr_importances.sort_values(ascending=False)[:20]
ftr_top20
plt.figure(figsize=(8,6))
plt.title('Top 20 Feature Importances')
sns.barplot(x=ftr_top20, y=ftr_top20.index)
plt.show()

# (예측기법/시각화)
# 3. mlbench패키지 내 BostonHousing 데이터셋을 대상으로 예측기법 2개를 적용하여
# 기법별 결과를 비교하고 시각화하시오. (R과 python 버전으로 모두 실행)
# -종속변수는MEDV 또는CMEDV를사용
# https://joyfuls.tistory.com/62
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing, load_boston
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import seaborn as sns
import math
import random

# boston_url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ST0151EN-SkillsNetwork/labs/boston_housing.csv'
# dataset=pd.read_csv(boston_url)
# dataset = dataset.drop(["Unnamed: 0"],axis=1)
# y = dataset["MEDV"]
# y = y.astype('int')
# x = dataset.drop(["MEDV"],axis=1)
boston = load_boston()
x = boston.data
y = boston.target

colnames = boston.feature_names # 13개 칼럼 이름 가져올때
colnames.head()
x.shape
y.shape
print(x)
print(y)

# 학습데이터/ 검정데이터 분류
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size = 0.3)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

# 랜덤 포레스트 학습 및 별도의 테스트 세트로 예측 성능 평가
rf_clf = RandomForestRegressor(n_estimators=400, min_samples_split=3)
rf_clf.fit(X_train, y_train)
pred = rf_clf.predict(X_test)
mse = mean_squared_error(y_test, pred)
rmse = math.sqrt(mse)
print('mse=', rmse)
print('R2=',r2_score(y_test, pred))

# 중요도
imp = rf_clf.feature_importances_
imp
len(imp)
random.seed(1234)
# 시각화
colnames
len(imp)
plt.barh(range(13), imp) # (x, y) # 중요도 (y에 얼마나 영향을 미치는지)
plt.yticks(range(13), colnames)

# 선형회귀
# https://byunghyun23.tistory.com/43
# 신경망 클래스의 정의

import pandas as pd
import numpy as np
import random
import tensorflow as tf
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import math

# 랜덤 시드 고정
SEED = 2021
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# 데이터 전처리
from sklearn import datasets

housing = datasets.load_boston()
X_data = housing.data
y_data = housing.target
housing.head()
# 피처 스케일링
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_data_scaled = scaler.fit_transform(X_data)

X_data_scaled[0]

# 학습 - 테스트 데이터셋 분할
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, shuffle=True, random_state=SEED)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

# MLP 모델 아키텍처 정의

from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense


def build_model(num_input=1):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=num_input))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


model = build_model(num_input=13)

# 미니 배치 학습
model.fit(X_train, y_train, epochs=200, batch_size=32, verbose=2)

# 모델 평가
model.evaluate(X_test, y_test)
pred = model.predict(X_test)
mse = mean_squared_error(y_test, pred)
rmse = math.sqrt(mse)
print('mse=', rmse)
print('R2=',r2_score(y_test, pred))

# 교차 검증
model = build_model(num_input=13)
history = model.fit(X_train, y_train, batch_size=32, epochs=200, validation_split=0.25, verbose=2)

# 시각화
import matplotlib.pyplot as plt


def plot_loss_curve(total_epoch=10, start=1):
    plt.figure(figsize=(15, 5))
    plt.plot(range(start, total_epoch + 1), history.history['loss'][start - 1:total_epoch], label='Train')
    plt.plot(range(start, total_epoch + 1), history.history['val_loss'][start - 1:total_epoch], label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('mse')
    plt.legend()
    plt.show()

plot_loss_curve(total_epoch=200, start=1)

plot_loss_curve(total_epoch=200, start=20)

# (데이터 시각화)
# 6. R의 ggplot2 패키지 내 함수와 python의 matplotlib 패키지 내 함수를 사용하여
# 막대 차트(가로, 세로), 누적막대 차트, 점 차트, 원형 차트, 상자 그래프, 히스토그램,
# 산점도, 중첩자료 시각화, 변수간의 비교 시각화, 밀도그래프를 수업자료pdf 내 데이터를
# 이용하여 각각 시각화하고 비교하시오.

import matplotlib as mpl
import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import seaborn as sns

iris = sns.load_dataset('iris') # 데이터
iris_mean = iris.groupby('species').mean() # 편균데이터

# 막대차트(세로)
iris_mean.plot(kind='barh')
iris_mean.plot(kind='bar')
plt.title("iris")
plt.ylabel("species")
plt.xlabel("cm")

# 누적막대 차트
iris_mean.plot(kind='bar',stacked = True)
plt.title("iris")
plt.xlabel("species")
plt.ylabel("cm")

# 점 차트
plt.scatter(iris_mean.index, iris_mean['sepal_length'])
plt.scatter(iris_mean.index, iris_mean['sepal_width'])
plt.scatter(iris_mean.index, iris_mean['petal_length'])
plt.scatter(iris_mean.index, iris_mean['petal_width'])
plt.legend(loc = (0.6, 0.7), labels = ['sepal_length', 'sepal_width', 'petal_length','petal_width'],title = 'species')

# 원형 차트
plt.pie(iris_mean['sepal_length'],labels = iris_mean.index,autopct = '%.1f%%', wedgeprops = {'width':0.7, 'edgecolor':'w', 'linewidth' : 3})
plt.legend(loc = (1, 0.25), title = 'species')
plt.pie(iris_mean['sepal_width'],labels = iris_mean.index,autopct = '%.1f%%',wedgeprops = {'width':0.7, 'edgecolor':'w', 'linewidth' : 3})
plt.legend(loc = (1, 0.25), title = 'species')
plt.pie(iris_mean['petal_length'],labels = iris_mean.index,autopct = '%.1f%%',wedgeprops = {'width':0.7, 'edgecolor':'w', 'linewidth' : 3})
plt.legend(loc = (1, 0.25), title = 'species')
plt.pie(iris_mean['petal_width'],labels = iris_mean.index,autopct = '%.1f%%',wedgeprops = {'width':0.7, 'edgecolor':'w', 'linewidth' : 3})
plt.legend(loc = (1, 0.25), title = 'species')

# 상자 그래프
data = [iris[iris['species']=="setosa"]['sepal_length'],
        iris[iris['species']=="versicolor"]['sepal_length'],
        iris[iris['species']=="virginica"]['sepal_length']]
plt.boxplot(data, labels=['setosa', 'versicolor', 'virginica'], showmeans=True, patch_artist=True)
data = [iris[iris['species']=="setosa"]['sepal_width'],
        iris[iris['species']=="versicolor"]['sepal_width'],
        iris[iris['species']=="virginica"]['sepal_width']]
plt.boxplot(data, labels=['setosa', 'versicolor', 'virginica'], showmeans=True, patch_artist=True)
data = [iris[iris['species']=="setosa"]['petal_length'],
        iris[iris['species']=="versicolor"]['petal_length'],
        iris[iris['species']=="virginica"]['petal_length']]
plt.boxplot(data, labels=['setosa', 'versicolor', 'virginica'], showmeans=True, patch_artist=True)
data = [iris[iris['species']=="setosa"]['petal_width'],
        iris[iris['species']=="versicolor"]['petal_width'],
        iris[iris['species']=="virginica"]['petal_width']]
plt.boxplot(data, labels=['setosa', 'versicolor', 'virginica'], showmeans=True, patch_artist=True)

# 히스토그램
plt.hist(iris['sepal_length'],bin=10)
plt.hist(iris['sepal_width'],bin=10)
plt.hist(iris['petal_length'],bin=10)
plt.hist(iris['petal_width'],bin=10)

groups = iris.groupby('species')
fig, ax = plt.subplots()
for name, group in groups:
    ax.plot(group.petal_length,
            group.petal_width,
            marker='o',
            linestyle='',
            label=name)
ax.legend(fontsize=12, loc='upper left') # legend position
plt.title('iris', fontsize=20)
plt.xlabel('Petal Length', fontsize=14)
plt.ylabel('Petal Width', fontsize=14)
plt.show()
str(iris)
groups = iris.groupby('species')
fig, ax = plt.subplots()
for name, group in groups:
    ax.plot(group.sepal_length,
            group.sepal_width,
            marker='o',
            linestyle='',
            label=name)
ax.legend(fontsize=12, loc='upper left') # legend position
plt.title('iris', fontsize=20)
plt.xlabel('sepal_length', fontsize=14)
plt.ylabel('sepal_width', fontsize=14)
plt.show()

# 중첩자료 시각화
iris_sepla = iris.groupby(['sepal_length','sepal_width'])
sepal = iris_sepla.size()
sepal = sepal.reset_index(name="count")

iris_peta = iris.groupby(['petal_length','petal_width'])
petal = iris_peta.size()
petal = petal.reset_index(name="count")

plt.scatter(sepal["sepal_length"],sepal["sepal_width"],s = sepal["count"]*100, c = sepal["count"] ,cmap='Wistia',alpha =0.8)
plt.colorbar()
plt.scatter(petal["petal_length"],petal["petal_width"],s = petal["count"]*100, c = petal["count"] ,cmap='Wistia',alpha =0.8)
plt.colorbar()

# 변수간의 비교 시각화, 밀도그래프
# iris
g = sns.pairplot(iris, hue="species")
