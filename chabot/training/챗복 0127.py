import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import warnings

mpl.rc('font', family='NanumGothic') # 폰트 설정
mpl.rc('axes', unicode_minus=False) # 유니코드에서 음수 부호 설정

# 차트 스타일 설정
sns.set(font="NanumGothic", rc={"axes.unicode_minus":False}, style='darkgrid')
plt.rc("figure", figsize=(10,8))

warnings.filterwarnings("ignore")

# 원본 행렬 R(4 x 5) 생성
R = np.array([[4, np.NaN, np.NaN, 2, np.NaN ],
              [np.NaN, 5, np.NaN, 3, 1 ],
              [np.NaN, np.NaN, 3, 4, 4 ],
              [5, 2, 1, 2, np.NaN ]])

print("원본 행렬 R Shape:", R.shape)

# 잠재요인 차원 K는 3으로 가정
K=3
num_users, num_items = R.shape

# 임의의 P(4 x 3), Q(5 x 3) 생성
np.random.seed(1)
P = np.random.normal(scale=1./K, size=(num_users, K))
Q = np.random.normal(scale=1./K, size=(num_items, K))

print("원본 행렬 P Shape:", P.shape)
print("원본 행렬 Q Shape:", Q.shape)

from sklearn.metrics import mean_squared_error


def get_rmse(R, P, Q, not_nan_index):
    error = 0

    # 예측 R 행렬 생성
    full_pred_matrix = P @ Q.T

    # Null이 아닌 실제 R 행렬과 예측 행렬
    R_not_null = R[not_nan_index]
    full_pred_matrix_not_null = full_pred_matrix[not_nan_index]

    # RMSE 계산
    mse = mean_squared_error(R_not_null, full_pred_matrix_not_null)
    rmse = np.sqrt(mse)

    return rmse


# 실제 R 행렬에서 Null이 아닌 index
not_nan_index = np.where(np.isnan(R) == False)

# 반복수, 학습률, L2 규제
steps = 1000
learning_rate = 0.01
r_lambda = 0.01

# SGD 기법으로 P, Q 업데이트
for step in range(steps):

    # Null이 아닌 행 index, 열 index, 값
    for u, i, r in zip(not_nan_index[0], not_nan_index[1], R[not_nan_index]):
        # 실제 값과 예측 값의 차이인 오류 값 구함
        r_hat_ui = P[u, :] @ Q[i, :].T
        e_ui = r - r_hat_ui

        # SGD 업데이트 공식
        P[u, :] = P[u, :] + learning_rate * (e_ui * Q[i, :] - r_lambda * P[u, :])
        Q[i, :] = Q[i, :] + learning_rate * (e_ui * P[u, :] - r_lambda * Q[i, :])

    rmse = get_rmse(R, P, Q, not_nan_index)

    if ((step + 1) % 50) == 0:
        print("### iteration step: ", step + 1, " rmse: ", np.round(rmse, 3))

pred_matrix = P @ Q.T
print('예측 행렬:')
print(np.round(pred_matrix, 3))

print("-"*35)

print('실제 행렬:')
print(R)


def matrix_factorization(R, K, steps=200, learning_rate=0.01, r_lambda=0.01):
    num_users, num_items = R.shape

    # 임의의 P(4 x K), Q(5 x K) 생성
    np.random.seed(1)
    P = np.random.normal(scale=1. / K, size=(num_users, K))
    Q = np.random.normal(scale=1. / K, size=(num_items, K))

    break_count = 0

    # 실제 R 행렬에서 Null이 아닌 index
    not_nan_index = np.where(np.isnan(R) == False)

    # SGD 기법으로 P, Q 업데이트
    for step in range(steps):

        # Null이 아닌 행 index, 열 index, 값
        for u, i, r in zip(not_nan_index[0], not_nan_index[1], R[not_nan_index]):
            # 실제 값과 예측 값의 차이인 오류 값 구함
            r_hat_ui = P[u, :] @ Q[i, :].T
            e_ui = r - r_hat_ui

            # SGD 업데이트 공식
            P[u, :] = P[u, :] + learning_rate * (e_ui * Q[i, :] - r_lambda * P[u, :])
            Q[i, :] = Q[i, :] + learning_rate * (e_ui * P[u, :] - r_lambda * Q[i, :])

        rmse = get_rmse(R, P, Q, not_nan_index)

        if ((step + 1) % 10) == 0:
            print("### iteration step: ", step + 1, " rmse: ", np.round(rmse, 3))

    return P, Q

# Grouplens MovieLens 데이터
movies = pd.read_csv('./ml-latest-small/movies.csv')
ratings = pd.read_csv('./ml-latest-small/ratings.csv')

# ratings 데이터와 movies 데이터 결합
rating_movies = pd.merge(ratings, movies, on="movieId")

# 사용자-아이템 평점 행렬 생성
ratings_matrix = rating_movies.pivot_table("rating", "userId", "title")

ratings_matrix.head(3)

# 예측 행렬 계산
P, Q = matrix_factorization(ratings_matrix.values, K=50, steps=200, learning_rate=0.01, r_lambda = 0.01)
pred_matrix = P @ Q.T


# 아직 보지 않은 영화 리스트 함수
def get_unseen_movies(ratings_matrix, userId):
    # user_rating: userId의 아이템 평점 정보 (시리즈 형태: title을 index로 가진다.)
    user_rating = ratings_matrix.loc[userId, :]

    # user_rating이 notnull인 리스트
    unseen_movie_list = user_rating[user_rating.isnull()].index.tolist()

    # 모든 영화명을 list 객체로 만듬.
    movies_list = ratings_matrix.columns.tolist()

    # 한줄 for + if문으로 안본 영화 리스트 생성
    unseen_list = [movie for movie in movies_list if movie in unseen_movie_list]

    return unseen_list


# 보지 않은 영화 중 예측 높은 순서로 시리즈 반환
def recomm_movie_by_userid(pred_df, userId, unseen_list, top_n=10):
    recomm_movies = pred_df.loc[userId, unseen_list].sort_values(ascending=False)[:top_n]

    return recomm_movies

# 아직 보지 않은 영화 리스트
unseen_list = get_unseen_movies(ratings_matrix, 9)

# 아이템 기반의 인접 이웃 협업 필터링으로 영화 추천
recomm_movies = recomm_movie_by_userid(ratings_pred_matrix, 9, unseen_list, top_n=10)

# 데이터 프레임 생성
recomm_movies = pd.DataFrame(data=recomm_movies.values,index=recomm_movies.index,columns=['pred_score'])
recomm_movies