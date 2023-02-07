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

import surprise
print(surprise.__version__)

from surprise import Dataset
from surprise.model_selection import train_test_split

# ml-100k: 10만 개 평점 데이터
data = Dataset.load_builtin('ml-100k')

# surprise의 train_test_split() 사용
trainset, testset = train_test_split(data, test_size=0.25, random_state=0)
from surprise import SVD

# SVD를 이용한 잠재 요인 협업 필터링
algo = SVD()
algo.fit(trainset)

# 사용자 아이디(uid), 아이템 아이디(iid)는 문자열로 입력
uid = str(196)
iid = str(302)

# 추천 예측 평점 (.predict)
pred = algo.predict(uid, iid)
pred

# 추천 예측 평점 (.test)
predictions = algo.test( testset )

print('prediction type :',type(predictions), ' size:',len(predictions))
print('prediction 결과의 최초 5개 추출')

predictions[:5]

# 속성 확인
[ (pred.uid, pred.iid, pred.est, pred.details) for pred in predictions[:3] ]

from surprise import accuracy

# 성능 평가
accuracy.rmse(predictions)

# index와 header를 제거한 ratings_noh.csv 파일 생성
ratings = pd.read_csv('./ml-latest-small/ratings.csv')
ratings.to_csv('./ml-latest-small/ratings_noh.csv', index=False, header=False)

from surprise import Reader

# Reader 객체 생성
reader = Reader(line_format='user item rating timestamp', sep=',', rating_scale=(0.5, 5))

data = Dataset.load_from_file('./ml-latest-small/ratings_noh.csv', reader=reader)
data

# surprise의 train_test_split() 사용
trainset, testset = train_test_split(data, test_size=0.25, random_state=0)

# SVD를 이용한 잠재 요인 협업 필터링 (잠재 요인 크기 = 50)
algo = SVD(n_factors=50, random_state=0)
algo.fit(trainset)

# 추천 예측 평점 (.test)
predictions = algo.test( testset )

# 성능 평가
accuracy.rmse(predictions)

from surprise.model_selection import cross_validate

# 데이터 불러오기 (데이터 프레임)
ratings = pd.read_csv('./ml-latest-small/ratings.csv')

# Reader 객체 생성
reader = Reader(rating_scale=(0.5, 5.0))

# 사용자 아이디, 아이템 아이디, 평점 순서 (원래는 timestamp도 있으나 제외)
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# SVD를 이용한 잠재 요인 협업 필터링
algo = SVD(random_state=0)

# 교차 검증 수행
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# 데이터 불러오기 (데이터 프레임)
ratings = pd.read_csv('./ml-latest-small/ratings.csv')

# Reader 객체 생성
reader = Reader(rating_scale=(0.5, 5.0))

# 사용자 아이디, 아이템 아이디, 평점 순서 (원래는 timestamp도 있으나 제외)
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# surprise의 train_test_split() 사용
trainset, testset = train_test_split(data, test_size=.25, random_state=0)

# SVD를 이용한 잠재 요인 협업 필터링 (잠재 요인 크기 = 50)
algo = SVD(n_factors=50, random_state=0)
algo.fit(trainset)

# 추천 예측 평점 (.test)
predictions = algo.test( testset )

# 성능 평가
accuracy.rmse(predictions)

from surprise.model_selection import cross_validate

# 데이터 불러오기 (데이터 프레임)
ratings = pd.read_csv('./ml-latest-small/ratings.csv')

# Reader 객체 생성
reader = Reader(rating_scale=(0.5, 5.0))

# 사용자 아이디, 아이템 아이디, 평점 순서 (원래는 timestamp도 있으나 제외)
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# SVD를 이용한 잠재 요인 협업 필터링
algo = SVD(random_state=0)

# 교차 검증 수행
cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

from surprise.dataset import DatasetAutoFolds

# Reader 객체 생성
reader = Reader(line_format='user item rating timestamp', sep=',', rating_scale=(0.5, 5))

# DatasetAutoFolds 객체 생성
data_folds = DatasetAutoFolds(ratings_file='./ml-latest-small/ratings_noh.csv', reader=reader)

# 전체 데이터를 train으로 지정
trainset = data_folds.build_full_trainset()
trainset

# SVD를 이용한 잠재 요인 협업 필터링 (잠재 요인 크기 = 50)
algo = SVD(n_epochs=20, n_factors=50, random_state=0)
algo.fit(trainset)

# 사용자 아이디, 아이템 아이디 문자열로 입력
uid = str(9)
iid = str(42)

# 추천 예측 평점 (.predict)
pred = algo.predict(uid, iid, verbose=True)

# 영화에 대한 상세 속성 정보 DataFrame로딩
movies = pd.read_csv('./ml-latest-small/movies.csv')


# 아직 보지 않은 영화 리스트 함수
def get_unseen_surprise(ratings, movies, userId):
    # 특정 userId가 평점을 매긴 모든 영화 리스트
    seen_movies = ratings[ratings['userId'] == userId]['movieId'].tolist()

    # 모든 영화명을 list 객체로 만듬.
    total_movies = movies['movieId'].tolist()

    # 한줄 for + if문으로 안 본 영화 리스트 생성
    unseen_movies = [movie for movie in total_movies if movie not in seen_movies]

    # 일부 정보 출력
    total_movie_cnt = len(total_movies)
    seen_cnt = len(seen_movies)
    unseen_cnt = len(unseen_movies)

    print(f"전체 영화 수: {total_movie_cnt}, 평점 매긴 영화 수: {seen_cnt}, 추천 대상 영화 수: {unseen_cnt}")

    return unseen_movies

unseen_movies = get_unseen_surprise(ratings, movies, 9)


def recomm_movie_by_surprise(algo, userId, unseen_movies, top_n=10):
    # 아직 보지 않은 영화의 예측 평점: prediction 객체 생성
    predictions = []
    for movieId in unseen_movies:
        predictions.append(algo.predict(str(userId), str(movieId)))

    # 리스트 내의 prediction 객체의 est를 기준으로 내림차순 정렬
    def sortkey_est(pred):
        return pred.est

    predictions.sort(key=sortkey_est, reverse=True)  # key에 리스트 내 객체의 정렬 기준을 입력

    # 상위 top_n개의 prediction 객체
    top_predictions = predictions[:top_n]

    # 영화 아이디, 제목, 예측 평점 출력
    print(f"Top-{top_n} 추천 영화 리스트")

    for pred in top_predictions:
        movie_id = int(pred.iid)
        movie_title = movies[movies["movieId"] == movie_id]["title"].tolist()
        movie_rating = pred.est

        print(f"{movie_title}: {movie_rating:.2f}")

recomm_movie_by_surprise(algo, 9, unseen_movies, top_n=10)

from surprise.model_selection import GridSearchCV

# n_epochs: SGD 수행 시 반복 횟수, n_factors: 잠재 요인 크기
param_grid = {
    'n_epochs': [20, 40, 60],
    'n_factors': [50, 100, 200]
}

# GridSearchCV
gs = GridSearchCV(SVD, param_grid, measures=['rmse', 'mae'], cv=3) # algo가 아닌 SVD 입력하였다.
gs.fit(data)

# 최적 하이퍼 파라미터 및 그 때의 최고 성능
print(gs.best_params['rmse'])
print(gs.best_score['rmse'])

from surprise.dataset import DatasetAutoFolds

# Reader 객체 생성
reader = Reader(line_format='user item rating timestamp', sep=',', rating_scale=(0.5, 5))

# DatasetAutoFolds 객체 생성
data_folds = DatasetAutoFolds(ratings_file='./ml-latest-small/ratings_noh.csv', reader=reader)

# 전체 데이터를 train으로 지정
trainset = data_folds.build_full_trainset()
trainset

# SVD를 이용한 잠재 요인 협업 필터링 (잠재 요인 크기 = 50)
algo = SVD(n_epochs=20, n_factors=50, random_state=0)
algo.fit(trainset)

# 사용자 아이디, 아이템 아이디 문자열로 입력
uid = str(9)
iid = str(42)

# 추천 예측 평점 (.predict)
pred = algo.predict(uid, iid, verbose=True)

# 영화에 대한 상세 속성 정보 DataFrame로딩
movies = pd.read_csv('./ml-latest-small/movies.csv')


# 아직 보지 않은 영화 리스트 함수
def get_unseen_surprise(ratings, movies, userId):
    # 특정 userId가 평점을 매긴 모든 영화 리스트
    seen_movies = ratings[ratings['userId'] == userId]['movieId'].tolist()

    # 모든 영화명을 list 객체로 만듬.
    total_movies = movies['movieId'].tolist()

    # 한줄 for + if문으로 안 본 영화 리스트 생성
    unseen_movies = [movie for movie in total_movies if movie not in seen_movies]

    # 일부 정보 출력
    total_movie_cnt = len(total_movies)
    seen_cnt = len(seen_movies)
    unseen_cnt = len(unseen_movies)

    print(f"전체 영화 수: {total_movie_cnt}, 평점 매긴 영화 수: {seen_cnt}, 추천 대상 영화 수: {unseen_cnt}")

    return unseen_movies


def recomm_movie_by_surprise(algo, userId, unseen_movies, top_n=10):
    # 아직 보지 않은 영화의 예측 평점: prediction 객체 생성
    predictions = []
    for movieId in unseen_movies:
        predictions.append(algo.predict(str(userId), str(movieId)))

    # 리스트 내의 prediction 객체의 est를 기준으로 내림차순 정렬
    def sortkey_est(pred):
        return pred.est

    predictions.sort(key=sortkey_est, reverse=True)  # key에 리스트 내 객체의 정렬 기준을 입력

    # 상위 top_n개의 prediction 객체
    top_predictions = predictions[:top_n]

    # 영화 아이디, 제목, 예측 평점 출력
    print(f"Top-{top_n} 추천 영화 리스트")

    for pred in top_predictions:
        movie_id = int(pred.iid)
        movie_title = movies[movies["movieId"] == movie_id]["title"].tolist()
        movie_rating = pred.est

        print(f"{movie_title}: {movie_rating:.2f}")

recomm_movie_by_surprise(algo, 9, unseen_movies, top_n=10)