from surprise import Dataset
from surprise.model_selection import train_test_split
from surprise import accuracy
from surprise import SVD

# 1. surprise package 내 Dataset에서 제공된 ‘ml-100k’ 데이터를 loading 하시오
data = Dataset.load_builtin('ml-100k')

# 2. load된 데이터를 7:3 비율로 training data set, testing data set으로 분리하시오.
trainset, testset = train_test_split(data, test_size=0.7, random_state=0)

# 3. training data set 대상으로 특이값 분해(SVD)를 적용하여 학습시키시오.
algo = SVD()
algo.fit(trainset)

# 4. test data set을 대상으로 예측을 하시오
predictions = algo.test( testset )
predictions

# 5. 예측결과 객체의 type과 size를 print하시오.
print('type :',type(predictions), ' size:',len(predictions))

# 6. 예측결과 10개(uid, iid, est)를 리스트내포로 나열하시오
predictions[:10]

# 7. uid가 str(200), iid가 str(300)인 경우 예측하시오.
uid = str(200)
iid = str(300)
pred = algo.predict(uid, iid)
pred

# 8. 예측결과 객체 전체의 RMSE값을 산출하시오.
accuracy.rmse(predictions)