# -*- coding = utf-8 -*-

import numpy as np
import pandas as pd
from sklearn import preprocessing
import xgboost as xgb

#load train and test
train = pd.read_csv('../data/train.csv',index_col=0)
test = pd.read_csv('../data/test.csv',index_col=0)

train_y = train['y']
y_mean = np.mean(train_y)
train.drop('y', axis=1, inplace=True)

no_info_col = []
obj_col = []
for c in train.columns:
		if len(set(train[c]))==1:
				no_info_col.append(c)
		if train[c].dtype==object:
				lbe = preprocessing.LabelEncoder()
				lbe.fit(list(train[c].values)+list(test[c].values))
				train[c] = lbe.transform(train[c])
				test[c] = lbe.transform(test[c])

print(no_info_col)

train.drop(no_info_col, axis=1, inplace=True)
test.drop(no_info_col, axis=1, inplace=True)


enc = preprocessing.OneHotEncoder(handle_unknown='ignore')
trainx = enc.fit_transform(train)
testx = enc.transform(test)
print(trainx.shape,testx.shape)

from sklearn.linear_model import SGDRegressor


# prepare dict of params for xgboost to run with
xgb_params = {
    'n_trees': 500, 
    'eta': 0.005,
    'max_depth': 4,
    'subsample': 0.95,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'base_score': y_mean, # base prediction = mean(target)
    'silent': 1
}

# form DMatrices for Xgboost training
dtrain = xgb.DMatrix(train, train_y)
dtest = xgb.DMatrix(test)

# xgboost, cross-validation
cv_result = xgb.cv(xgb_params, 
                   dtrain, 
                   num_boost_round=800, # increase to have better results (~700)
                   early_stopping_rounds=50,
                   verbose_eval=50, 
                   show_stdv=False
                  )

num_boost_rounds = len(cv_result)
print(num_boost_rounds)

model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)

from sklearn.metrics import r2_score

# now fixed, correct calculation
print(r2_score(dtrain.get_label(), model.predict(dtrain)))

preds = model.predict(dtest)

output = pd.DataFrame({'id':test.index,'y':preds})
output.to_csv('../result/oneHot_xgb_default.csv', index=False)
