# -*- coding = utf-8 -*-

import numpy as np
import pandas as pd
from sklearn import preprocessing
import xgboost as xgb

#load train and test
train = pd.read_csv('../data/train.csv',index_col=0)
test = pd.read_csv('../data/test.csv',index_col=0)

train_y = train['y']
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


enc = preprocessing.OneHotEncoder()
trainx = enc.fit_transform(train)
testx = enc.transform(test)
print(trainx.shape,testx.shape)

from sklearn.linear_model import SGDRegressor

model = SGDRegressor()
model.fit(trainx,train_y)

#preds = model.predict(testx)

