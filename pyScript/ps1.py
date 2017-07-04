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
for c in train:
		if len(set(train[c]))==1:
				no_info_col.append(c)

print(no_info_col)
train.drop(no_info_col, axis=1, inplace=True)
test.drop(no_info_col, axis=1, inplace=True)

enc = preprocessing.OneHotEncoder()
enc.fit(train.to_arr)
train_pre = enc.transform(train)
