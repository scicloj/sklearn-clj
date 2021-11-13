#!/usr/bin/env python3

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import os
data_train = pd.read_csv('https://gist.githubusercontent.com/behrica/8601bd8d6dfee59bd92d0cf5cc8db834/raw/bad7f3f8ef381cc3cb816ba16124d7aa7f83b930/flight_delays_train.csv')

def removeC(data):
  return data[2:]

data_train = data_train.drop(columns = ['UniqueCarrier', 'Origin', 'Dest'])
data_train['dep_delayed_15min'].replace({"Y": 1, "N": 0}, inplace=True)

data_train['Month'] = data_train.apply(lambda row : removeC(row['Month']), axis=1)
data_train['DayofMonth'] = data_train.apply(lambda row : removeC(row['DayofMonth']), axis=1)
data_train['DayOfWeek'] = data_train.apply(lambda row : removeC(row['DayOfWeek']), axis=1)
data_train = data_train.apply(pd.to_numeric)
reg = GradientBoostingClassifier(random_state=0, n_estimators=350, learning_rate=0.16)

X_train, y_train = data_train.drop(columns=['dep_delayed_15min']).values, data_train.dep_delayed_15min.values
import time
start = time.time()
reg.fit(X_train, y_train)
end = time.time()
execution_time = (end - start)
tensor_train=data_train.to_numpy()
