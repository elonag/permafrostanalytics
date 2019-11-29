#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
import statsmodels.api as sm


prec = pd.read_csv('data/MH25_vaisalawxt520prec_2017.csv')
wind = pd.read_csv('data/MH25_vaisalawxt520windpth_2017.csv')
temp = pd.read_csv('data/MH30_temperature_rock_2017.csv')
radio = pd.read_csv('data/MH15_radiometer__conv_2017.csv')

temp_inner = pd.merge(left= temp, right=prec, left_on='time',right_on='time')
temp_inner = pd.merge(left= temp_inner, right=wind, left_on='time',right_on='time')
temp_inner = pd.merge(left=temp_inner, right=radio, left_on='time',right_on='time')


diff = temp_inner['temperature_100cm [°C]'] - temp_inner['temperature_5cm [°C]']


del temp_inner['time']



del temp_inner['position []_x']



diff.shape



del temp_inner['delta_t']
temp_inner.insert(0,'delta_t',diff)




abs(temp_inner.corr()['delta_t']).sort_values(ascending=False)



temp_inner.reset_index()



n = temp_inner.shape[0]
train_ind = int(np.round(0.7 * n))
print(train_ind)



temp_inner.columns = temp_inner.columns.str.replace("[", "_")
temp_inner.columns = temp_inner.columns.str.replace("]", "_")


temp_inner = temp_inner.fillna(0)
temp_inner.replace([np.inf, -np.inf], 0)



X, y = temp_inner.iloc[:,1:],temp_inner.iloc[:,0]




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)




X_train.shape



model = sm.OLS(y_train,X_train).fit()




preds = model.predict(X_test)




model.summary()

plt.plot(preds, y_test)




