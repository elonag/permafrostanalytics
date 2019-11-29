import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

# all data
prec = pd.read_csv('../data/MH25_vaisalawxt520prec_2017.csv')
prec['time'] = pd.to_datetime(prec['time'])
wind = pd.read_csv('../data/MH25_vaisalawxt520windpth_2017.csv')
wind['time'] = pd.to_datetime(wind['time'])
temp = pd.read_csv('../data/MH30_temperature_rock_2017.csv')
temp['time'] = pd.to_datetime(temp['time'])
radio = pd.read_csv('../data/MH15_radiometer__conv_2017.csv')
radio['time'] = pd.to_datetime(radio['time'])

# join all data by time
temp0 = pd.merge(left=temp, right=prec, left_on='time', right_on='time')
temp0 = pd.merge(left=temp0, right=wind, left_on='time', right_on='time')
temp0 = pd.merge(left=temp0, right=radio, left_on='time', right_on='time')

# difference between ground and deepest temperature --> objective
diff = temp0['temperature_100cm [°C]'] - temp0['temperature_5cm [°C]']
temp0.insert(0, 'delta_t', diff)

# remove useless columns
del temp0['position []_x']
del temp0['position []_y']

# show correlation (within same time period)
print(abs(temp0.corr()['delta_t']).sort_values(ascending=False))

# clean column names and remove inf/nas
temp0.columns = temp0.columns.str.replace("[", "_")
temp0.columns = temp0.columns.str.replace("]", "_")
temp0 = temp0.fillna(0)
temp0.replace([np.inf, -np.inf], 0)

# create test and train samples
X, y = temp0.iloc[:, 1:], temp0.iloc[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# train model and predict
model = sm.OLS(y_train, X_train).fit()
preds = model.predict(X_test)

model.summary()
plt.plot(preds, y_test)
