from tree_clf.random_forest import ForestBuilder
from pandas import DataFrame
import talib
forest1 = ForestBuilder('000001.XSHE')
forest1_data = forest1.data('2013-01-01', '2019-02-13')

temp = DataFrame(forest1_data[0])
temp1 = DataFrame(forest1_data[1])

param_grid = {"max_depth": [7, 9, 11, None],
              "max_features": [None],
              "min_samples_split": [0.008, 0.005, 0.002],
              "bootstrap": [False],
              "criterion": ["entropy"]}

result = forest1.param_tunning(param_grid, forest1_data[0], forest1_data[1], cv=5)
best_rf = result.best_estimator_


temp['ma5'] = temp['close'].rolling(window=5).mean()
temp['ma10'] = temp['close'].rolling(window=10).mean()
temp['sign'] = temp['ma5'] < temp['ma10']
temp['sign'] = temp['sign'].replace({True: 1, False: -1})
temp['ma5_minus_ma10'] = temp['ma5'] - temp['ma10']
temp['ma5_minus_ma10_lagged'] = temp['ma5_minus_ma10'].shift(1)
temp['delta'] = (temp['ma5_minus_ma10'] - temp['ma5_minus_ma10_lagged']) * temp['sign']
temp.drop(columns=['ma5', 'ma10', 'sign', 'ma5_minus_ma10', 'ma5_minus_ma10_lagged'], inplace=True)


temp['SAR'] = talib.SAR(temp['high'], temp['low'])
temp['high_greater_SAR'] = temp['high'] > temp['SAR']
temp['high_greater_SAR_lagged'] = temp['high_greater_SAR'].shift(1)
result = (~temp.high_greater_SAR_lagged[1:]) & temp.high_greater_SAR[1:]
result[temp.high_greater_SAR.index[0]] = np.nan
temp['high_cross_SAR'] = result
# # split and scale X, y
# X_train, X_test, y_train, y_test = forest1.split_scale(temp, temp1)
# best_rf.fit(X_train, y_train)
# prediction = best_rf.predict(X_test)
# from sklearn.metrics import accuracy_score
# accuracy_score(y_test, prediction)
# accuracy_score(y_train, best_rf.predict(X_train))
def cross(s1, s2):
    s1_greater_s2 = s1 > s2
    s1_greater_s2_lagged = s1_greater_s2.shift(1)
    result = s1_greater_s2[1:] & (~s1_greater_s2_lagged[1:]).replace({-1: True, -2: False})
    result[s1.index[0]] = np.nan
    return result

temp['exaime'] = cross(temp.high, temp.SAR)



