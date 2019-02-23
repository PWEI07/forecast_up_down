# coding=utf-8
from builtins import *

import rqdatac as rd
import numpy as np
import talib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from time import time
from sklearn.metrics import f1_score, make_scorer, accuracy_score, precision_recall_curve, confusion_matrix
from pandas import Series

rd.init()


class ForestBuilder(object):

    def __init__(self, code):
        self.code = code

    def __xdata(self, start_date, end_date):
        test_data1 = rd.get_price(self.code, start_date=start_date, end_date=end_date, frequency='1d')
        test_data1['RSI'] = talib.RSI(test_data1['close'], timeperiod=9)
        test_data1['CMO'] = talib.CMO(test_data1['close'], timeperiod=14)
        test_data1['CCI'] = talib.CCI(test_data1['high'], test_data1['low'], test_data1['close'])
        test_data1['MACD'] = talib.MACD(test_data1['close'], 12, 26, 9)[-1]
        test_data1['K'], test_data1['D'] = talib.STOCH(test_data1['high'], test_data1['low'], test_data1['close'])
        test_data1['J'] = 3 * test_data1['K'] - 2 * test_data1['D']
        test_data1['ADOSC'] = talib.ADOSC(test_data1['high'], test_data1['low'], test_data1['close'],
                                          test_data1['volume'])
        temp0 = Series(data=0, index=test_data1.index)
        test_data1['ADOSC_up_cross'] = self.cross(test_data1.ADOSC, temp0)
        test_data1['ADOSC_down_cross'] = self.cross(temp0, test_data1.ADOSC)
        test_data1.drop(columns=['ADOSC'], inplace=True)
        test_data1.drop(columns=['limit_down', 'limit_up'], inplace=True)

        test_data1['ma5'] = test_data1['close'].rolling(window=5).mean()
        test_data1['ma10'] = test_data1['close'].rolling(window=10).mean()
        test_data1['sign'] = test_data1['ma5'] < test_data1['ma10']
        test_data1['sign'] = test_data1['sign'].replace({True: 1, False: -1})
        test_data1['ma5_minus_ma10'] = test_data1['ma5'] - test_data1['ma10']
        test_data1['ma5_minus_ma10_lagged'] = test_data1['ma5_minus_ma10'].shift(1)
        test_data1['delta'] = (test_data1['ma5_minus_ma10'] - test_data1['ma5_minus_ma10_lagged']) * test_data1['sign']
        test_data1.drop(columns=['ma5', 'ma10', 'sign', 'ma5_minus_ma10', 'ma5_minus_ma10_lagged'], inplace=True)

        test_data1['SAR'] = talib.SAR(test_data1['high'], test_data1['low'])
        test_data1['high_cross_sar'] = self.cross(test_data1.high, test_data1.SAR)
        test_data1['sar_cross_low'] = self.cross(test_data1.SAR, test_data1.low)
        test_data1.drop(columns=['SAR'], inplace=True)

        test_data1['hammer'] = talib.CDLHAMMER(test_data1.open, test_data1.high, test_data1.low, test_data1.close)
        test_data1['two_crows'] = talib.CDL2CROWS(test_data1.open, test_data1.high, test_data1.low, test_data1.close)
        test_data1['3inside'] = talib.CDL3INSIDE(test_data1.open, test_data1.high, test_data1.low, test_data1.close)
        test_data1['breakaway'] = talib.CDLBREAKAWAY(test_data1.open, test_data1.high, test_data1.low, test_data1.close)
        test_data1['3outside'] = talib.CDL3OUTSIDE(test_data1.open, test_data1.high, test_data1.low, test_data1.close)
        test_data1['gold_belt'] = talib.CDLBELTHOLD(test_data1.open, test_data1.high, test_data1.low, test_data1.close)
        test_data1['morning_star'] = talib.CDLMORNINGDOJISTAR(test_data1.open, test_data1.high, test_data1.low,
                                                              test_data1.close)
        test_data1['baby'] = talib.CDLCONCEALBABYSWALL(test_data1.open, test_data1.high, test_data1.low,
                                                       test_data1.close)
        test_data1['white_soldiers'] = talib.CDL3WHITESOLDIERS(test_data1.open, test_data1.high, test_data1.low,
                                                               test_data1.close)
        test_data1['grave_stone'] = talib.CDLGRAVESTONEDOJI(test_data1.open, test_data1.high, test_data1.low,
                                                            test_data1.close)
        test_data1['ladder_bottom'] = talib.CDLLADDERBOTTOM(test_data1.open, test_data1.high, test_data1.low,
                                                            test_data1.close)
        test_data1['doji_star'] = talib.CDLMORNINGDOJISTAR(test_data1.open, test_data1.high, test_data1.low,
                                                           test_data1.close)
        test_data1['piercing'] = talib.CDLPIERCING(test_data1.open, test_data1.high, test_data1.low, test_data1.close)

        test_data1.drop(columns=['total_turnover', 'close', 'high', 'low', 'open', 'volume'], inplace=True)
        test_data1.dropna(axis=0, inplace=True)  # get rid of nan rows
        return test_data1

    def __ydata(self, start_date, end_date, window):
        data_lagged = rd.get_price(self.code, start_date=start_date, end_date=end_date, frequency='1d')
        rise = self.posterior_rolling_max(data_lagged['high'], window).values > 1.01 * data_lagged[
            'open'].values  # target
        return Series(data=rise, index=data_lagged.index)

    @staticmethod
    def posterior_rolling_max(series, window=5):
        s1 = series.rolling(window=window).max()
        s1[:-(window - 1)] = s1[window - 1:]
        s1[-(window - 1):] = np.nan
        return s1

    def data(self, start_date, end_date, window=5):
        X = self.__xdata(start_date, end_date)
        y = self.__ydata(start_date, end_date, window)

        # match the length of target to features
        y = y[y.index.date >= X.index[0].date()]
        y = y[:-(window - 1)]
        X = X[:-(window - 1)]

        return X, y

    def split_scale(self, X, y, test_size=0.2):
        X_train_vali, X_test, y_train_vali, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
        scaler = MinMaxScaler()
        scaler.fit(X_train_vali)
        X_train_vali_transformed = scaler.transform(X_train_vali)
        X_test_transformed = scaler.transform(X_test)
        return X_train_vali_transformed, X_test_transformed, y_train_vali, y_test, scaler

    @staticmethod
    def report(results, n_top=3):
        for i in range(1, n_top + 1):
            candidates = np.flatnonzero(results['rank_test_score'] == i)
            for candidate in candidates:
                print("Model with rank: {0}".format(i))
                print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                    results['mean_test_score'][candidate],
                    results['std_test_score'][candidate]))
                print("Parameters: {0}".format(results['params'][candidate]))
                print("")

    def param_tunning(self, param_dist, X, y, test_size=0.2, cv=4, estimators=500):
        clf = RandomForestClassifier(n_estimators=estimators)
        grid_search = GridSearchCV(clf, param_grid=param_dist, cv=cv, scoring=make_scorer(f1_score),
                                   n_jobs=-1, refit=True)
        # split into train_vali and test set
        X_train_vali, X_test, y_train_vali, y_test, scaler = self.split_scale(X, y, test_size=test_size)
        grid_search.fit(X_train_vali, y_train_vali)

        return grid_search, X_train_vali, X_test, y_train_vali, y_test, scaler

    def get_eligible_rf(self, start_date, end_date, precision_threshold=0.85, param_dist={"max_depth": [6],
                                                                                      "max_features": [None],
                                                                                      "min_samples_split": [0.01],
                                                                                      "bootstrap": [False],
                                                                                          "criterion": ["entropy"]},
                        test_size=0.2, cv=4,
                        estimators=500, train_precision_threshold=0.9, test_precision_threshold=0.7, F_threshold=0.7):
        """
        调用param_tunning 找到并返回符合precision_threshold要求的模型中，F score最高的那个模型
        :param param_dist:
        :param cv:
        :param test_size:
        :param start_date:
        :param end_date:
        :return:
        """
        print("start getting best random forest model for ", self.code, "\n\n")
        X, y = self.data(start_date, end_date)
        start = time()
        search_result, X_train_vali, X_test, y_train_vali, y_test, scaler = self.param_tunning(param_dist=param_dist, X=X, y=y, test_size=test_size, cv=cv,
                                                                                               estimators=estimators)
        print("GridSearchCV took %.2f seconds" % (time() - start))
        best_rf = search_result.best_estimator_  # 这是最佳参数组成的模型

        # 找到使得X_train的precision大于0.85的最小的阈值
        precision, recall, thresholds = precision_recall_curve(y_train_vali, best_rf.predict_proba(X_train_vali)[:,1])
        if len(np.where(precision > 0.85)[0]) == 0:
            return None
        else:
            pass
        min_threshold = thresholds[np.min(np.where(precision > train_precision_threshold)[0])]

        # among thresholds greater than the min_threshold, find one that maximize F score
        train_prob_pred = best_rf.predict_proba(X_train_vali)[:, 1]
        largest_F = 0
        best_threshold = min_threshold
        for threshold in thresholds[thresholds > min_threshold]:
            classcification = train_prob_pred >= threshold
            if f1_score(y_train_vali, classcification) > largest_F:
                largest_F = f1_score(y_train_vali, classcification)
                best_threshold = threshold
            else:
                pass

        if largest_F < F_threshold:
            return None
        else:
            pass

        # test on the test set
        test_prob_pred = best_rf.predict_proba(X_test)[:, 1]
        test_pred = test_prob_pred > best_threshold
        # test_confusion_mat = confusion_matrix(y_test, test_pred)
        test_precision = sum(test_pred[test_pred == 1] == y_test[test_pred == 1])/sum(test_pred)
        if f1_score(y_test, test_pred) > F_threshold and test_precision > test_precision_threshold:
            scaler.fit(X)  # refit the scaler on all data available
            best_rf_all_data = best_rf.fit(scaler.transform(X), y)  # fit the model on all data available
            return {'best_rf': best_rf_all_data, 'saler': scaler, 'threshold': best_threshold, 'test_precision': test_precision}
        else:
            return None

    @staticmethod
    def cross(s1, s2):
        s1_greater_s2 = s1 > s2
        s1_greater_s2_lagged = s1_greater_s2.shift(1)
        result = s1_greater_s2[1:] & (~s1_greater_s2_lagged[1:]).replace({-1: True, -2: False})
        result[s1.index[0]] = np.nan
        return result
