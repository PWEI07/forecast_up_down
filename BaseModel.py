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


class BaseModel(object):

    def __init__(self, code):
        self.code = code

    def __xdata(self, start_date, end_date, *args, **kwargs):
        """

        :param start_date:
        :param end_date:
        :param args:
        :param kwargs:
        :return:
        """
        raise NotImplementedError

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

    def data(self, start_date, end_date, window=5, *args, **kwargs):
        X = self.__xdata(start_date, end_date, *args, **kwargs)
        y = self.__ydata(start_date, end_date, window=window)

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

    def param_tunning(self):
        """

        :return: grid_search, X_train_vali, X_test, y_train_vali, y_test, scaler
        """
        return NotImplementedError

    def get_eligible_model(self, start_date, end_date, param_dist,
                           test_size=0.2, cv=4, *args, **kwargs):
        """
        调用param_tunning，得到最优参数对应的模型，评估模型是否达到要求。若不达标，则返回None
        :param start_date:
        :param end_date:
        :param param_dist:
        :param test_size:
        :param cv:
        :param args:
        :param kwargs:
        :return:
        """
        raise NotImplementedError

    @staticmethod
    def cross(s1, s2):
        """

        :param s1: pd.Series
        :param s2: pd.Series
        :return: pd.Series data=s1是否上穿s2
        """
        s1_greater_s2 = s1 > s2
        s1_greater_s2_lagged = s1_greater_s2.shift(1)
        result = s1_greater_s2[1:] & (~s1_greater_s2_lagged[1:]).replace({-1: True, -2: False})
        result[s1.index[0]] = np.nan
        return result
