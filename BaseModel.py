# coding=utf-8
from builtins import *
import abc
import rqdatac as rd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from pandas import Series
rd.init()


class BaseModel(object):

    def __init__(self, code):
        self.code = code

    @abc.abstractmethod
    def xdata(self, start_date, end_date, *args, **kwargs):
        """

        :param start_date:
        :param end_date:
        :param args:
        :param kwargs:
        :return:
        """
        pass

    @staticmethod
    def roll(df, w, **kwargs):
        roll_array = np.dstack([df.values[i:i + w, :] for i in range(len(df.index) - w + 1)]).T
        panel = pd.Panel(roll_array,
                         items=df.index[w - 1:],
                         major_axis=df.columns,
                         minor_axis=pd.Index(range(w), name='roll'))
        return panel.to_frame().unstack().T.groupby(level=0, **kwargs)

    def ydata(self, start_date, end_date, window, rise=0.015, stop=0.025):
        """

        :param stop:
        :param start_date:
        :param end_date:
        :param window:
        :param rise: 涨幅超过多少才标为1 默认0.015
        :return:
        """
        data_lagged = rd.get_price(self.code, start_date=start_date, end_date=end_date, frequency='1d',
                                   fields=['close', 'high', 'low'])
        data_lagged['high'] = data_lagged.high.shift(-1)
        data_lagged['low'] = data_lagged.low.shift(-1)

        y = []
        for i in range(len(data_lagged) - window + 1):
            temp_df = data_lagged[i: i+window]
            y.append(self.find_max_min(temp_df, rise=rise, stop=stop))
        y += [np.nan] * (window - 1)
        y = Series(data=y, index=data_lagged.index)
        return y
        # temp = self.roll(data_lagged, window)
        # label = temp.apply(func=lambda x: self.find_max_min(x, rise=rise, stop=stop))
        # return label
        # return self.posterior_rolling_max_min(data_lagged[['high', 'low']], window, rise=rise) > 1.015 * data_lagged[
        #     'open']
        # return Series(data=rise, index=data_lagged.index)

    # @staticmethod
    # def posterior_rolling_max_min(high_low, window=5, rise):
    #     high_low['high_1'] = high_low.high.shift(-1)
    #     high_low['low_1'] = high_low.low.shift(-1)
    #     high1 = high_low.shift(-1)
    #     low1 = low_series.shift(-1)
    #     s2 = high1.rolling(window=window).max()
    #     s2 = s2.shift(-(window - 1))
    #     return s2

    @staticmethod
    def find_max_min(df, rise, stop):
        """

        :param df:
        :param rise: 止盈点 eg 0.015
        :param stop: 止损点 eg 0.02
        :return:
        """
        buy_price = df['close'][0]
        threshold = (1 + rise) * buy_price
        if max(df['high']) <= threshold:
            return False
        else:
            sell_date = df.index[df['high'] > threshold][0]
            lowest_before_sell_date = np.min(df[df.index < sell_date]['low'])
            if lowest_before_sell_date >= (1 - stop) * buy_price:
                return True
            else:
                return False

    def data(self, X, y, window):
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

    @abc.abstractmethod
    def param_tunning(self, param_dist, X, y, test_size=0.2, cv=4):
        """

        :return: grid_search, X_train_vali, X_test, y_train_vali, y_test, scaler
        """
        pass

    @abc.abstractmethod
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
        pass

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
