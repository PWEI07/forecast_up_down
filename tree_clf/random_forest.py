# coding=utf-8
from builtins import *
from BaseModel import BaseModel
import rqdatac as rd
import talib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from time import time
from sklearn.metrics import f1_score, make_scorer, accuracy_score, precision_recall_curve, confusion_matrix

rd.init()


class ForestBuilder(BaseModel):
    def __init__(self, code):
        super(ForestBuilder, self).__init__(code=code)

    def xdata(self, start_date, end_date):
        test_data1 = rd.get_price(self.code, start_date=start_date, end_date=end_date, frequency='1d')
        test_data1['RSI'] = talib.RSI(test_data1['close'], timeperiod=9)
        test_data1['CMO'] = talib.CMO(test_data1['close'], timeperiod=14)
        test_data1['CCI'] = talib.CCI(test_data1['high'], test_data1['low'], test_data1['close'])
        test_data1['MACD'] = talib.MACD(test_data1['close'], 12, 26, 9)[-1]
        test_data1['K'], test_data1['D'] = talib.STOCH(test_data1['high'], test_data1['low'], test_data1['close'])
        test_data1['J'] = 3 * test_data1['K'] - 2 * test_data1['D']
        test_data1['ADOSC'] = talib.ADOSC(test_data1['high'], test_data1['low'], test_data1['close'],
                                          test_data1['volume'])
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
        test_data1['gold_belt'] = talib.CDLBELTHOLD(test_data1.open, test_data1.high, test_data1.low, test_data1.close)

        test_data1.drop(columns=['total_turnover', 'close', 'high', 'low', 'open', 'volume'], inplace=True)
        test_data1.dropna(axis=0, inplace=True)  # get rid of nan rows
        return test_data1

    def data(self, start_date, end_date, window=5, rise=0.015, stop=0.02):
        X = self.xdata(start_date, end_date)
        y = super(ForestBuilder, self).ydata(start_date, end_date, window=window, rise=rise, stop=stop)
        # y = super().__ydata(start_date, end_date, window=window, equity=self.code)

        # match the length of target to features
        X, y = super().data(X, y, window=window)
        return X, y

    def param_tunning(self, param_dist, X, y, test_size=0.2, cv=4, estimators=500):
        clf = RandomForestClassifier(n_estimators=estimators)
        grid_search = GridSearchCV(clf, param_grid=param_dist, cv=cv, scoring=make_scorer(f1_score),
                                   n_jobs=-1, refit=True)
        # split into train_vali and test set
        X_train_vali, X_test, y_train_vali, y_test, scaler = self.split_scale(X, y, test_size=test_size)
        grid_search.fit(X_train_vali, y_train_vali)

        return grid_search, X_train_vali, X_test, y_train_vali, y_test, scaler

    def get_eligible_model(self, start_date, end_date, param_dist={"max_depth": [6],
                                                                "max_features": [None],
                                                                "min_samples_split": [0.01],
                                                                "bootstrap": [False],
                                                                "criterion": ["entropy"]},
                           test_size=0.2, cv=4,
                           estimators=100, precision_threshold=0.7, F_threshold=0.7):
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
        X, y = self.data(start_date, end_date, )
        start = time()
        search_result, X_train_vali, X_test, y_train_vali, y_test, scaler = self.param_tunning(param_dist=param_dist,
                                                                                               X=X, y=y,
                                                                                               test_size=test_size,
                                                                                               cv=cv,
                                                                                               estimators=estimators)
        print("GridSearchCV took %.2f seconds" % (time() - start))
        best_rf = search_result.best_estimator_  # 这是最佳参数组成的模型

        # 如果训练集或测试集F score小于F score阈值，返回None
        test_pred = best_rf.predict(X_test)
        f1_test = f1_score(y_test, test_pred)
        if min(f1_score(y_train_vali, best_rf.predict(X_train_vali)), f1_test) < F_threshold:
            return None
        else:
            pass

        test_confusion_mat = confusion_matrix(y_test, test_pred)
        test_precision = test_confusion_mat[1, 1] / sum(test_confusion_mat[:, 1])
        if test_precision < precision_threshold:
            return None
        else:
            scaler.fit(X)  # refit the scaler on all data available
            best_rf_all_data = best_rf.fit(scaler.transform(X), y)  # fit the model on all data available
            return {'best_rf': best_rf_all_data, 'scaler': scaler, 'test_precision': test_precision, 'test_f1': f1_test}


