import rqdatac as rd
import numpy as np
import talib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from time import time
from sklearn.metrics import f1_score, make_scorer
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
        test_data1['AD'] = talib.AD(test_data1['high'], test_data1['low'], test_data1['close'], test_data1['volume'])
        test_data1['ADOSC'] = talib.ADOSC(test_data1['high'], test_data1['low'], test_data1['close'],
                                          test_data1['volume'])
        test_data1.drop(columns=['limit_down', 'limit_up'], inplace=True)
        test_data1.dropna(axis=0, inplace=True)  # get rid of nan rows
        return test_data1
    
    def __ydata(self, start_date, end_date):
        data_lagged = rd.get_price(self.code, start_date=start_date, end_date=end_date, frequency='1d')
        rise = data_lagged['close'] > data_lagged['open']  # target
        return rise
    
    def data(self, start_date, end_date):
        X = self.__xdata(start_date, end_date)
        y = self.__ydata(start_date, end_date)

        # match the length of target to features
        y = y[y.index.date >= rd.get_next_trading_date(X.index[0].date())]
        X.drop(index=X.index[-1], inplace=True)
        return X, y

    def split_scale(self, X, y, test_size=0.2):
        X_train_vali, X_test, y_train_vali, y_test = train_test_split(X, y, test_size=test_size)
        scaler = MinMaxScaler()
        scaler.fit(X_train_vali)
        X_train_vali_transformed = scaler.transform(X_train_vali)
        X_test_transformed = scaler.transform(X_test)
        return X_train_vali_transformed, X_test_transformed, y_train_vali, y_test

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

    def param_tunning(self, param_dist, X, y, test_size=0.2, n_iter=100, cv=4):
        clf = RandomForestClassifier(n_estimators=1000)
        random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=n_iter, cv=cv,
                                           scoring=make_scorer(f1_score), n_jobs=-1)
        # split into train_vali and test set
        X_train_vali, X_test, y_train_vali, y_test = self.split_scale(X, y, test_size=test_size)
        start = time()
        random_search.fit(X_train_vali, y_train_vali)
        print("RandomizedSearchCV took %.2f seconds for %d candidates"
              " parameter settings." % ((time() - start), n_iter))
        self.report(random_search.cv_results_)

        return random_search



#     # ======================================================================================================================
#     X_train, X_vali_test, y_train, y_vali_test = train_test_split(test_data1, rise, test_size=0.4)
#     X_vali, X_test, y_vali, y_test = train_test_split(X_vali_test, y_vali_test, test_size=0.5)
#     # 既然目前不是RNN，所以就随机分割，而不是按顺序把靠前的划为训练集，靠后的划为测试集

#
#     # ======================================================================================================================
#     # 开始构造随机森林
#     forest = RandomForestClassifier(n_estimators=1000,
#                                     verbose=1, min_samples_split=0.025, n_jobs=-1, max_features=0.5)
#     forest.fit(X_train_transformed, y_train)
#     forest.score(X_vali_transformed, y_vali)
#     forest.score(X_train_transformed, y_train)
#
#
# # ======================================================================================================================
# # build machine learning model
# model = keras.Sequential()
# model.add(keras.layers.BatchNormalization())







