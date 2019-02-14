import rqdatac as rd
import pandas as pd
import talib
import keras
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
rd.init()

class ForestBuilder():
    def __init__(self, code, start_date, end_date):
        self.code = code
        self.start_date = start_date
        self.end_date = end_date

    def __xdata(self):
        test_data1 = rd.get_price(self.code, start_date=self.start_date, end_date=self.end_date, frequency='1d')
        test_data1['RSI'] = talib.RSI(test_data1['close'], timeperiod=9)
        test_data1['CMO'] = talib.CMO(test_data1['close'], timeperiod=14)
        test_data1['CCI'] = talib.CCI(test_data1['high'], test_data1['low'], test_data1['close'])
        test_data1['MACD'] = talib.MACD(test_data1['close'], 12, 26, 9)[-1]
        test_data1['K'], test_data1['D'] = talib.STOCH(test_data1['high'], test_data1['low'], test_data1['close'])
        test_data1['J'] = 3 * test_data1['K'] - 2 * test_data1['D']
        test_data1['AD'] = talib.AD(test_data1['high'], test_data1['low'], test_data1['close'], test_data1['volume'])
        test_data1['ADOSC'] = talib.ADOSC(test_data1['high'], test_data1['low'], test_data1['close'],
                                          test_data1['volume'])
        test_data1.dropna(axis=0, inplace=True)  # get rid of nan rows

    test_data1 = pd.DataFrame(test_data1)
    test_data1.shape  # (1382, 8)

    data_lagged = rd.get_price('510050.XSHG', start_date='2013-06-04', end_date='2019-01-30', frequency='1d')
    rise = data_lagged['close'] > data_lagged['open']  # target

    # match the length of target to features
    rise = rise[rise.index.date >= rd.get_next_trading_date(test_data1.index[0].date())]
    test_data1.drop(columns=['limit_down', 'limit_up'], inplace=True)
    # scale the data

    # ======================================================================================================================
    X_train, X_vali_test, y_train, y_vali_test = train_test_split(test_data1, rise, test_size=0.4)
    X_vali, X_test, y_vali, y_test = train_test_split(X_vali_test, y_vali_test, test_size=0.5)
    # 既然目前不是RNN，所以就随机分割，而不是按顺序把靠前的划为训练集，靠后的划为测试集
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train_transformed = scaler.transform(X_train)
    X_vali_transformed = scaler.transform(X_vali)
    X_test_transformed = scaler.transform(X_test)

    # ======================================================================================================================
    # 开始构造随机森林
    forest = RandomForestClassifier(criterion='entropy', n_estimators=1000,
                                    verbose=1, min_samples_split=0.025, n_jobs=-1, max_features=0.5)
    forest.fit(X_train_transformed, y_train)
    forest.score(X_vali_transformed, y_vali)
    forest.score(X_train_transformed, y_train)


# ======================================================================================================================
# build machine learning model
model = keras.Sequential()
model.add(keras.layers.BatchNormalization())







