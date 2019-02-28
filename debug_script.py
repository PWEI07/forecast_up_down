from tree_clf.random_forest import ForestBuilder

builder = ForestBuilder('000001.XSHE')
data1 = builder.data('2018-01-01', '2018-12-31')



# import rqdatac as rd
# from keras.models import Sequential
# from keras.layers import Dense
# from sklearn.model_selection import train_test_split
# import numpy as np
# import keras
# import matplotlib.pyplot as plt
# import pandas as pd
#
# rd.init()
#
# spot_input_data = rd.get_price('510050.XSHG', start_date='2015-06-01', end_date='2019-01-23', frequency='10m')
# future_data = rd.get_price('IH88', start_date='2015-06-01', end_date='2019-01-23', frequency='10m')
# future_data.drop(columns=['trading_date', 'limit_up', 'limit_down'], inplace=True)
# spot_output_data = rd.get_price('510050.XSHG', start_date='2015-06-02', end_date='2019-01-24', frequency='1d',
#                                 fields=['open', 'close'])
# spot_output_data = spot_output_data['close'] > spot_output_data['open']
#
#
# def combine_spot_future(spot, future, direction):
#     spot1 = spot.copy()
#     spot1.columns = list(map(lambda x: 'spot_' + x, spot1.columns))
#
#     future1 = future.copy()
#     future1.columns = list(map(lambda x: 'future_' + x, future1.columns))
#
#     direction1 = pd.DataFrame(direction)
#     direction1.columns = ['next_trading_day_direction']
#     direction1.index = list(map(rd.get_previous_trading_date, direction1.index))
#
#     future_time_set = set(list(map(lambda x: x.time(), future1.index)))
#     spot_time_set = set(list(map(lambda x: x.time(), spot.index)))
#
#     for t1 in spot_time_set:
#         temp_df = spot1[spot1.index.time == t1]
#         temp_df.columns = list(map(lambda x: x + str(t1), temp_df.columns))
#         temp_df.index = temp_df.index.date
#         direction1 = pd.DataFrame.merge(direction1, temp_df, left_index=True,
#                                         right_index=True, how='left')
#
#     for t2 in future_time_set:
#         temp_df = future1[future1.index.time == t2]
#         temp_df.columns = list(map(lambda x: x + str(t1), temp_df.columns))
#         temp_df.index = temp_df.index.date
#         direction1 = pd.DataFrame.merge(direction1, temp_df, left_index=True,
#                                         right_index=True, how='left')
#
#     return direction1
#
#
# total_data = combine_spot_future(spot_input_data, future_data, spot_output_data)
# total_data.fillna(method='ffill', inplace=True)
#
#
# def get_model():
#     model = Sequential()
#
#     # model.add(keras.layers.BatchNormalization(input_shape=(333,)))
#     model.add(Dense(256, kernel_initializer='glorot_normal', activation='relu',
#                     kernel_regularizer=keras.regularizers.l1(0.003)))
#     model.add(keras.layers.BatchNormalization())
#     model.add(keras.layers.Dropout(0.05))
#
#     model.add(Dense(128, kernel_initializer='glorot_normal', activation='relu',
#                     kernel_regularizer=keras.regularizers.l1(0.003)))
#     # model.add(keras.layers.BatchNormalization())
#     model.add(keras.layers.Dropout(0.05))
#
#     # model.add(Dense(64, kernel_initializer='glorot_normal', activation='relu',
#     #                 kernel_regularizer=keras.regularizers.l1(0.001)))
#     # model.add(keras.layers.BatchNormalization())
#     # model.add(keras.layers.Dropout(0.2))
#
#     model.add(Dense(32, kernel_initializer='glorot_normal', activation='relu',
#                     kernel_regularizer=keras.regularizers.l1(0.003)))
#     # model.add(keras.layers.BatchNormalization())
#     model.add(keras.layers.Dropout(0.05))
#
#     # model.add(Dense(16, kernel_initializer='glorot_normal', activation='relu',
#     #                 kernel_regularizer=keras.regularizers.l1(0.001)))
#     # model.add(keras.layers.BatchNormalization())
#     # model.add(keras.layers.Dropout(0.2))
#
#     model.add(Dense(1, kernel_initializer='glorot_normal', activation='sigmoid',
#                     kernel_regularizer=keras.regularizers.l1(0.003)))
#     return model
#
#
# def fit_model(model, X, Y, nb_epoch=1000):
#     # np.random.seed(seed)
#
#     (X_train_validation, X_test, Y_train_validation, Y_test) = train_test_split(X, Y, test_size=0.2)
#     (X_train, X_validation, Y_train, Y_validation) = train_test_split(X_train_validation, Y_train_validation,
#                                                                       test_size=0.2)
#
#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#     history = model.fit(X_train, Y_train,
#                         validation_data=(X_validation, Y_validation),
#                         epochs=nb_epoch, batch_size=32)
#
#     plt.plot(history.history['acc'])
#     plt.plot(history.history['val_acc'])
#     plt.title('model1 accuracy')
#     plt.ylabel('accuracy')
#     plt.xlabel('epoch')
#     plt.legend(['train', 'test'], loc='upper left')
#     plt.show()
#     # summarize history for loss
#     plt.plot(history.history['loss'])
#     plt.plot(history.history['val_loss'])
#     plt.title('model1 loss')
#     plt.ylabel('loss')
#     plt.xlabel('epoch')
#     plt.legend(['train', 'test'], loc='upper left')
#     plt.show()
#
#     scores = model.evaluate(X_test, Y_test)
#     print("\n\nAccuracy in test set: %.2f%%" % (scores[1] * 100))
#     return model
#
#
# total_data1 = total_data.dropna(axis='columns')
# myX = np.array(total_data1.iloc[:, 1:])
# myY = np.array(total_data1.iloc[:, 0])
#
# # ======================================================================================================================
# model2 = get_model()
# fit_model(model2, myX, myY, nb_epoch=500)
#
# import numpy as np
#
# X = np.array(
#     [[-0.066, -0.124, 0.259, 0.289, -0.318, -0.015, -0.06, 0.14, 0.203, 0.249, -0.229, 0.174, 0.149, -0.343, -0.308],
#      [-0.204, -0.079, 0.565, 0.607, -0.425, -0.135, 0.011, 0.411, 0.35, 0.535, -0.668, 0.128, 0.317, -0.543, -0.87]])
# from sklearn.decomposition import PCA
# pca = PCA(n_components=2)
# pca.fit(X.T)
# print(pca.explained_variance_ratio_)
# print(pca.components_)
# print(pca.explained_variance_)
#
# phy = np.array([[0.1, 0.2, 0.7], [0, 0, 1], [0.5, 0.5, 0]])
#
#
# data_lagged = rd.get_price('000001.XSHE', start_date='2015-01-01', end_date='2018-01-01', frequency='1d')
# data_lagged = pd.DataFrame(data_lagged)
# data_lagged['rolling_max_high'] = data_lagged.rolling(window=5, on='high').max()
#
# #=======================================================================================================================
#
# # 可以自己import我们平台支持的第三方python模块，比如pandas、numpy等。
# # coding=utf-8
# from builtins import *
# import abc
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler
# import math
#
#
# class BaseModel(object):
#
#     def __init__(self, code):
#         self.code = code
#
#     @abc.abstractmethod
#     def xdata(self, bar_count, *args, **kwargs):
#         """
#
#         :param start_date:
#         :param end_date:
#         :param args:
#         :param kwargs:
#         :return:
#         """
#         pass
#
#     def ydata(self, bar_count, window):
#         data_lagged = history_bars()(self.code, bar_count=bar_count, frequency='1d')
#         return self.posterior_rolling_max(data_lagged['high'], window) > 1.01 * data_lagged[
#             'open']
#
#     @staticmethod
#     def posterior_rolling_max(series, window=5):
#         s1 = series.shift(-1)
#         s2 = s1.rolling(window=window).max()
#         s2 = s2.shift(-(window - 1))
#         return s2
#
#     def data(self, X, y, window):
#         # match the length of target to features
#         y = y[y.datetime >= X.datetime[0]]
#         y = y[:-(window - 1)]
#         X = X[:-(window - 1)]
#
#         return X, y
#
#     def split_scale(self, X, y, test_size=0.2):
#         train_vali_len = floor(len(X) * (1 - test_size))
#         X_train_vali = X[: train_vali_len + 1].copy()
#         y_train_vali = y[: train_vali_len + 1].copy()
#         X_test = X[train_vali_len + 1:].copy()
#         y_test = y[train_vali_len + 1:].copy()
#         # X_train_vali, X_test, y_train_vali, y_test = train_test_split(X, y, test_size=test_size)
#         scaler = MinMaxScaler()
#         scaler.fit(X_train_vali)
#         X_train_vali_transformed = scaler.transform(X_train_vali)
#         X_test_transformed = scaler.transform(X_test)
#         return X_train_vali_transformed, X_test_transformed, y_train_vali, y_test, scaler
#
#     @staticmethod
#     def report(results, n_top=3):
#         for i in range(1, n_top + 1):
#             candidates = np.flatnonzero(results['rank_test_score'] == i)
#             for candidate in candidates:
#                 print("Model with rank: {0}".format(i))
#                 print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
#                     results['mean_test_score'][candidate],
#                     results['std_test_score'][candidate]))
#                 print("Parameters: {0}".format(results['params'][candidate]))
#                 print("")
#
#     @abc.abstractmethod
#     def param_tunning(self, param_dist, X, y, test_size=0.2, cv=4):
#         """
#
#         :return: grid_search, X_train_vali, X_test, y_train_vali, y_test, scaler
#         """
#         pass
#
#     @abc.abstractmethod
#     def get_eligible_model(self, bar_count, param_dist,
#                            test_size=0.2, cv=4, *args, **kwargs):
#         """
#         调用param_tunning，得到最优参数对应的模型，评估模型是否达到要求。若不达标，则返回None
#         :param start_date:
#         :param end_date:
#         :param param_dist:
#         :param test_size:
#         :param cv:
#         :param args:
#         :param kwargs:
#         :return:
#         """
#         pass
#
#     @staticmethod
#     def cross(s1, s2):
#         """
#
#         :param s1: pd.Series
#         :param s2: pd.Series
#         :return: pd.Series data=s1是否上穿s2
#         """
#         s1_greater_s2 = s1 > s2
#         s1_greater_s2_lagged = s1_greater_s2.shift(1)
#         result = s1_greater_s2[1:] & (~s1_greater_s2_lagged[1:]).replace({-1: True, -2: False})
#         result[s1.index[0]] = np.nan
#         return result
#
#
# import talib
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.ensemble import RandomForestClassifier
# from time import time
# from sklearn.metrics import f1_score, make_scorer, accuracy_score, precision_recall_curve, confusion_matrix
#
#
# class ForestBuilder(BaseModel):
#     def __init__(self, code):
#         super(ForestBuilder, self).__init__(code=code)
#
#     def xdata(self, bar_count=225 * 4):
#         test_data1 = history_bars(self.code, bar_count=bar_count, frequency='1d')
#         test_data1['RSI'] = talib.RSI(test_data1['close'].values, timeperiod=9)
#         test_data1['CMO'] = talib.CMO(test_data1['close'].values, timeperiod=14)
#         test_data1['CCI'] = talib.CCI(test_data1['high'].values, test_data1['low'].values, test_data1['close'].values)
#         test_data1['MACD'] = talib.MACD(test_data1['close'].values, 12, 26, 9)[-1]
#         test_data1['K'], test_data1['D'] = talib.STOCH(test_data1['high'].values, test_data1['low'].values, test_data1['close'].values)
#         test_data1['J'] = 3 * test_data1['K'] - 2 * test_data1['D']
#         test_data1['ADOSC'] = talib.ADOSC(test_data1['high'].values, test_data1['low'].values, test_data1['close'].values,
#                                           test_data1['volume'].values)
#         test_data1.drop(columns=['ADOSC'], inplace=True)
#
#         test_data1['ma5'] = test_data1['close'].rolling(window=5).mean()
#         test_data1['ma10'] = test_data1['close'].rolling(window=10).mean()
#         test_data1['sign'] = test_data1['ma5'] < test_data1['ma10']
#         test_data1['sign'] = test_data1['sign'].replace({True: 1, False: -1})
#         test_data1['ma5_minus_ma10'] = test_data1['ma5'] - test_data1['ma10']
#         test_data1['ma5_minus_ma10_lagged'] = test_data1['ma5_minus_ma10'].shift(1)
#         test_data1['delta'] = (test_data1['ma5_minus_ma10'] - test_data1['ma5_minus_ma10_lagged']) * test_data1['sign']
#         test_data1.drop(columns=['ma5', 'ma10', 'sign', 'ma5_minus_ma10', 'ma5_minus_ma10_lagged'], inplace=True)
#
#         test_data1['SAR'] = talib.SAR(test_data1['high'].values, test_data1['low'].values)
#         test_data1['high_cross_sar'] = self.cross(test_data1.high, test_data1.SAR)
#         test_data1['sar_cross_low'] = self.cross(test_data1.SAR, test_data1.low)
#         test_data1.drop(columns=['SAR'], inplace=True)
#         test_data1['gold_belt'] = talib.CDLBELTHOLD(test_data1.open.values, test_data1.high.values, test_data1.low.values, test_data1.close.values)
#
#         test_data1.drop(columns=['total_turnover', 'close', 'high', 'low', 'open', 'volume'], inplace=True)
#         test_data1.dropna(axis=0, inplace=True)  # get rid of nan rows
#         return test_data1
#
#     def data(self, bar_count, window=5):
#         X = self.xdata(bar_count)
#         y = super(ForestBuilder, self).ydata(bar_count, window=window)
#
#         # match the length of target to features
#         X, y = super().data(X, y, window=window)
#         return X, y
#
#     def param_tunning(self, param_dist, X, y, test_size=0.2, cv=4, estimators=500):
#         clf = RandomForestClassifier(n_estimators=estimators)
#         grid_search = GridSearchCV(clf, param_grid=param_dist, cv=cv, scoring=make_scorer(f1_score),
#                                    n_jobs=-1, refit=True)
#         # split into train_vali and test set
#         X_train_vali, X_test, y_train_vali, y_test, scaler = self.split_scale(X, y, test_size=test_size)
#         grid_search.fit(X_train_vali, y_train_vali)
#
#         return grid_search, X_train_vali, X_test, y_train_vali, y_test, scaler
#
#     def get_eligible_model(self, bar_count=225 * 4, param_dist={"max_depth": [6], "max_features": [None], "min_samples_split": [0.01], "bootstrap": [False], "criterion": ["entropy"]}, test_size=0.2, cv=4, estimators=100, precision_threshold=0.7, F_threshold=0.7):
#         """
#         调用param_tunning 找到并返回符合precision_threshold要求的模型中，F score最高的那个模型
#         :param param_dist:
#         :param cv:
#         :param test_size:
#         :param start_date:
#         :param end_date:
#         :return:
#         """
#         print("start getting best random forest model for ", self.code, "\n\n")
#         X, y = self.data(bar_count)
#         start = time()
#         search_result, X_train_vali, X_test, y_train_vali, y_test, scaler = self.param_tunning(param_dist=param_dist, X=X, y=y, test_size=test_size, cv=cv, estimators=estimators)
#         print("GridSearchCV took %.2f seconds" % (time() - start))
#         best_rf = search_result.best_estimator_  # 这是最佳参数组成的模型
#
#         # 如果训练集或测试集F score小于F score阈值，返回None
#         test_pred = best_rf.predict(X_test)
#         f1_test = f1_score(y_test, test_pred)
#         if min(f1_score(y_train_vali, best_rf.predict(X_train_vali)), f1_test) < F_threshold:
#             return None
#         else:
#             pass
#
#         test_confusion_mat = confusion_matrix(y_test, test_pred)
#         test_precision = test_confusion_mat[1, 1] / sum(test_confusion_mat[:, 1])
#         if test_precision < precision_threshold:
#             return None
#         else:
#             scaler.fit(X)  # refit the scaler on all data available
#             best_rf_all_data = best_rf.fit(scaler.transform(X), y)  # fit the model on all data available
#             return {'best_rf': best_rf_all_data, 'scaler': scaler, 'test_precision': test_precision, 'test_f1': f1_test}
#
# from datetime import timedelta
# from pandas import Series, DataFrame
# import numpy as np
#
#
# # 在这个方法中编写任何的初始化逻辑。context对象将会在你的算法策略的任何方法之间做传递。
# def init(context):
#     context.eligible_models = {}  # 用于存储通过标准的模型
#     context.eligible_models_weights = Series()  # 用于存储达标模型的资金权重
#     context.eligible_models_scalers = {}  # 用于存储达标模型的scaler
#
#     scheduler.run_monthly(get_models, tradingday=1)  # 每个月训练一次模型，选出符合标准的放入context.eligible_models
#     context.precision_threshold = 0.8  # 设置精确度阈值，通过此阈值的模型才可以入选
#     context.F_threshold = 0.8  # 设置F1 score阈值，通过此阈值的模型才可以入选
#     context.transaction = DataFrame(columns=['date', 'cost'])  # 用于记录买入的品种，日期，价格
#     scheduler.run_daily(sell_old, time_rule=market_open(minute=0))  # 每天09:31卖掉持有日大于等于5天的股票
#     scheduler.run_daily(make_prediction, time_rule=market_open(minute=1))  # 每天09:32进行预测
#     scheduler.run_daily(order_buy_list, time_rule=market_open(minute=2))  # 每天09:33买入context.buy_list中的股票
#
#     context.buy_list = []  # 预测要涨 需要买的股票
#
#
# def get_models(context, bar_dict):
#     """
#     获得该期达标模型，以及其scaler，并根据其模型得分计算对其分配的资金权重
#     :param context:
#     :return:
#     """
#     # 每个月 首先更新股票池
#     context.pool = index_components('000300.XSHG')
#     # context.pool = ['600489.XSHG', '600585.XSHG', '601899.XSHG', '600188.XSHG', '600348.XSHG']
#     context.model_generators = {}  # 用于存储投资池中各个股票的模型生成器
#     for i in context.pool:
#         context.model_generators[i] = ForestBuilder(i)
#
#     context.eligible_models = {}  # 先将上一期达标模型清空
#     context.eligible_models_scalers = {}
#     context.eligible_models_weights = Series()
#     # end_date = get_previous_trading_date(context.now, n=1).strftime('%Y-%m-%d')
#     # start_date = (context.now - timedelta(days=365 * 4)).strftime('%Y-%m-%d')
#
#     for i in context.pool:
#         model_i = context.model_generators[i].get_eligible_model(bar_count=225 * 4, precision_threshold=context.precision_threshold,F_threshold=context.F_threshold)
#         if model_i is not None:
#             print(i, ':')
#             print(model_i, '\n\n')
#             context.eligible_models[i] = model_i['best_rf']
#             context.eligible_models_scalers[i] = model_i['scaler']
#             context.eligible_models_weights[i] = model_i['test_precision'] + model_i['test_f1']
#     context.eligible_models_weights /= context.eligible_models_weights.sum()
#
#
# def sell_old(context, bar_dict):
#     """
#     卖掉持有日超过5天的股票
#     :param context:
#     :param bar_dict:
#     :return:
#     """
#     temp_sell_df = context.transaction[(context.now - context.transaction['date']) >= timedelta(days=5)]
#     for i in temp_sell_df.index:
#         print('start to sell old ', i)
#         order_shares(i, amount=-1 * context.portfolio.positions[i].sellable)
#         if context.portfolio.positions[i].quantity == 0:
#             context.transaction.drop(index=i, inplace=True)
#         else:
#             pass
#
#
# def record_transaction(context, bar_dict, order_book_id):
#     """
#     记录做多的股票代码 开仓时间 开仓价格
#     :param context:
#     :return:
#     """
#     context.transaction = context.transaction.append(Series(index=['date', 'cost'], data=[context.now, bar_dict[order_book_id].last], name=order_book_id))
#
#
# def make_prediction(context, bar_dict):
#     """
#     对每个达标模型，获取对应股票最近60个交易日的输入数据，利用模型进行预测
#     :param context:
#     :return:
#     """
#     context.buy_list = []
#     # start_date = get_previous_trading_date(context.now, 101)
#     # end_date = get_previous_trading_date(context.now, 1)
#
#     for stock in context.eligible_models:
#         if stock in context.portfolio.positions:
#             continue  # 如果该模型已经开仓 则跳过
#         else:
#             pass
#         model_generator = context.model_generators[stock]  # 调取模型生成器
#         model_X = model_generator.xdata(100)  # 获取对应股票最近100个交易日的输入数据
#         predictions = context.eligible_models[stock].predict(context.eligible_models_scalers[stock].transform(model_X))
#         if predictions[-1]:
#             context.buy_list.append(stock)
#         else:
#             pass
#
#
# def order_buy_list(context, bar_dict):
#     for stock in context.buy_list:
#         print('buy stock ', stock)
#         order = order_percent(stock, 0.8 * context.eligible_models_weights[stock])
#         if order is not None:
#             if order.filled_quantity != 0:
#                 record_transaction(context, bar_dict, stock)
#             else:
#                 pass
#         else:
#             pass
#
#
# def check_and_sell(context, bar_dict):
#     """
#     对于持有的股票，如果现在市场价高于其成本价1%以上，则全部卖出
#     :param context:
#     :param bar_dict:
#     :return:
#     """
#     for i in context.transaction.index:
#
#         if bar_dict[i].last >= np.mean(context.transaction.loc[i].cost) * 1.01:
#             print('sell and make profit on ', i, '\n\n')
#             order_shares(i, amount=-1 * context.portfolio.positions[i].sellable)
#             if context.portfolio.positions[i].quantity == 0:
#                 context.transaction.drop(index=i, inplace=True)
#             else:
#                 pass
#         else:
#             pass
#
#
# # before_trading此函数会在每天策略交易开始前被调用，当天只会被调用一次
# def before_trading(context):
#     for i in context.transaction.index:
#         if i not in context.portfolio.positions.keys():
#             print(i, 'is in context.transaction.index, but not in context.portfolio.positions.keys() !\n\n')
#             context.transaction.drop(index=i, inplace=True)
#         else:
#             pass
#
#
# # 你选择的证券的数据更新将会触发此段逻辑，例如日或分钟历史数据切片或者是实时数据切片更新
# def handle_bar(context, bar_dict):
#     minute = context.now.minute
#     if minute == 1 or minute == 29:
#         check_and_sell(context, bar_dict)  # 加快运行速度 只在每小时的第1 29分钟检查是否需要获利了结
#     else:
#         pass
#
#
# # after_trading函数会在每天交易结束后被调用，当天只会被调用一次
# def after_trading(context):
#     pass
#
#
# import numpy as np
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# x = np.linspace(start=0,stop=2,num=1000)
# y = np.linspace(start=0,stop=1,num=1000)
# X, Y = np.meshgrid(x, y)
# X, Y = np.meshgrid(x, y)
# zs = np.array([3*(x**2+y)/11 for x,y in zip(np.ravel(X), np.ravel(Y))])
# Z = zs.reshape(X.shape)
# ax.plot_surface(X, Y, Z)
# plt.show()
#
# plt.plot(x, 3*(2*x**2+1)/22)
# plt.show()
#
# plt.plot(y, (6*y+8)/11)
# plt.show()
#
#
# plt.plot(y, 2*y)
# plt.title('f(y|0)')
# plt.show()
#
# plt.plot(y, (2+2*y)/3)
# plt.title('f(y|1)')
# plt.show()
#
# plt.plot(y, (8+2*y)/9)
# plt.title('f(y|2)')
# plt.show()
#
# from fractions import Fraction
# Fraction(116,55)-Fraction(15*15,11*11)
# Fraction(25,66)-Fraction(36, 121)
# Fraction(8,11)-Fraction(15*6,11*11)
#
# # CEF:
# plt.plot(x, (x**2+2/3)/(2*x**2+1))
# # BLP:
# plt.plot(x, -10/151*x+96/151)
#
# plt.legend(['CEF', 'BLP'], loc='upper right')
# plt.show()
#
# X_t=np.array([[1,1,1,1],[4,-2,3,-5]])
#
#
