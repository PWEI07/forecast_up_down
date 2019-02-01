import rqdatac as rd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import numpy as np
import keras
import matplotlib.pyplot as plt
import pandas as pd

rd.init()

spot_input_data = rd.get_price('510050.XSHG', start_date='2015-06-01', end_date='2019-01-23', frequency='10m')
future_data = rd.get_price('IH88', start_date='2015-06-01', end_date='2019-01-23', frequency='10m')
future_data.drop(columns=['trading_date', 'limit_up', 'limit_down'], inplace=True)
spot_output_data = rd.get_price('510050.XSHG', start_date='2015-06-02', end_date='2019-01-24', frequency='1d',
                                fields=['open', 'close'])
spot_output_data = spot_output_data['close'] > spot_output_data['open']


def combine_spot_future(spot, future, direction):
    spot1 = spot.copy()
    spot1.columns = list(map(lambda x: 'spot_' + x, spot1.columns))

    future1 = future.copy()
    future1.columns = list(map(lambda x: 'future_' + x, future1.columns))

    direction1 = pd.DataFrame(direction)
    direction1.columns = ['next_trading_day_direction']
    direction1.index = list(map(rd.get_previous_trading_date, direction1.index))

    future_time_set = set(list(map(lambda x: x.time(), future1.index)))
    spot_time_set = set(list(map(lambda x: x.time(), spot.index)))

    for t1 in spot_time_set:
        temp_df = spot1[spot1.index.time == t1]
        temp_df.columns = list(map(lambda x: x + str(t1), temp_df.columns))
        temp_df.index = temp_df.index.date
        direction1 = pd.DataFrame.merge(direction1, temp_df, left_index=True,
                                        right_index=True, how='left')

    for t2 in future_time_set:
        temp_df = future1[future1.index.time == t2]
        temp_df.columns = list(map(lambda x: x + str(t1), temp_df.columns))
        temp_df.index = temp_df.index.date
        direction1 = pd.DataFrame.merge(direction1, temp_df, left_index=True,
                                        right_index=True, how='left')

    return direction1


total_data = combine_spot_future(spot_input_data, future_data, spot_output_data)
total_data.fillna(method='ffill', inplace=True)


def get_model():
    model = Sequential()

    # model.add(keras.layers.BatchNormalization(input_shape=(333,)))
    model.add(Dense(256, kernel_initializer='glorot_normal', activation='relu',
                    kernel_regularizer=keras.regularizers.l1(0.003)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.05))

    model.add(Dense(128, kernel_initializer='glorot_normal', activation='relu',
                    kernel_regularizer=keras.regularizers.l1(0.003)))
    # model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.05))

    # model.add(Dense(64, kernel_initializer='glorot_normal', activation='relu',
    #                 kernel_regularizer=keras.regularizers.l1(0.001)))
    # model.add(keras.layers.BatchNormalization())
    # model.add(keras.layers.Dropout(0.2))

    model.add(Dense(32, kernel_initializer='glorot_normal', activation='relu',
                    kernel_regularizer=keras.regularizers.l1(0.003)))
    # model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.05))

    # model.add(Dense(16, kernel_initializer='glorot_normal', activation='relu',
    #                 kernel_regularizer=keras.regularizers.l1(0.001)))
    # model.add(keras.layers.BatchNormalization())
    # model.add(keras.layers.Dropout(0.2))

    model.add(Dense(1, kernel_initializer='glorot_normal', activation='sigmoid',
                    kernel_regularizer=keras.regularizers.l1(0.003)))
    return model


def fit_model(model, X, Y, nb_epoch=1000):
    # np.random.seed(seed)

    (X_train_validation, X_test, Y_train_validation, Y_test) = train_test_split(X, Y, test_size=0.2)
    (X_train, X_validation, Y_train, Y_validation) = train_test_split(X_train_validation, Y_train_validation,
                                                                      test_size=0.2)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(X_train, Y_train,
                        validation_data=(X_validation, Y_validation),
                        epochs=nb_epoch, batch_size=32)

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model1 accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model1 loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    scores = model.evaluate(X_test, Y_test)
    print("\n\nAccuracy in test set: %.2f%%" % (scores[1] * 100))
    return model


total_data1 = total_data.dropna(axis='columns')
myX = np.array(total_data1.iloc[:, 1:])
myY = np.array(total_data1.iloc[:, 0])

# ======================================================================================================================
model2 = get_model()
fit_model(model2, myX, myY, nb_epoch=500)

import numpy as np

X = np.array(
    [[-0.066, -0.124, 0.259, 0.289, -0.318, -0.015, -0.06, 0.14, 0.203, 0.249, -0.229, 0.174, 0.149, -0.343, -0.308],
     [-0.204, -0.079, 0.565, 0.607, -0.425, -0.135, 0.011, 0.411, 0.35, 0.535, -0.668, 0.128, 0.317, -0.543, -0.87]])
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X.T)
print(pca.explained_variance_ratio_)
print(pca.components_)
print(pca.explained_variance_)

phy = np.array([[0.1, 0.2, 0.7], [0, 0, 1], [0.5, 0.5, 0]])
