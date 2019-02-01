import rqdatac as rd
import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
import keras
from sklearn.preprocessing import MinMaxScaler

rd.init()

spot_data1 = rd.get_price('510050.XSHG', start_date='2015-06-01', end_date='2019-01-29', frequency='1d')
future_data1 = rd.get_price('IH88', start_date='2015-06-01', end_date='2019-01-29', frequency='1d')

spot_data1 = pd.DataFrame(spot_data1)
future_data1 = pd.DataFrame(future_data1)
# future_data1都是50ETF期货主力合约信息，最好能把主力合约距离到期日的天数囊括进去
future_data1['dominant'] = rd.futures.get_dominant('IF', start_date='2015-06-01', end_date='2019-01-29')
future_data1['delisted_date'] = list(map(lambda x: datetime.datetime.strptime(rd.instruments(x).de_listed_date,
                                                                              '%Y-%m-%d'), future_data1['dominant']))
future_data1['remaining_days'] = future_data1['delisted_date'] - future_data1.index
future_data1['remaining_days'] = list(map(lambda x: x.days, future_data1['remaining_days']))
future_data1.drop(columns=['dominant', 'delisted_date'], inplace=True)

# 获取滞后了1天的数据
spot_data2 = rd.get_price('510050.XSHG', start_date='2015-06-02', end_date='2019-01-30', frequency='1d')
y_data = spot_data2['close'].values > spot_data1['close'].values
y_data = to_categorical(y_data)  # 第一列代表是否跌，第二列代表是否涨
spot_data1.columns = list(map(lambda x: 'spot_' + x, spot_data1.columns))
future_data1.columns = list(map(lambda x: 'future_' + x, future_data1.columns))
x_data = pd.DataFrame.merge(spot_data1, future_data1, left_index=True, right_index=True)


# 数据处理完毕，接下来就是丢进模型了！

# x_data = spot_data1.copy()
# 对数据进行划分
shift_days = 1
num_data = len(x_data)
# 数据量太少了 才不到900个
train_split = 0.6
num_train = int(train_split * num_data)
vali_split = 0.2
num_vali = int(vali_split * num_data)
num_test = num_data - num_train - num_vali

x_train = x_data[0:num_train]
x_vali = x_data[num_train:num_train + num_vali]
x_test = x_data[num_train + num_vali:]

y_train = y_data[0:num_train]
y_vali = y_data[num_train:num_train + num_vali]
y_test = y_data[num_train + num_vali:]

num_x_signals = x_data.shape[1]
num_x_signals  # this is the number of input-signals

num_y_signals = y_data.shape[1]
num_y_signals
x_scaler = MinMaxScaler()
x_train_scaled = x_scaler.fit_transform(x_train)
x_vali_scaled = x_scaler.transform(x_vali)
x_test_scaled = x_scaler.transform(x_test)


def batch_generator(batch_size, sequence_length):
    """
    Generator function for creating random batches of training-data.
    """

    # Infinite loop.
    while True:
        # Allocate a new array for the batch of input-signals.
        x_shape = (batch_size, sequence_length, num_x_signals)
        x_batch = np.zeros(shape=x_shape, dtype=np.float16)

        # Allocate a new array for the batch of output-signals.
        y_shape = (batch_size, sequence_length, num_y_signals)
        y_batch = np.zeros(shape=y_shape, dtype=np.float16)

        # Fill the batch with random sequences of data.
        for i in range(batch_size):
            # Get a random start-index.
            # This points somewhere into the training-data.
            idx = np.random.randint(num_train - sequence_length)

            # Copy the sequences of data starting at this index.
            x_batch[i] = x_train_scaled[idx:idx + sequence_length]
            y_batch[i] = y_train[idx:idx + sequence_length]

        yield (x_batch, y_batch)


batch_size = 8
sequence_length = 15

generator = batch_generator(batch_size=batch_size,
                            sequence_length=sequence_length)
x_batch, y_batch = next(generator)

batch = 0  # First sequence in the batch.
signal = 0  # First signal from the 20 input-signals.
seq = x_batch[batch, :, signal]
plt.plot(seq)

validation_data = (np.expand_dims(x_vali_scaled, axis=0),
                   np.expand_dims(y_vali, axis=0))

model = keras.Sequential()
# model.add(keras.layers.BatchNormalization(input_shape=(None, num_x_signals,)))
init = keras.initializers.RandomUniform(minval=-1, maxval=1)
model.add(keras.layers.GRU(units=6, input_shape=(None, num_x_signals,),
                           return_sequences=True,
                           kernel_initializer=init,
                           kernel_regularizer=keras.regularizers.l1(0.001)))

model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Dense(num_y_signals * 3,
                             activation='relu',
                             kernel_initializer=init,
                             kernel_regularizer=keras.regularizers.l1(0.01)))

model.add(keras.layers.BatchNormalization())
#
model.add(keras.layers.Dense(num_y_signals * 2,
                             activation='relu',
                             kernel_initializer=init,
                             kernel_regularizer=keras.regularizers.l1(0.01)))

model.add(keras.layers.BatchNormalization())

model.add(keras.layers.Dense(num_y_signals,
                             activation='softmax',
                             kernel_initializer=init,
                             kernel_regularizer=keras.regularizers.l1(0.001)))

optimizer = keras.optimizers.RMSprop(lr=1e-3)

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer)
model.summary()

path_checkpoint = '23_checkpoint.keras'
callback_checkpoint = keras.callbacks.ModelCheckpoint(filepath=path_checkpoint,
                                                      monitor='val_loss',
                                                      verbose=1,
                                                      save_weights_only=True,
                                                      save_best_only=True)
callback_early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                        patience=12, verbose=1)
callback_tensorboard = keras.callbacks.TensorBoard(log_dir='./23_logs/',
                                                   histogram_freq=0,
                                                   write_graph=False)
callback_reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss',
                                                       factor=0.8,
                                                       min_lr=1e-05,
                                                       patience=3,
                                                       verbose=1)
callbacks = [callback_early_stopping,
             callback_checkpoint,
             callback_tensorboard,
             callback_reduce_lr]

history = model.fit_generator(generator=generator,
                    epochs=100,
                    steps_per_epoch=16,
                    validation_data=validation_data,
                    callbacks=callbacks)

model.load_weights(path_checkpoint)


test_set_prediction = model.predict_classes(np.expand_dims(x_test_scaled, axis=0))[0]
true_test_label = np.argmax(y_test, axis=1)





# accuracy when market go down
np.sum(true_test_label[true_test_label == 0] == test_set_prediction[true_test_label == 0]) / len(true_test_label[true_test_label == 0])
# 2 / 126 = 0.016

# accuracy when market go up
np.sum(true_test_label[true_test_label == 1] == test_set_prediction[true_test_label == 1]) / len(true_test_label[true_test_label == 1])
# 109 / 111 = 0.982

# accuracy when predicting market will go down
np.sum(true_test_label[test_set_prediction == 0] == test_set_prediction[test_set_prediction == 0]) / len(test_set_prediction[test_set_prediction == 0])
# 2 / 4 = 0.5

# accuracy when prediction market will go up
np.sum(true_test_label[test_set_prediction == 1] == test_set_prediction[test_set_prediction == 1]) / len(test_set_prediction[test_set_prediction == 1])
# 109 / 233 = 0.468

# overall accuracy
np.sum(true_test_label == test_set_prediction) / len(true_test_label)
# 0.468




# from keras import backend as K
#
# inp = model.input                                           # input placeholder
# outputs = [layer.output for layer in model.layers]          # all layer outputs
# functors = [K.function([inp], [out]) for out in outputs]    # evaluation functions
#
# # Testing
# test = np.expand_dims(x_data, axis=0)
# layer_outs = [func([test]) for func in functors]
# print(layer_outs)