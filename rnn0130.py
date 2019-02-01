import rqdatac as rd
import pandas as pd
import datetime
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Input, Dense, GRU, Embedding
from tensorflow.python.keras.initializers import RandomUniform
from tensorflow.python.keras.optimizers import RMSprop
import tensorflow as tf
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.utils import to_categorical

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
y_data= to_categorical(y_data)  # 第一列代表是否跌，第二列代表是否涨
spot_data1.columns = list(map(lambda x: 'spot_' + x, spot_data1.columns))
future_data1.columns = list(map(lambda x: 'future_' + x, future_data1.columns))
x_data = pd.DataFrame.merge(spot_data1, future_data1, left_index=True, right_index=True)

x_data.iloc[1:, :-1] = x_data.iloc[1:, :-1].values/x_data.iloc[:-1, :-1].values-1
x_data.iloc[0, :-1] = 0
# 数据处理完毕，接下来就是丢进模型了！


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

# y_data = np.array(y_data)
# y_data = y_data.reshape(-1, 1)
y_train = y_data[0:num_train]
y_vali = y_data[num_train:num_train + num_vali]
y_test = y_data[num_train + num_vali:]

num_x_signals = x_data.shape[1]
num_x_signals  # this is the number of input-signals

num_y_signals = y_data.shape[1]
num_y_signals


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
            x_batch[i] = x_train[idx:idx + sequence_length]
            y_batch[i] = y_train[idx:idx + sequence_length]

        yield (x_batch, y_batch)

batch_size = 2
sequence_length = 25

generator = batch_generator(batch_size=batch_size,
                            sequence_length=sequence_length)
x_batch, y_batch = next(generator)

batch = 0   # First sequence in the batch.
signal = 0  # First signal from the 20 input-signals.
seq = x_batch[batch, :, signal]
plt.plot(seq)

validation_data = (np.expand_dims(x_vali, axis=0),
                   np.expand_dims(y_vali, axis=0))

model = Sequential()
model.add(GRU(units=8,
              return_sequences=True,
              input_shape=(None, num_x_signals,), kernel_regularizer=tf.keras.regularizers.l1(0.01)))

init = RandomUniform(minval=-0.05, maxval=0.05)
model.add(Dense(num_y_signals * 3,
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l1(0.01)))

model.add(Dense(num_y_signals * 2,
                activation='relu',
                kernel_regularizer=tf.keras.regularizers.l1(0.01)))

model.add(Dense(num_y_signals,
                activation='softmax',
                kernel_regularizer=tf.keras.regularizers.l1(0.01)))

optimizer = RMSprop(lr=1e-3)
model.compile(loss=tf.losses.softmax_cross_entropy, optimizer=optimizer)
model.summary()

path_checkpoint = '23_checkpoint.keras'
callback_checkpoint = ModelCheckpoint(filepath=path_checkpoint,
                                      monitor='val_loss',
                                      verbose=1,
                                      save_weights_only=True,
                                      save_best_only=True)
callback_early_stopping = EarlyStopping(monitor='val_loss',
                                        patience=12, verbose=1)
callback_tensorboard = TensorBoard(log_dir='./23_logs/',
                                   histogram_freq=0,
                                   write_graph=False)
callback_reduce_lr = ReduceLROnPlateau(monitor='loss',
                                       factor=0.8,
                                       min_lr=1e-05,
                                       patience=2,
                                       verbose=1)
callbacks = [callback_early_stopping,
             callback_checkpoint,
             callback_tensorboard,
             callback_reduce_lr]

model.fit_generator(generator=generator,
                    epochs=100,
                    steps_per_epoch=10,
                    validation_data=validation_data,
                    callbacks=callbacks)

model.load_weights(path_checkpoint)





