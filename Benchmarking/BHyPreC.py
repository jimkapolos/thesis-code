import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense,MaxPooling1D,Bidirectional,GRU
from keras.optimizers import Adam
from keras.layers.convolutional import Conv1D
import functionFile
import time
from datetime import timedelta


n_neurons = 10
n_epochs = 5
n_steps_in = 3
n_steps_out = 2
n_features = 1

list_of_csv = [1, 9, 37, 90, 185, 216, 228, 230, 234, 251, 274, 275, 283, 296, 297, 305, 317, 329, 332, 380, 381, 385]

for i in list_of_csv:
    arr = pd.read_csv('~/Downloads/rnd/2013-7/{}.csv'.format(i), sep=';	', engine='python',
                      usecols=['CPU usage [%]']).to_numpy().flatten()
    arr_train = arr[0:int(arr.shape[0] * 0.7)]
    arr_test = arr[int(arr.shape[0] * 0.7):int(arr.shape[0] * 0.9)]
    arr_val = arr[int(arr.shape[0] * 0.9):int(arr.shape[0])]
    tr_X, tr_y = functionFile.split_sequence(arr_train, n_steps_in, n_steps_out)
    te_X, te_y = functionFile.split_sequence(arr_test, n_steps_in, n_steps_out)
    va_X, va_y = functionFile.split_sequence(arr_val, n_steps_in, n_steps_out)
    if i == 1:
        train_X, train_y = tr_X, tr_y
        test_X, test_y = te_X, te_y
        val_X, val_y = va_X, va_y
    else:
        train_X = np.concatenate((train_X, tr_X), axis=0)
        train_y = np.concatenate((train_y, tr_y), axis=0)
        test_X = np.concatenate((test_X, te_X), axis=0)
        test_y = np.concatenate((test_y, te_y), axis=0)
        val_X = np.concatenate((val_X, va_X), axis=0)
        val_y = np.concatenate((val_y, va_y), axis=0)
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape, val_X.shape, val_y.shape)

train_X.reshape((train_X.shape[0], train_X.shape[1], n_features))
test_X.reshape((test_X.shape[0], test_X.shape[1], n_features))
val_X.reshape((val_X.shape[0], val_X.shape[1], n_features))

model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='tanh', input_shape=(n_steps_in, n_features), strides=1))  # first values is neurones
model.add(MaxPooling1D(pool_size=(1,)))
model.add(Bidirectional(LSTM(64, activation='tanh', return_sequences=True)))
model.add(Bidirectional(GRU(64, activation='tanh', return_sequences=True)))
model.add(LSTM(64, activation='tanh', return_sequences=True))
model.add(LSTM(64, activation='tanh', return_sequences=True))
model.add(GRU(64, activation='tanh', return_sequences=True))
model.add(LSTM(100, activation='tanh'))
model.add(Dense(n_steps_out))
model.compile(optimizer=Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07), loss='huber',
              metrics=[tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()])


start_time = time.monotonic()
model.fit(train_X, train_y, epochs=n_epochs, verbose=1, use_multiprocessing=True)
end_time = time.monotonic()
yhat = model.predict(test_X, verbose=0)
results = model.evaluate(test_X,test_y)
print(results)
functionFile.metrics_call("test",test_y, yhat)
yval = model.predict(val_X)
functionFile.metrics_call("val",val_y, yval)
print(timedelta(seconds=end_time - start_time))