import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import functionFile
import time
from datetime import timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error

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

# print(arr.shape[0] * 0.7)
# print(arr.shape[0] * 0.2)
# print(arr.shape[0] * 0.1)
# arr_train = arr[0:int(arr.shape[0] * 0.7)]
# x_train = np.concatenate((arr_train, arr_train), axis=0)
# print(x_train.shape)
# print(arr.shape)
# X_train, y_train = functionFile.split_sequence(arr_train, n_steps_in, n_steps_out)
# x_test, y_test = functionFile.split_sequence(test_arr, n_steps_in, n_steps_out)
# x_train = np.concatenate((X_train, X_train), axis=0)
# print(x_train.shape, y_train.shape)

model = LinearRegression()
start_time = time.monotonic()
model.fit(train_X, train_y)
end_time = time.monotonic()
yhat = model.predict(test_X)
print(yhat)
functionFile.metrics_call("test",test_y, yhat)
yval = model.predict(val_X)
functionFile.metrics_call("val",val_y, yval)
print(timedelta(seconds=end_time - start_time))
