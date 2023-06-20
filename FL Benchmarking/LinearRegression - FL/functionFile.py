# functionFile.py
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
import numpy as np
from typing import Tuple, Union, List


def split_sequence(sequence, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        if out_end_ix > len(sequence):
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


# print(train_seq.shape)
# print(train_seq)
def X_data_format(train_seq, test_seq, val_seq=[], n_steps_in=3, n_steps_out=2, n_features=1):
    X_train, y_train = split_sequence(train_seq, n_steps_in, n_steps_out)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], n_features))
    x_test, y_test = split_sequence(test_seq, n_steps_in, n_steps_out)
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], n_features))
    if val_seq == []:
        x_val, y_val = [], []
    else:
        x_val, y_val = split_sequence(val_seq, n_steps_in, n_steps_out)
        x_val = x_val.reshape((x_val.shape[0], x_val.shape[1], n_features))
    return X_train, y_train, x_test, y_test, x_val, y_val


#
# for i in range(len(X_train)):
#     print(X_train[i], y_train[i])
#
# for i in range(len(x_test)):
#     print(x_test[i], y_test[i])

def simpleLSTM(X_train, n_neurons, y_train, x_test, y_test, n_epochs, n_steps_in, n_steps_out, n_features):
    model = Sequential()
    model.add(LSTM(n_neurons, activation='relu', return_sequences=False, input_shape=(n_steps_in, n_features)))
    model.add(Dense(n_steps_out))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=n_epochs, verbose=0)
    yhat = model.predict(x_test, verbose=0)
    results = model.evaluate(x_test, y_test)
    print("----", results)
    return yhat


# for i in range(len(y_test)):
#     print(y_test[i], yhat[i], '---\n')
def metrics_call(pred_type, y_test, yhat):
    mse_metric = mean_squared_error(y_test, yhat)
    mae_metric = mean_absolute_error(y_test, yhat)
    r2_metric = r2_score(y_test, yhat, multioutput='variance_weighted')
    print(pred_type, '--', "mse:", mse_metric, '--', "mae:", mae_metric, '--', "r2:", r2_metric)


XY = Tuple[np.ndarray, np.ndarray]
Dataset = Tuple[XY, XY]
LinRegParams = Union[XY, Tuple[np.ndarray]]
XYList = List[XY]


def get_model_parameters(model: LinearRegression) -> LinRegParams:
    """Returns the paramters of a sklearn LinearRegression model."""
    if model.fit_intercept:
        params = [
            model.coef_,
            model.intercept_,
        ]
    else:
        params = [
            model.coef_,
        ]
    return params


def set_model_params(
        model: LinearRegression, params: LinRegParams
) -> LinearRegression:
    """Sets the parameters of a sklean LinearRegression model."""
    model.coef_ = params[0]
    if model.fit_intercept:
        model.intercept_ = params[1]
    return model


def set_initial_params(model: LinearRegression):
    """Sets initial parameters as zeros Required since model params are
    uninitialized until model.fit is called.

    But server asks for initial parameters from clients at launch. Refer
    to sklearn.linear_model.LinearRegression documentation for more
    information.
    """
    n_steps_in = 100
    n_steps_out = 50
    model.coef_ = np.zeros((n_steps_out, n_steps_in))
    if model.fit_intercept:
        model.intercept_ = np.zeros((n_steps_out,))
