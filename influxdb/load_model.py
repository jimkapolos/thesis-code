from keras.models import model_from_json
import os
from numpy import array
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from data_importer import query_cpu_metrics, connect_to_influxdb

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def split_sequence(sequence, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the sequence
        if out_end_ix > len(sequence):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


def load_lstm_model():
    # load model
    json_file = open('LSTM.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("LSTM.h5")

    print("LOAD MODEL FROM DISK")

    return loaded_model


def predict_with_lstm(model, X, y):
    # predicted
    X = X.reshape(X.shape[0], X.shape[1], n_features)
    yhat = model.predict(X).flatten()
    yhat = np.ravel(yhat)
    y = np.ravel(y)
    results = pd.DataFrame({'Prediction': yhat, 'Actual': y})
    print("result", results)

    return yhat


def metrics(y, yhat):
    mse_metric = mean_squared_error(y, yhat)
    mae_metric = mean_absolute_error(y, yhat)
    r2_metric = r2_score(y, yhat, multioutput='variance_weighted')
    print('--', "mse:", mse_metric, '--', "mae:", mae_metric, '--', "r2:", r2_metric)


if __name__ == "__main__":
    # Call the function from the imported module to get the Series
    # Connect to InfluxDB and get the query API
    INFLUXDB_HOST = 'localhost'
    INFLUXDB_PORT = 8086
    INFLUXDB_DATABASE = 'my_dataset'
    INFLUXDB_USER = 'jimkap'
    INFLUXDB_ORG = 'UP'
    INFLUXDB_TOKEN = "BJGdSkXbyCY7h1wjxbPJv2ZzzhdY-AlUyYA1yOlvOrAhyV_ymGq-O3iwX07leV3kClLALhCz0m-p6oWPtKHQmg=="

    query_api = connect_to_influxdb(INFLUXDB_HOST, INFLUXDB_PORT, INFLUXDB_ORG, INFLUXDB_TOKEN)
    cpu_metrics_series = query_cpu_metrics(query_api, INFLUXDB_ORG)

    series = cpu_metrics_series
    series = series.values

    # split into samples
    n_steps = 3
    n_features = 1
    n_out = 1
    X, y = split_sequence(series, n_steps, n_out)

    # Load LSTM model
    lstm_model = load_lstm_model()

    # Use the loaded model for prediction
    yhat = predict_with_lstm(lstm_model, X, y)

    # Calculate and print metrics
    metrics(y, yhat)
