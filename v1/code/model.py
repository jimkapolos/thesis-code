import os
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np


def split_sequence(sequence, n_steps):
    """Split a univariate sequence into samples"""
    X, y = list(), list()
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence) - 1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def create_and_train_lstm_model(output_dir):
    """Create, train, and save the LSTM model"""
    # input data
    raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90]

    # prepare data
    n_steps = 3
    X, y = split_sequence(raw_seq, n_steps)
    n_features = 1
    X = X.reshape((X.shape[0], X.shape[1], n_features))

    # model
    model = Sequential()
    model.add(LSTM(100, activation='relu', input_shape=(n_steps, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    # train model
    model.fit(X, y, epochs=200, verbose=0)

    # save model
    os.makedirs(output_dir, exist_ok=True)
    # model_path = os.path.join(output_dir, "model.keras")
    # model.save(model_path, save_format="keras")
    saved_model_path = os.path.join(output_dir, "saved_model/1")
    tf.saved_model.save(model, saved_model_path)
    print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    output_dir = "model_output"
    create_and_train_lstm_model(output_dir)
