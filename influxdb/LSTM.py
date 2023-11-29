from numpy import array
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import time
from pandas import read_csv
import warnings
import wx
import matplotlib.pyplot as plt


def load_data(file_path):
    series = read_csv(file_path)
    values = series["CPU usage [%]"].values
    return values


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


def create_lstm_model(n_steps, n_features, n_outs):
    model = Sequential()
    model.add(LSTM(100, activation="relu", return_sequences=False, input_shape=(n_steps, n_features)))
    model.add(Dense(n_outs))
    model.compile(optimizer=Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07), loss='mse',
                  metrics=['accuracy', 'mse'])
    return model


def train_model(model, X_train, y_train, X_valid, y_valid, epochs, batch_size):
    start_time = time.time()
    history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_valid, y_valid), batch_size=batch_size)
    end_time = time.time()
    print("Χρόνος εκτέλεσης: %s δευτερόλεπτα" % (end_time - start_time))
    return model, history


def save_model(model, model_json_path, model_weights_path):
    # Serialize model to JSON
    model_json = model.to_json()
    with open(model_json_path, "w") as json_file:
        json_file.write(model_json)
    # Serialize weights to HDF5
    model.save_weights(model_weights_path)
    print("Saved model to disk")


def plot_loss_accuracy(history):
    warnings.filterwarnings("ignore", category=UserWarning, module="wx")
    app = wx.App(True)
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['loss', 'val_loss'])
    plt.xlabel("epochs")
    plt.ylabel("loss")

    plt.subplot(2, 1, 2)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.legend(['accuracy', 'val_accuracy'])
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.show()


if __name__ == "__main__":
    # Load data
    file_path = "1.csv"
    values = load_data(file_path)

    # Choose a number of time steps
    n_steps = 3
    n_outs = 1

    # Split into samples
    X, y = split_sequence(values, n_steps, n_outs)

    # Reshape data
    n_features = 1
    X = X.reshape(X.shape[0], X.shape[1], n_features)

    # Split data into train, test, and validation sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=False)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.125, shuffle=False)

    # Create and compile model
    model = create_lstm_model(n_steps, n_features, n_outs)

    # Train model
    EPOCHS = 30
    BATCH_SIZE = round(len(X) / 2)
    trained_model, history = train_model(model, X_train, y_train, X_valid, y_valid, EPOCHS, BATCH_SIZE)

    # Save model
    model_json_path = "LSTM.json"
    model_weights_path = "LSTM.h5"
    save_model(trained_model, model_json_path, model_weights_path)

    # Plot loss and accuracy

    plot_loss_accuracy(history)
