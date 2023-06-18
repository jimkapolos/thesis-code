import argparse
import os
import pandas as pd
import functionFile
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional
# from sklearn.metrics import mean_squared_error, mean_absolute_error

import tensorflow as tf

import flwr as fl

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# Define Flower client
class CifarClient(fl.client.NumPyClient):
    def __init__(self, model, epochs, x_train, y_train, x_test, y_test):  # TODO add validation data x_val, y_val
        self.model = model
        self.epochs = epochs
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test
        # self.x_val, self.y_val = x_val, y_val  # TODO add validation data

    def get_properties(self, config):
        """Get properties of client."""
        raise Exception("Not implemented")

    def get_parameters(self, config):
        """Get parameters of the local model."""
        raise Exception("Not implemented (server-side parameter initialization)")

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""
        # print("params...............................", parameters)
        # Update local model parameters
        self.model.set_weights(parameters)

        # Get hyperparameters for this round
        # batch_size: int = config["batch_size"] # changed commented out
        # epochs: int = config["local_epochs"]
        # Train the model using hyperparameters from config
        history = self.model.fit(
            self.x_train,
            self.y_train,
            # batch_size, # changed commented out
            epochs=self.epochs,
            validation_split=0.1,
            # validation_data=(self.x_val, self.y_val),
            verbose=1
        )

        # Return updated model parameters and results
        parameters_prime = self.model.get_weights()
        num_examples_train = len(self.x_train)
        results = {
            "loss": history.history["loss"][0],
            "mean_squared_error": history.history["mean_squared_error"][0],
            "mean_absolute_error": history.history["mean_absolute_error"][0],
            # changed from accuracy to mean_squared_error
            "val_loss": history.history["val_loss"][0],
            "val_mean_squared_error": history.history["val_mean_squared_error"][0],
            "val_mean_absolute_error": history.history["val_mean_absolute_error"][0],
            # changed from val_accuracy to val_mean_squared_error
        }
        # print("x_test\n", self.x_test, "\ny_test\n", self.y_test)
        # yhat = self.model.predict(self.x_test)
        # print("predict:", yhat)
        # print("mae", sklearn.metrics.mean_absolute_error(yhat, self.y_test))
        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        # Update local model with global parameters
        self.model.set_weights(parameters)

        # Get config values
        # steps: int = config["val_steps"]

        # Evaluate global model parameters on the local test data and return results
        loss, mean_squared_error, mean_absolute_error = self.model.evaluate(self.x_test, self.y_test)  # , steps=steps)
        num_examples_test = len(self.x_test)

        return loss, num_examples_test, {"mean_squared_error": mean_squared_error,
                                         "mean_absolute_error": mean_absolute_error}


def main() -> None:
    # Parse command line argument `partition`
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument("--partition", type=int, choices=range(0, 22), required=True)  # changed from 0 10 to 0 2
    parser.add_argument("--n_neurons", type=int, choices=range(0, 1000), required=True)
    parser.add_argument("--n_epochs", type=int, choices=range(0, 1000), required=True)
    args = parser.parse_args()

    # Load and compile Keras model
    model = Sequential()
    model.add(
        Bidirectional(LSTM(args.n_neurons, activation='tanh', return_sequences=False), input_shape=(n_steps_in, n_features)))
    model.add(Dense(n_steps_out))
    model.compile(optimizer='adam', loss='huber',
                  metrics=[tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()
                           ])
    # print("sumary......................",model.summary())

    (x_train, y_train), (x_test, y_test) = load_partition(args.partition)

    # Start Flower client
    client = CifarClient(model, args.n_epochs, x_train, y_train, x_test, y_test)

    fl.client.start_numpy_client(
        server_address="127.0.0.1:8080",
        client=client,
        # root_certificates=Path(".cache/certificates/ca.crt").read_bytes(),
    )


def load_partition(idx: int):
    """Load 1/10th of the training and test data to simulate a partition."""
    assert idx in range(23)  # changed from 10 to 2

    list_of_csv = [1, 9, 37, 90, 185, 216, 228, 230, 234, 251, 274, 275, 283, 296, 297, 305, 317, 329, 332, 380, 381,
                   385]
    arr = pd.read_csv('~/Downloads/rnd/2013-7/{}.csv'.format(list_of_csv[idx]), sep=';	', engine='python',
                      usecols=['CPU usage [%]']).to_numpy().flatten()

    arr_train = arr[0:int(arr.shape[0] * 0.7)]
    arr_test = arr[int(arr.shape[0] * 0.7):int(arr.shape[0] * 0.9)]

    aa_train_X, aa_train_y = functionFile.split_sequence(arr_train, n_steps_in, n_steps_out)
    aa_test_X, aa_test_y = functionFile.split_sequence(arr_test, n_steps_in, n_steps_out)

    aa_train_X.reshape((aa_train_X.shape[0], aa_train_X.shape[1], n_features))
    aa_test_X.reshape((aa_test_X.shape[0], aa_test_X.shape[1], n_features))

    return (aa_train_X, aa_train_y), (aa_test_X, aa_test_y)


if __name__ == "__main__":
    # changes in those values bust be also performed in the server
    n_steps_in = 3
    n_steps_out = 2
    n_features = 1
    n_neurons = 10
    # n_epochs = 3
    main()
