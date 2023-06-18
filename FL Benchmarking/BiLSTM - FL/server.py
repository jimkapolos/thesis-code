from typing import Dict, Optional, List, Tuple
import argparse
# from pathlib import Path
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional
import flwr as fl
import tensorflow as tf
import numpy as np
from numpy import array, stack
import functionFile
from functionFile import split_sequence
from flwr.common import Metrics
import pandas as pd


def main() -> None:
    # Parse command line argument `partition`
    parser = argparse.ArgumentParser(description="Flower Server")
    parser.add_argument("--rounds", type=int, choices=range(0, 100), required=True)
    parser.add_argument("--n_neurons", type=int, choices=range(0, 1000), required=True)
    args = parser.parse_args()

    # Load and compile model for
    # 1. server-side parameter initialization
    # 2. server-side parameter evaluation
    model = Sequential()
    model.add(
        Bidirectional(LSTM(args.n_neurons, activation='tanh', return_sequences=False),
                      input_shape=(n_steps_in, n_features)))
    model.add(Dense(n_steps_out))
    model.compile(optimizer='adam', loss='huber',
                  metrics=[tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.MeanAbsoluteError()
                           ])

    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.8,
        fraction_evaluate=0.7,
        min_fit_clients=15,  # changed commented out
        min_evaluate_clients=10,
        min_available_clients=20,
        evaluate_fn=get_evaluate_fn(model),
        on_fit_config_fn=fit_config,
        # on_evaluate_config_fn=evaluate_config,
        initial_parameters=fl.common.ndarrays_to_parameters(model.get_weights()),
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    # Start Flower server (SSL-enabled) for four rounds of federated learning
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=args.rounds),  # changed from 4 to 2
        strategy=strategy,
        # certificates=(
        #     Path(".cache/certificates/ca.crt").read_bytes(),
        #     Path(".cache/certificates/server.pem").read_bytes(),
        #     Path(".cache/certificates/server.key").read_bytes(),
        # ),
    )


def get_evaluate_fn(model):
    """Return an evaluation function for server-side evaluation."""

    # # Load data and model here to avoid the overhead of doing it in `evaluate` itself
    # (x_train, y_train), _ = tf.keras.datasets.cifar10.load_data() # model changed
    # # Use the last 5k training examples as a validation set
    # x_val, y_val = x_train[450:500], y_train[450:500]
    # # x_val, y_val = x_train[45000:50000], y_train[45000:50000] #changed from 45000 and 50000 to 450 and 500

    list_of_csv = [1, 9, 37, 90, 185, 216, 228, 230, 234, 251, 274, 275, 283, 296, 297, 305, 317, 329, 332, 380, 381,
                   385]

    for i in list_of_csv:
        arr = pd.read_csv('~/Downloads/rnd/2013-7/{}.csv'.format(i), sep=';	', engine='python',
                          usecols=['CPU usage [%]']).to_numpy().flatten()
        arr_val = arr[int(arr.shape[0] * 0.9):int(arr.shape[0])]
        va_X, va_y = functionFile.split_sequence(arr_val, n_steps_in, n_steps_out)
        if i == 1:
            val_X, val_y = va_X, va_y
        else:
            val_X = np.concatenate((val_X, va_X), axis=0)
            val_y = np.concatenate((val_y, va_y), axis=0)
        # print(val_X.shape, val_y.shape)

    val_X.reshape((val_X.shape[0], val_X.shape[1], n_features))


    def evaluate(
            server_round: int,
            parameters: fl.common.NDArrays,
            config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        model.set_weights(parameters)  # Update model with the latest parameters
        loss, mean_squared_error, mean_absolute_error = model.evaluate(val_X, val_y)
        return loss, {"mean_squared_error": mean_squared_error, "mean_absolute_error": mean_absolute_error}

    return evaluate


def fit_config(server_round: int):
    """Return training configuration dict for each round.

    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
    config = {
        # "batch_size": 32, # changed commented out
        # "local_epochs": 1 if server_round < 2 else 2,
        "local_epochs": 5
    }
    return config


def evaluate_config(server_round: int):
    """Return evaluation configuration dict for each round.

    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to ten local
    evaluation steps.
    """
    val_steps = None  # changed from the one below to None
    # val_steps = 5 if server_round < 4 else 10
    return {"val_steps": val_steps}


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    mean_squared_error = [num_examples * m["mean_squared_error"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    mean_absolute_error = [num_examples * m["mean_absolute_error"] for num_examples, m in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"mean_squared_error": sum(mean_squared_error) / sum(examples),
            "mean_absolute_error": sum(mean_absolute_error) / sum(examples)}


if __name__ == "__main__":
    # changes in those values bust be also performed in the client
    n_steps_in = 3
    n_steps_out = 2
    n_features = 1
    # n_neurons = 10
    n_epochs = 3
    main()
