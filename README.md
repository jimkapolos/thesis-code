# thesis-code

from the model code folder is an explanation of the models.py and models_main.py files

explaine models_main.py
This code defines several functions that are useful for loading, preparing, training, and evaluating a deep learning model for time series forecasting. Here is a summary of the main functions:

load_data(filepath, target_col): loads data from a CSV file and returns the target column as a 1D numpy array.

split_sequence(sequence, n_steps_in, n_steps_out): takes a 1D numpy array representing a time series and splits it into samples, where each sample has a specified number of input time steps (n_steps_in) and output time steps (n_steps_out). Returns two arrays: X, which contains the input sequences, and y, which contains the corresponding output sequences.

prepare_data(values, n_steps, n_outs): uses split_sequence() to split a 1D numpy array of time series values into samples of n_steps input time steps and n_outs output time steps. Returns two arrays: X, which contains the input sequences reshaped into a 3D numpy array, and y, which contains the corresponding output sequences.

train_test_valid_split_data(X, y, test_size, valid_size, shuffle): splits the input and output data into training, validation, and test sets. Returns six arrays: X_train, X_valid, X_test, y_train, y_valid, y_test.

train_model(model, X_train, y_train, X_valid, y_valid, epochs, batch_size): trains a Keras deep learning model on the input and output data. Returns a history object containing information about the training process.

save_model(model, model_path, weights_path): saves a Keras model and its weights to disk.

evaluate_model(model, X_evaluate, y_evaluate): evaluates the performance of a trained Keras model on the given input and output data.

In addition to these functions, the code imports several Python packages, including numpy, pandas, sklearn, and matplotlib, and defines a few helper functions.

explaine models.py

These are different architectures of neural networks designed for sequence prediction tasks using LSTMs, GRUs, and Conv1D layers.

The first architecture (build_model1) is a simple LSTM with three stacked layers of decreasing size, followed by a dense layer to produce the output. 
It uses the "relu" activation function, "mse" loss function, and "adam" optimizer.

The second architecture (build_model2) is a more complex LSTM model that uses Bidirectional LSTMs and GRUs.
It has six layers, including two Bidirectional LSTMs and one Bidirectional GRU, followed by two additional LSTMs and a dense layer to produce the output. 
It also includes a MaxPooling1D layer to reduce the number of parameters. This model uses the "relu" activation function, "mse" loss function, and "adam" optimizer.

The third architecture (build_model3) uses Conv1D layers, Bidirectional LSTMs, and GRUs. 
It has six layers, including a Conv1D layer, three LSTMs (one of them Bidirectional), and two GRUs (one of them Bidirectional), followed by a dense layer to produce the output. 
It uses the "relu" activation function, "mse" loss function, and "Adam" optimizer with a customized learning rate, beta, and epsilon.

The fourth architecture (build_model4) uses Bidirectional LSTMs and a regular LSTM layer. It has three layers, including two Bidirectional LSTMs and one LSTM, followed by a dense layer to produce the output. 
It uses the "relu" activation function, "mse" loss function, and "Adam" optimizer with a customized learning rate, beta, and epsilon.
