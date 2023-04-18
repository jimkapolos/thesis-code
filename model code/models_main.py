from numpy import array
from sklearn.model_selection import train_test_split
from pandas import read_csv
import matplotlib.pyplot as plt
import time
import models

def load_data(filepath, target_col):
    ''' load data from csv file'''
    series = read_csv(filepath)
    values = series[target_col].values
    return values

def split_sequence(sequence, n_steps_in, n_steps_out):
    ''' The split_sequence() function below implements this behavior
    and will split a given univariate sequence into multiple samples where each sample has a specified number of time steps and the output is a single time step.'''
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

def prepare_data(values, n_steps, n_outs):
    '''The function first calls a helper function split_sequence() to split the values array into samples of n_steps input time steps and n_outs output time steps.
    The inputs and outputs are returned as separate arrays X and y.

        Next, the function reshapes the X array from a 2-dimensional array with shape (samples, timesteps) to a 3-dimensional array with shape (samples, timesteps, features).
        Here, features is set to 1 since the input only has one feature (i.e. univariate time series).'''
    # split into samples
    X, y = split_sequence(values, n_steps, n_outs)
    # reshape from [samples, timesteps] into [samples, timesteps, features]
    n_features = 1
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    return X, y

def train_test_valid_split_data(X, y, test_size, valid_size, shuffle):
    "Split data to 70% train , 10% validation,20% test"
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=shuffle)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=valid_size, shuffle=shuffle)
    return X_train, X_valid, X_test, y_train, y_valid, y_test

def train_model(model, X_train, y_train, X_valid, y_valid, epochs, batch_size):
    ''' 1.model: a Keras model object that has been defined and compiled.
        2.X_train: a 3-dimensional numpy array representing the input training data.
        3.y_train: a 2-dimensional numpy array representing the output training data.
        4.X_valid: a 3-dimensional numpy array representing the input validation data.
        5.y_valid: a 2-dimensional numpy array representing the output validation data.
        6.epochs: an integer representing the number of times to iterate over the entire training dataset.
        7.batch_size: an integer representing the number of samples to use for each update of the model weights.

        The function then fits the model to the training data using the fit() method of the model object, specifying the number of epochs, batch_size, and validation data X_valid and y_valid.

        The function also measures the execution time of the training process using the time module and prints it to the console.
        '''
    start_time = time.time()
    history = model.fit(X_train, y_train, epochs=epochs,validation_data=(X_valid, y_valid),batch_size=batch_size)
    end_time = time.time()
    train_execution_time = end_time - start_time
    print("Χρόνος εκτέλεσης: %s δευτερόλεπτα" % (train_execution_time))
    return history

def save_model(model, model_path, weights_path):
    ''' The function first uses the to_json() method of the Keras model object to serialize the model architecture to a JSON string.
        It then opens the specified model_path file in write mode and writes the JSON string to the file.
        Next, the function uses the save_weights() method of the Keras model object to save the model weights to the specified weights_path file in HDF5 format.'''
    # serialize model to JSON
    model_json = model.to_json()
    with open(model_path, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(weights_path)
    print("Saved model to disk")

def evaluate_model(model, X_evaluate, y_evaluate):
    ''' 1.model: a Keras model object that has been defined and compiled.
        2.X_evaluate: a 3-dimensional numpy array representing the input data for evaluation.
        3.y_evaluate: a 2-dimensional numpy array representing the output data for evaluation.

        The function evaluates the performance of the model on the given input X_evaluate and output y_evaluate using the evaluate() method of the model object.

        The evaluate() method returns the model's performance metric as specified during model compilation.
        Here, the verbose argument is set to 2, which means that progress updates will be printed to the console during evaluation.

        The function then returns the accuracy metric as computed by the evaluate() method.'''
    accuracy = model.evaluate(X_evaluate, y_evaluate, verbose=2)
    return accuracy

def predict(model, X_input):
    ''' 1.X_input: a 3-dimensional numpy array representing the input data to be predicted.
        The function uses the predict() method of the model object to generate predictions for the input data X_input.
        '''
    start_time = time.time()
    yhat = model.predict(X_input)
    end_time = time.time()
    predict_execution_time=end_time - start_time
    print("Χρόνος προβλεψης: %s δευτερόλεπτα" % (predict_execution_time))
    return yhat

def plot_history(history):
    ''' 1.history: a Keras History object representing the training and validation metrics of a trained model.

        The function uses the plot() function of the Matplotlib library to create a figure with two subplots.
        The function uses the plot() function of the Matplotlib library to create a figure with two subplots.

        In the first subplot, the function plots the training and validation loss values as a function of the number of epochs using the history.history['loss'] and history.history['val_loss'] attributes of the history object.
        The function also adds a legend to the plot indicating which line represents which value, and labels the x-axis as "epochs" and the y-axis as "loss".

        In the second subplot, the function plots the training and validation MSE and MAE values as a function of the number of epochs using the history.history['mse'] and history.history['mae'] attributes of the history object.
        The function also adds a legend to the plot indicating which line represents which value, and labels the x-axis as "epochs" and the y-axis as "metrics".
        '''

    # plot loss
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.legend(['loss(mse)','val_loss(mse)'])
    plt.xlabel("epochs")
    plt.ylabel("loss")

    # plot metrics
    plt.subplot(2, 1, 2)
    plt.plot(history.history['mse'])
    plt.plot(history.history['mae'])
    plt.legend(['mse','mae'])
    plt.xlabel("epoch")
    plt.ylabel("metrics")

    plt.show()

def main():

    ''' 1.Loads data from a file specified by the filepath variable, with the target column specified by the target_col variable.
        2.Prepares the data by calling the prepare_data() function and splitting the data into training, validation, and testing sets using the train_test_valid_split_data() function.
        3.Builds a Keras model using the build_model4() function defined in a models module.
        4.Trains the model by calling the train_model() function.
        5,Saves the trained model and its weights to disk using the save_model() function.
        6.Predicts the output values for a single input sequence from the validation set using the predict() function.
        7.Evaluates the model performance on the testing and validation sets using the evaluate_model() function.
        8.Plots the training and validation loss and metrics over the number of epochs using the plot_history() function.
        9.Returns the predicted output values and the evaluation metrics for the testing and validation sets.'''
    # LOAD DATA
    values = load_data(filepath, target_col)

    #PREPARE DATA
    X,y = prepare_data(values,n_steps,n_outs)
    X_train, X_valid, X_test, y_train, y_valid, y_test = train_test_valid_split_data(X, y, test_size=test_size, valid_size=valid_size,
                                                                    shuffle=shuffle)
    #BUILD MODEL
    model = models.build_model4(n_steps,n_outs,n_features)

    #TRAIN MODEL
    batch_size = round(len(X) / 2)
    history = train_model(model, X_train, y_train, X_valid, y_valid, epochs, batch_size)

    #SAVE MODEL
    save_model(model, model_path, weights_path)

    # PREDICT
    X_input_valid = X_valid[:1]
    yhat_valid = predict(model, X_input_valid)
    print("Predicted sequence:")
    print(yhat_valid[0])

    #EVALUATION
    print("\nevaluate test")
    evaluate_test=evaluate_model(model, X_test, y_test)
    print("\nevaluate valid")
    evaluate_valid=evaluate_model(model, X_valid, y_valid)

    #PLOT
    plot_history(history)

    return yhat_valid,evaluate_test,evaluate_valid

if __name__ == '__main__':
    #PATH FILE
    filepath = r"C:\Users\JIM\Desktop\thesis code\rnd\2013-7\1new.csv"
    target_col = "CPU usage [%]"

    # prepare data
    n_steps = 30  # look back
    n_outs = 20   # look forward
    n_features=1

    # train model
    epochs = 40
    test_size = 0.2
    valid_size = 0.125
    shuffle = False

    #SAVE MODEL
    model_path = "model.json"
    weights_path = "model_weights.h5"

    #RUN MAIN
    main()

