from numpy import asarray
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
import math
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot
import plotly.express as px
from numpy import asarray
from sklearn.metrics import accuracy_score
from sklearn import tree


# transform a time series dataset into a supervised learning dataset
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    print("\n-------------------------series_to_supervised---------------------\n")
    n_vars = 1 if type(data) is list else data.shape[1]
    print("data", data, "\n")
    df = DataFrame(data)
    print("df\n", df)
    cols = list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
    print("\n\ncols\n", cols, "\n\n")

    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        print("n_out\n", df.shift(-i))
    # put it all together
    agg = concat(cols, axis=1)
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg.values


# split a univariate dataset into train/test sets
def train_test_split(data, n_test):
    print("\n-------------------------train_test_split---------------------\n")
    print("n_test", n_test, "\n")
    print("\ntrain",data[:n_test, :], "\n")
    print("\ntest",data[n_test, :], "\n")

    return data[:round(values.shape[0]*75/100), :], data[-round(values.shape[0]*25/100):, :]


# fit an random forest model and make a one step prediction
def random_forest_forecast(train, testX):
    print("\n-------------------------random_forest_forecast---------------------\n")
    # transform list into array
    train = asarray(train)
    print("\ntrain", train)
    # split into input and output columns
    trainX, trainy = train[:, :-1], train[:, -1]
    print("\ntrainy", train[-round(values.shape[0]*25/100):, :])
    print("\ntrainX", train[:, :-1])
    # fit model
    model = RandomForestRegressor(n_estimators=1000,verbose=2)

    model.fit(trainX, trainy)

    # make a one-step prediction
    yhat = model.predict([testX])
    score = model.score(trainX, trainy)
    print("Accuracy", score)

    return yhat[0]


# walk-forward validation for univariate data
def walk_forward_validation(data, n_test):
    print("\n------------------------- walk_forward_validation---------------------\n")
    predictions = list()
    # split dataset
    train, test = train_test_split(data, n_test)
    print("\ntrain1\n", train)
    print("\ntest1\n", test)

    # seed history with training dataset
    history = [x for x in train]


    # step over each time-step in the test set
    for i in range(len(test)):
        print("len", len(test))
        print("i",i)
        # split test row into input and output columns
        testX, testy = test[i, :-1], test[i, -1]
        print("\ntestX1", testX)
        print("\ntesty1", testy)
        # fit model on history and make a prediction
        yhat= random_forest_forecast(history, testX)
        # store forecast in list of predictions
        predictions.append(yhat)
        # add actual observation to history for the next loop
        history.append(test[i])
        # summarize progress
        print('>expected=%.12f, predicted=%.12f' % (testy, yhat))

    # estimate prediction error
    # Error Metrics
    error = mean_absolute_error(test[:, -1], predictions)
    error1 = mean_absolute_percentage_error(test[:, -1], predictions)
    error2 = mean_squared_error(test[:, -1], predictions)
    error3 = math.sqrt(error2)






    return error, test[:, -1], predictions, error1, error2, error3


# load the dataset
series = read_csv(r"C:\Users\JIM\Desktop\thesis code\cpu_mine.csv")
#r"C:\Users\JIM\Desktop\thesis code\rnd\2013-7\1.csv"
#series1=series.drop(['CPU capacity provisioned [MHZ]','CPU cores','Timestamp [ms]','CPU usage [MHZ]','Memory capacity provisioned [KB]', 'Memory usage [KB]','Disk read throughput [KB/s]',
 #                'Disk write throughput [KB/s]','Network received throughput [KB/s]',
  #               'Network transmitted throughput [KB/s]'], axis=1)
values = series.values

print("values", values, '\n\n')
print("values.shape", values.shape, '\n\n')
# transform the time series data into supervised learning
data = series_to_supervised(values, n_in=1)
print("\ndata\n", data, '\n\n')
print("data.shape", data.shape)

# evaluate
mae, y, yhat, mape, mse, rmse = walk_forward_validation(data, 1)

# Error Metrics
print('MAE: %.12f' % mae)
print('MAPE: %.12f' % mape)
print('MSE: %.12f' % mse)
print('RMSE: %.12f' % rmse)





# plot expected vs predicted
pyplot.figure()
pyplot.plot(y,marker = 'o', label='Expected')
pyplot.plot(yhat,marker = 'o', label='Predicted')

pyplot.legend()
pyplot.grid()


pyplot.show()

fig=px.scatter(x=y,y=yhat,labels={
                     "x":"expected","y":"predicted"})
fig.show()





