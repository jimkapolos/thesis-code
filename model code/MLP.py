# Simple LSTM

from numpy import array
from keras.models import Sequential
from keras.layers import LSTM,Bidirectional,GRU
from keras.layers.convolutional import Conv1D
from keras.layers import Dense,TimeDistributed,RepeatVector,PReLU
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.metrics import r2_score,f1_score
from keras.callbacks import ModelCheckpoint
import pandas as pd
import numpy as np
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from fast_ml.model_development import train_valid_test_split
import matplotlib.pyplot as plt
from keras.layers import Dropout,MaxPooling1D,Flatten
from keras.optimizers import Adam
import tensorflow as tf
from keras.models import model_from_json
from keras.models import load_model
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import time

# split a univariate sequence into samples
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





# define input sequence
#raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90,100]

series = read_csv(r"C:\Users\JIM\Desktop\thesis code\rnd\2013-7\1new.csv")



values = series["CPU usage [%]"].values
#print(values)
# choose a number of time steps
n_steps = 3 # look back
n_outs = 1
# split into samples
X, y = split_sequence(values, n_steps,n_outs)
for i in range(len(X)):
 print(i,X[i], y[i])



# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))


# define model
model3 = Sequential()

model3.add(Bidirectional(LSTM(100,activation="relu", return_sequences=True), input_shape=(n_steps,n_features)))



model3.add(LSTM(250,activation="relu",return_sequences=True))

model3.add(LSTM(150,activation="relu"))



model3.add(Dense(n_outs))

opt = tf.keras.optimizers.Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False,
    weight_decay=None,
    clipnorm=None,
    clipvalue=None,
    global_clipnorm=None,
    use_ema=False,
    ema_momentum=0.99,
    ema_overwrite_frequency=None,
    jit_compile=True,
    name='Adam'
)

model3.compile(optimizer=Adam(learning_rate=0.001,beta_1=0.9,beta_2=0.999,epsilon=1e-07), loss='mse', metrics=['accuracy','mse'])#'binary_crossentropy'
print(model3.summary())

#split data(train ,test,val)
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.20,shuffle= False)
X_train,X_valid, y_train,y_valid  = train_test_split(X_train,y_train, test_size=0.125,shuffle= False)

EPOCHS=500
BATCH_SIZE=round(len(X)/2)

start_time = time.time()
history=model3.fit(X_train, y_train, epochs=EPOCHS,validation_data=(X_valid,y_valid),batch_size=BATCH_SIZE)
end_time = time.time()
print("Χρόνος εκτέλεσης: %s δευτερόλεπτα" % (end_time - start_time))

#SAVE MODEL
# serialize model to JSON
model_json = model3.to_json()
with open("model3.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model3.save_weights("model3.h5")
print("Saved model to disk")

# demonstrate prediction
x_input = array([28.83333333,32.9,30.1])#28.83333333,32.9,30.1    12.86666667,13.63333333,13.43333333
x_input = x_input.reshape((1, n_steps, n_features))

# predicted
yhat= model3.predict(x_input)
print("predicted:",array(yhat))



# metrics

average_loss= sum(history.history['loss'])/len(history.history['loss'])
print("average_loss",average_loss)

average_val_loss= sum(history.history['val_loss'])/len(history.history['val_loss'])
print("average_val_loss",average_val_loss)



# make predictions on the test set

accuracy_test = model3.evaluate(X_test,y_test, batch_size=BATCH_SIZE,verbose=2)
print(accuracy_test)
test_predict = model3.predict(X_test)
print('\nTest Mean Squared Error(MSE): %.2f; Test Mean Absolute Error(MAE) : %.2f '
      % (mean_squared_error(y_test, test_predict[:,0]), mean_absolute_error(y_test, test_predict[:,0])))
# R-square
r2_value = r2_score(y_test,test_predict)
r2_value= r2_value.reshape(-1,1)
print("r2_score",r2_value)


#validation

valid_predict=model3.predict(X_valid)
print('\nValidation:Test Mean Squared Error(MSE): %.2f; Test Mean Absolute Error(MAE) : %.2f '
      % (mean_squared_error(y_valid, valid_predict[:,0]), mean_absolute_error(y_valid,valid_predict[:,0])))

accuracy_val = model3.evaluate(X_valid,y_valid, batch_size=BATCH_SIZE,verbose=2)
print(accuracy_val)

#R-square
r2_value_val = r2_score(y_valid,valid_predict)
r2_value_val= r2_value_val.reshape(-1,1)
print("r2_score_val",r2_value_val)


# TRAIN PREDICT
train_predictions=model3.predict(X_train).flatten()

train_predictions=np.ravel(train_predictions)
y_train = np.ravel(y_train)
train_results=pd.DataFrame({'Train predictions':train_predictions,'Actuals':y_train})
print("train_result",train_results)

#VALIDATION PREDICT
val_predictions = model3.predict(X_valid).flatten()
val_predictions = np.ravel(val_predictions)
y_valid = np.ravel(y_valid)
val_results = pd.DataFrame(data={'Val predictions':val_predictions,'Actuals':y_valid})
print("val_results",val_results)

#TEST PREDICTION
test_predictions = model3.predict(X_test).flatten()
test_predictions = np.ravel(test_predictions)
y_test = np.ravel(y_test)
test_results = pd.DataFrame(data={'Test predictions':test_predictions,'Actuals':y_test})
print("test_result",test_results)

#plot loss
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['loss','val_loss'])
plt.xlabel("epochs")
plt.ylabel("loss")

#plot accuracy
plt.subplot(2, 1, 2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['accuracy','val_accuracy'])
plt.xlabel("epoch")
plt.ylabel("accuracy")

#plot train
plt.figure()
plt.plot(train_results['Train predictions'])
plt.plot(train_results['Actuals'])
plt.title("TRAIN PREDICT")

#plot validation
plt.figure()
plt.plot(val_results['Val predictions'])
plt.plot(val_results['Actuals'])
plt.title("Val PREDICT")

#plot test

plt.figure()
plt.plot(test_results['Test predictions'])
plt.plot(test_results['Actuals'])
plt.title("Test PREDICT")

plt.show()

# y_pred = model.predict(X_test,verbose=2)
# y_pred = y_pred.reshape(-1,1)
# r2_value = r2_score(y_test, y_pred)
# print("r2_score",r2_value)
#
#
#
# # compare the predictions with the true values
# for i in range(len(y_pred)):
#     print("Predicted:", y_pred[i], "Expected:", y_test[i])



