# univariate lstm example
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM,Bidirectional,GRU
from keras.layers.convolutional import Conv1D
from keras.layers import Dense,TimeDistributed,RepeatVector
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.metrics import r2_score,f1_score
import pandas as pd
import numpy as np
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from fast_ml.model_development import train_valid_test_split
import matplotlib.pyplot as plt
from keras.layers import Dropout,MaxPooling1D,Flatten
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
import tensorflow as tf
from keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
    # find the end of this pattern
        end_ix = i + n_steps
    # check if we are beyond the sequence
        if end_ix > len(sequence) - 1:
            break
    # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


# define input sequence
'''
raw_seq = [10, 20, 30, 40, 50, 60, 70, 80, 90,100]
series = read_csv(r"C:\ Users\JIM\Desktop\ thesis code\ rnd\ 2013-7\ 1new.csv")

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(series.to_numpy())
data_scaled = pd.DataFrame(data_scaled,columns=['Timestamp [ms]', 'CPU cores', 'CPU capacity provisioned [MHZ]','CPU usage [MHZ]',
                                              'CPU usage [%]', 'Memory capacity provisioned [KB]', 'Memory usage [KB]',
                                              'Disk read throughput [KB/s]', 'Disk write throughput [KB/s]',
                                              'Network received throughput [KB/s]',
                                              'Network transmitted throughput [KB/s]'
                                              ])
values = series["CPU usage [%]"].values
'''
'''
series1 = read_csv(r"C:\ Users\JIM\Desktop\ thesis code\ rnd\ 2013-7\ 1new.csv")
values1 = series1["CPU usage [%]"].values

series2 = read_csv(r"C:\ Users\JIM\Desktop\ thesis code\ rnd\ 2013-7\ 270.csv")
values2 = series2["CPU usage [%]"].values
values=np.concatenate((values1, values2), axis=0)
'''
series = read_csv(r"C:\Users\JIM\Desktop\thesis code\rnd\2013-7\1new.csv")
#series = read_csv(r"C:\Users\JIM\Desktop\thesis code\cpu_mine.csv")
values = series["CPU usage [%]"].values
#print(values)
# choose a number of time steps
n_steps = 3
# split into samples
X, y = split_sequence(values, n_steps)

# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))
# define model
model = Sequential()

model.add(LSTM(200, activation='relu',return_sequences=True,input_shape=(n_steps,n_features) ,name="layer1")) # first values is neurones
model.add(LSTM(100, activation='relu',return_sequences=True,name="layer2"))
model.add(MaxPooling1D(pool_size=(1,)))

model.add(Bidirectional(LSTM(300, activation='relu',return_sequences=True)))

model.add(Bidirectional(GRU(500, activation='relu',return_sequences=True)))
model.add(Bidirectional(GRU(250, activation='relu')))

model.add(Dense(1))

cp1 = ModelCheckpoint('model1/', save_best_only=False)
model.compile(optimizer='adam', loss='mse',metrics=['accuracy','mse'])
print(model.summary())
# fit model
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.20,shuffle= False)
X_train,X_valid, y_train,y_valid  = train_test_split(X_train,y_train, test_size=0.125,shuffle= False)
print("X_train",X_train.shape)
print("y_train",y_train.shape)

print("X_test",X_test.shape)
print("y_test",y_test.shape)

print("X_val",X_valid.shape)
print("y_val",y_valid.shape)



EPOCHS=1500
BATCH_SIZE=len(X)
history=model.fit(X_train, y_train, epochs=EPOCHS,validation_data=(X_valid,y_valid),batch_size=BATCH_SIZE,callbacks=[cp1])

# demonstrate prediction
x_input = array([28.83333333,32.9,30.1])
x_input = x_input.reshape((1, n_steps, n_features))

yhat = model.predict(x_input, verbose=2)
print(yhat)
accuracy = model.evaluate(X_valid,y_valid, batch_size=BATCH_SIZE,verbose=2)
print("val_accuracy",accuracy[1]*100,"%","\n")
average_loss= sum(history.history['loss'])/len(history.history['loss'])
print("average_loss",average_loss)


test_predict = model.predict(X_test)

print('Test Mean Squared Error(MSE): %.2f; Test Mean Absolute Error(MAE) : %.2f '
      % (mean_squared_error(y_test, test_predict[:,0]), mean_absolute_error(y_test, test_predict[:,0])))
r2_value = r2_score(y_test,test_predict)
r2_value= r2_value.reshape(-1,1)
print("r2_score",r2_value)


#validation

valid_predict=model.predict(X_valid)
print('Validation:Test Mean Squared Error(MSE): %.2f; Test Mean Absolute Error(MAE) : %.2f '
      % (mean_squared_error(y_valid, valid_predict[:,0]), mean_absolute_error(y_valid,valid_predict[:,0])))

accuracy = model.evaluate(X_valid,y_valid, batch_size=BATCH_SIZE,verbose=2)
print("val_accuracy",accuracy[1]*100,"%","\n")
average_val_loss= sum(history.history['val_loss'])/len(history.history['val_loss'])
print("average_val_loss",average_val_loss)
#R-square
r2_value_val = r2_score(y_valid,valid_predict)
r2_value_val= r2_value.reshape(-1,1)
print("\nr2_score_val",r2_value_val)


# TRAIN PREDICT
train_predictions=model.predict(X_train).flatten()
print(train_predictions)
train_results=pd.DataFrame(data={'Train predictions':train_predictions,'Actuals':y_train})
print("train_result",train_results)

#VALIDATION PREDICT
val_predictions = model.predict(X_valid).flatten()
val_results = pd.DataFrame(data={'Val predictions':val_predictions,'Actuals':y_valid})
print("val_results",val_results)

#TEST PREDICTION
test_predictions = model.predict(X_test).flatten()
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


#train 70% test 20% validation 10% data(train,validation,test)
# X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.20,shuffle= False)
# X_train,X_valid, y_train,y_valid  = train_test_split(X_train,y_train, test_size=0.125,shuffle= False)
# print("x_TRAIN\n",X_train,X_train.shape,"\n"), print(y_train,y_train.shape,"\n")
# print("X_test\n",X_test,X_test.shape,"\n"), print(y_test,y_test.shape,"\n")
# print("xVAL\n",X_valid,X_valid.shape,"\n"), print(y_valid,y_valid.shape,"\n")


#train 70% test 20% validation 10% data(train,test,validation)

# X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.30,shuffle= False)
# X_test,X_valid, y_test,y_valid  = train_test_split(X_test,y_test, test_size=0.333,shuffle= False)
# print("x_TRAIN\n",X_train,X_train.shape,"\n"), print(y_train,y_train.shape,"\n")
# print("X_test\n",X_test,X_test.shape,"\n"), print(y_test,y_test.shape,"\n")
# print("xVAL\n",X_valid,X_valid.shape,"\n"), print(y_valid,y_valid.shape,"\n")