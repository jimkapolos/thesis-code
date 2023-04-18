from keras.models import Sequential
from keras.layers import LSTM, Dense,MaxPooling1D,Bidirectional,GRU
from keras.optimizers import Adam
from keras.layers.convolutional import Conv1D

#SIMPLE LSTM
def build_model1(n_steps, n_outs,n_features):
    model1 = Sequential()
    model1.add(LSTM(200, activation='relu',return_sequences=True,input_shape=(n_steps, n_features)))
    model1.add(LSTM(100, activation='relu',return_sequences=True))
    model1.add(LSTM(50, activation='relu'))
    model1.add(Dense(n_outs))
    model1.compile(optimizer='adam', loss='mse', metrics=['accuracy','mse','mae'])
    print(model1.summary())
    return model1

#MODEL MINE
def build_model2(n_steps, n_outs,n_features):
    model2 = Sequential()
    model2.add(LSTM(200, activation='relu', return_sequences=True, input_shape=(n_steps, n_features),
                    name="layer1"))  # first values is neurones
    model2.add(LSTM(100, activation='relu', return_sequences=True, name="layer2"))
    model2.add(MaxPooling1D(pool_size=(1,)))
    model2.add(Bidirectional(LSTM(300, activation='relu', return_sequences=True)))
    model2.add(Bidirectional(GRU(500, activation='relu', return_sequences=True)))
    model2.add(Bidirectional(GRU(250, activation='relu')))
    model2.add(Dense(n_outs, activation='relu'))
    model2.compile(optimizer='adam', loss='mse', metrics=['accuracy', 'mse', 'mae'])
    print(model2.summary())
    return model2

#MODEL PAPER
def build_model3(n_steps, n_outs,n_features):
    model3 = Sequential()
    model3.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_steps, n_features),
                     strides=1))  # first values is neurones
    model3.add(MaxPooling1D(pool_size=(1,)))
    model3.add(Bidirectional(LSTM(64, activation='relu', return_sequences=True)))
    model3.add(Bidirectional(GRU(64, activation='relu', return_sequences=True)))
    model3.add(LSTM(64, activation='relu', return_sequences=True))
    model3.add(LSTM(64, activation='relu', return_sequences=True))
    model3.add(GRU(64, activation='relu', return_sequences=True))
    model3.add(LSTM(100, activation='relu'))
    model3.add(Dense(n_outs))
    model3.compile(optimizer=Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07), loss='mse',
                  metrics=['accuracy', 'mse', 'mae'])  # 'binary_crossentropy'
    print(model3.summary())
    return model3

#LSTM and Bidirectional
def build_model4(n_steps, n_outs,n_features):
    model4 = Sequential()
    model4.add(Bidirectional(LSTM(100, activation="relu", return_sequences=True), input_shape=(n_steps, n_features)))
    model4.add(LSTM(250, activation="relu", return_sequences=True))
    model4.add(LSTM(150, activation="relu"))
    model4.add(Dense(n_outs))
    model4.compile(optimizer=Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07), loss='mse',
                   metrics=['accuracy', 'mse', 'mae'])  # 'binary_crossentropy'
    print(model4.summary())
    return  model4
