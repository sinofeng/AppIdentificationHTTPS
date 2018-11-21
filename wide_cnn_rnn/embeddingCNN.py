import pandas as pd
import numpy as np
from keras.layers import SimpleRNN
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Activation
train_data=pd.read_csv("./train.csv",names=['c1','c2','c3','c4','c5','label'])
val_data=pd.read_csv("./val.csv",names=['c1','c2','c3','c4','c5','label'])
# print(train_data)


y_train_rnn=train_data.pop("label")
X_train_rnn = train_data
y_train_rnn=np.asarray(y_train_rnn)
X_train_rnn=np.asarray(X_train_rnn)
print(X_train_rnn)
print(y_train_rnn)


y_test_rnn=val_data.pop("label")
X_test_rnn = val_data
y_test_rnn=np.asarray(y_test_rnn)
X_test_rnn=np.asarray(X_test_rnn)


# print(y_train_rnn)
# print(X_train_rnn)
EMBEDDING_SIZE = 5
HIDDEN_LAYER_SIZE = 5
model = Sequential()
model.add(Embedding(6, EMBEDDING_SIZE,input_length=5))
#model.add(LSTM(HIDDEN_LAYER_SIZE, dropout=0.2, recurrent_dropout=0.2))
model.add(SimpleRNN(HIDDEN_LAYER_SIZE))
model.add(Dense(1))
model.add(Activation("sigmoid"))
model.compile(loss="binary_crossentropy", optimizer="adam",metrics=["accuracy"])


BATCH_SIZE = 32
NUM_EPOCHS = 100
model.fit(X_train_rnn, y_train_rnn,epochs=NUM_EPOCHS,validation_data=(X_test_rnn, y_test_rnn))

