from keras.layers import Input
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import BatchNormalization
from keras.layers import Dense
from keras.regularizers import l2, l1_l2
import pandas as pd
import numpy as np
from keras.models import Model

train_data=pd.read_csv("./train.csv",names=['c1','c2','c3','c4','c5','label'])
val_data=pd.read_csv("./val.csv",names=['c1','c2','c3','c4','c5','label'])
# print(train_data)


y_train_rnn=train_data.pop("label")
X_train_rnn = train_data
y_train_rnn=np.asarray(y_train_rnn)
X_train_rnn=np.asarray(X_train_rnn).reshape((-1,5,1))

# print(X_train_rnn)

y_test_rnn=val_data.pop("label")
X_test_rnn = val_data
y_test_rnn=np.asarray(y_test_rnn)
X_test_rnn=np.asarray(X_test_rnn).reshape((-1,5,1))


batch_size=128

# deep模型卷积参数
nb_filters=32
kernel_size=3
# CNN处理数值特征
# 特征个数为5,通道个数为1
cnn_input_shape=(5,1)


cnn_inp = Input(shape=cnn_input_shape, dtype='float32', name='cnn')
# 两层卷积操作
c = Conv1D(nb_filters, kernel_size=kernel_size,padding='same',strides=1)(cnn_inp)
print(c)
c = MaxPooling1D()(c)
c = Flatten()(c)
# 1×100维
c = Dense(100, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(c)
c=Dense(1)(c)
c=Activation("sigmoid")(c)

C=Model(inputs=cnn_inp,outputs=c)

C.compile(loss="binary_crossentropy", optimizer="adam",metrics=["accuracy"])

print(C.summary())

C.fit(X_train_rnn, y_train_rnn,epochs=100,validation_data=(X_test_rnn, y_test_rnn))