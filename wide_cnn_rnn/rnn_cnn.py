from keras.layers import Input,concatenate
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
from keras.layers import Embedding
from keras.layers import SimpleRNN
from keras.optimizers import Adam
import config
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
def onehot(x):
    return np.array(OneHotEncoder().fit_transform(x).todense())

cnn_train_data=pd.read_csv(config.HTTPS_CONFIG["packet_length_train_path"])
cnn_val_data=pd.read_csv(config.HTTPS_CONFIG["packet_length_val_path"])
# print(train_data)

rnn_train_data=pd.read_csv(config.HTTPS_CONFIG["record_type_train_path"])
rnn_val_data=pd.read_csv(config.HTTPS_CONFIG["record_type_val_path"])


y_train_cnn=cnn_train_data.pop("label")
X_train_cnn = cnn_train_data.drop(["id"],axis=1)
y_train_cnn=np.asarray(y_train_cnn).reshape(-1,1)
y_train_cnn=onehot(y_train_cnn)
X_train_cnn=np.asarray(X_train_cnn).reshape((-1,128,1))

# print(X_train_rnn)

y_test_cnn=cnn_val_data.pop("label")
X_test_cnn = cnn_val_data.drop(["id"],axis=1)
y_test_cnn=np.asarray(y_test_cnn).reshape(-1,1)
y_test_cnn=onehot(y_test_cnn)
X_test_cnn=np.asarray(X_test_cnn).reshape((-1,128,1))



y_train_rnn=rnn_train_data.pop("label")
X_train_rnn = rnn_train_data.drop(["id"],axis=1)
y_train_rnn=np.asarray(y_train_rnn).reshape(-1,1)
y_train_rnn=onehot(y_train_rnn)
X_train_rnn=np.asarray(X_train_rnn)

# print(X_train_rnn)

y_test_rnn=rnn_val_data.pop("label")
X_test_rnn = rnn_val_data.drop(["id"],axis=1)
y_test_rnn=np.asarray(y_test_rnn).reshape(-1,1)
y_test_rnn=onehot(y_test_rnn)
X_test_rnn=np.asarray(X_test_rnn)

batch_size=128

nb_filters=32
kernel_size=3

cnn_input_shape=(128,1)
cnn_inp = Input(shape=cnn_input_shape, dtype='float32', name='cnn')
# 两层卷积操作
c = Conv1D(nb_filters, kernel_size=kernel_size,padding='same',strides=1)(cnn_inp)
c = MaxPooling1D()(c)
c = Flatten()(c)
c = Dense(32, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(c)

rnn_inp= Input(shape=(128,))
r=Embedding(257,16,input_length=128)(rnn_inp)
r=SimpleRNN(128)(r)
r=Dense(32,activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(r)

cr_inp = concatenate([c, r])
# wide特征和deep特征拼接，wide特征直接和输出节点相连
cr = Dense(32,activation='relu')(cr_inp)
cr_out = Dense(17, activation='softmax', name='cnn_rnn')(cr)

# 模型网络的入口和出口
cr = Model(inputs=[cnn_inp, rnn_inp], outputs=cr_out)
cr.compile(optimizer=Adam(lr=0.0001),loss="categorical_crossentropy",metrics=["accuracy"])
# 以下输入数据进行wide and deep模型的训练
print(cr.summary())

X_tr = [X_train_cnn, X_train_rnn]
Y_tr = y_train_cnn
# 测试集
X_te = [X_test_cnn, X_test_rnn]
Y_te = y_test_cnn
cr.fit(X_tr, Y_tr, epochs=100, batch_size=128)

results = cr.evaluate(X_te, Y_te)
print("\n", results)
