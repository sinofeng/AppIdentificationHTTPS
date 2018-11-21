from keras.layers import Input,concatenate
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import BatchNormalization
from keras.layers import Dense
from keras.layers import Multiply
from keras.regularizers import l2, l1_l2
import pandas as pd
import numpy as np
from keras.models import Model
from keras.layers import Embedding
from keras.layers import SimpleRNN
from keras.layers import LSTM
from keras.optimizers import Adam
import config
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
<<<<<<< HEAD
import sys
log_file=open("../result/rnn_cnn.result",'w')
#将打印结果写入文件
sys.stdout=log_file
def onehot(x):
    return np.array(OneHotEncoder().fit_transform(x).todense())
=======
from result import figures
import os
from keras.utils.np_utils import to_categorical
seed = 8
np.random.seed(seed)
>>>>>>> a08da2413b3b42cbda333ab376ee520b238f004d

choose=config.HTTPS_CONFIG[config.HTTPS_CONFIG["choose"]]
names=os.listdir(choose)
alphabet=[names[i][:-4] for i in range(len(names))]

def onehot(x):
    return np.array(OneHotEncoder().fit_transform(x).todense())

cnn_x_name=["Seq_%d_y"%i for i in range(128)]
rnn_x_name=["Seq_%d_x"%i for i in range(128)]
filtered_train_data=pd.read_csv(config.HTTPS_CONFIG["all_train_path"])
filtered_val_data=pd.read_csv(config.HTTPS_CONFIG["all_val_path"])

<<<<<<< HEAD
y_train_cnn=cnn_train_data.pop("label")
X_train_cnn = cnn_train_data.drop(["id"],axis=1)/1400.0
y_train_cnn=np.asarray(y_train_cnn).reshape(-1,1)
y_train_cnn=onehot(y_train_cnn)
=======
y_train_cnn=filtered_train_data["label"]
X_train_cnn = filtered_train_data[cnn_x_name]
# y_train_cnn=np.asarray(y_train_cnn).reshape(-1,1)
# y_train_cnn=onehot(y_train_cnn)
y_train_cnn=to_categorical(y_train_cnn)
>>>>>>> a08da2413b3b42cbda333ab376ee520b238f004d
X_train_cnn=np.asarray(X_train_cnn).reshape((-1,128,1))

# print(X_train_rnn)

<<<<<<< HEAD
y_test_cnn=cnn_val_data.pop("label")
X_test_cnn = cnn_val_data.drop(["id"],axis=1)/1400.0
y_test_cnn=np.asarray(y_test_cnn).reshape(-1,1)
y_test_cnn=onehot(y_test_cnn)
=======
y_test_cnn=filtered_val_data["label"]
X_test_cnn = filtered_val_data[cnn_x_name]
# y_test_cnn=np.asarray(y_test_cnn).reshape(-1,1)
# y_test_cnn=onehot(y_test_cnn)
y_test_cnn=to_categorical(y_test_cnn)
>>>>>>> a08da2413b3b42cbda333ab376ee520b238f004d
X_test_cnn=np.asarray(X_test_cnn).reshape((-1,128,1))



y_train_rnn=filtered_train_data["label"]
X_train_rnn = filtered_train_data[rnn_x_name]
# y_train_rnn=np.asarray(y_train_rnn).reshape(-1,1)
# y_train_rnn=onehot(y_train_rnn)
y_train_rnn=to_categorical(y_train_rnn)
X_train_rnn=np.asarray(X_train_rnn)

# print(X_train_rnn)

y_test_rnn=filtered_val_data["label"]
X_test_rnn = filtered_val_data[rnn_x_name]
# y_test_rnn=np.asarray(y_test_rnn).reshape(-1,1)
# y_test_rnn=onehot(y_test_rnn)
y_test_rnn=to_categorical(y_test_rnn)
X_test_rnn=np.asarray(X_test_rnn)

#
# cnn_train_data=pd.read_csv(config.HTTPS_CONFIG["packet_length_train_path"])
# cnn_val_data=pd.read_csv(config.HTTPS_CONFIG["packet_length_val_path"])
# # print(train_data)
#
# rnn_train_data=pd.read_csv(config.HTTPS_CONFIG["record_type_train_path"])
# rnn_val_data=pd.read_csv(config.HTTPS_CONFIG["record_type_val_path"])
#
#
# y_train_cnn=cnn_train_data.pop("label")
# X_train_cnn = cnn_train_data.drop(["id"],axis=1)
# y_train_cnn=np.asarray(y_train_cnn).reshape(-1,1)
# y_train_cnn=onehot(y_train_cnn)
# X_train_cnn=np.asarray(X_train_cnn).reshape((-1,128,1))
#
# # print(X_train_rnn)
#
# y_test_cnn=cnn_val_data.pop("label")
# X_test_cnn = cnn_val_data.drop(["id"],axis=1)
# y_test_cnn=np.asarray(y_test_cnn).reshape(-1,1)
# y_test_cnn=onehot(y_test_cnn)
# X_test_cnn=np.asarray(X_test_cnn).reshape((-1,128,1))
#
#
#
# y_train_rnn=rnn_train_data.pop("label")
# X_train_rnn = rnn_train_data.drop(["id"],axis=1)
# y_train_rnn=np.asarray(y_train_rnn).reshape(-1,1)
# y_train_rnn=onehot(y_train_rnn)
# X_train_rnn=np.asarray(X_train_rnn)
#
# # print(X_train_rnn)
#
# y_test_rnn=rnn_val_data.pop("label")
# X_test_rnn = rnn_val_data.drop(["id"],axis=1)
# y_test_rnn=np.asarray(y_test_rnn).reshape(-1,1)
# y_test_rnn=onehot(y_test_rnn)
# X_test_rnn=np.asarray(X_test_rnn)

batch_size=128

nb_filters=32
<<<<<<< HEAD
kernel_size=8
=======
kernel_size=5
>>>>>>> a08da2413b3b42cbda333ab376ee520b238f004d

cnn_input_shape=(128,1)
cnn_inp = Input(shape=cnn_input_shape, dtype='float32', name='cnn')
# 两层卷积操作
c = Conv1D(nb_filters, kernel_size=kernel_size,padding='same',strides=1)(cnn_inp)
c = MaxPooling1D()(c)
c = Conv1D(nb_filters, kernel_size=kernel_size,padding='same',strides=1)(c)
c = MaxPooling1D()(c)
c = Flatten()(c)

c = Dense(32, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(c)
c = Dense(32, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(c)

rnn_inp= Input(shape=(128,))
r=Embedding(257,16,input_length=128)(rnn_inp)
r=SimpleRNN(128,return_sequences=True)(r)
r=SimpleRNN(128)(r)
# r=LSTM(128,return_sequences=True)(r)
# r=LSTM(128)(r)
r=Dense(32,activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(r)
# r=Dense(32,activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(r)

cr_inp = concatenate([c, r])
# 加入注意力机制
<<<<<<< HEAD
attention_probs = Dense(64, activation='softmax', name='attention_probs')(cr_inp)
cr_inp=Multiply()([cr_inp, attention_probs])

=======
attention_probs=Dense(64,activation='softmax',name="attention_probs")(cr_inp)
cr_inp=Multiply()([cr_inp,attention_probs])
>>>>>>> a08da2413b3b42cbda333ab376ee520b238f004d
# wide特征和deep特征拼接，wide特征直接和输出节点相连
cr = Dense(256,activation='relu')(cr_inp)
cr = Dense(128,activation='relu')(cr)

cr_out = Dense(config.HTTPS_CONFIG["num_class"], activation='softmax', name='cnn_rnn')(cr)

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
<<<<<<< HEAD

print("\n", results)
log_file.close()
=======
predicts= cr.predict(X_te)
print(predicts)
y_pre=[np.argmax(i) for i in predicts]
y_ture=[np.argmax(i) for i in Y_te]
print("\n", results)
figures.plot_confusion_matrix(y_ture, y_pre,alphabet, "./")
>>>>>>> a08da2413b3b42cbda333ab376ee520b238f004d
