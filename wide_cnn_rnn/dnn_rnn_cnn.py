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
from keras.optimizers import Adam
import config
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from result import figures
import os
choose=config.HTTPS_CONFIG[config.HTTPS_CONFIG["choose"]]
names=os.listdir(choose)
alphabet=[names[i][:-4] for i in range(len(names))]

def onehot(x):
    return np.array(OneHotEncoder().fit_transform(x).todense())

dnn_x_name=['push_flag_ratio',
            'average_len',
            'average_payload_len',
            'pkt_count',
            'flow_average_inter_arrival_time',
            'kolmogorov',
            'shannon',
            'max_len',
            'min_len',
            'std_len',
            'len_cipher_suites',
            'avrg_tcp_window',
            'max_tcp_window',
            'min_tcp_window',
            'var_tcp_window',
            #'session_id_length',
            'avrg_ip_ttl',
            'max_ip_ttl',
            'min_ip_ttl']
cnn_x_name=["Seq_%d_y"%i for i in range(128)]
rnn_x_name=["Seq_%d_x"%i for i in range(128)]
filtered_train_data=pd.read_csv(config.HTTPS_CONFIG["all_train_path"])
filtered_val_data=pd.read_csv(config.HTTPS_CONFIG["all_val_path"])

y_train_dnn=filtered_train_data["label"]
X_train_dnn = filtered_train_data[dnn_x_name]
y_train_dnn=np.asarray(y_train_dnn).reshape(-1,1)
y_train_dnn=onehot(y_train_dnn)
X_train_dnn=np.asarray(X_train_dnn)
y_test_dnn=filtered_val_data["label"]
X_test_dnn = filtered_val_data[dnn_x_name]
y_test_dnn=np.asarray(y_test_dnn).reshape(-1,1)
y_test_dnn=onehot(y_test_dnn)
X_test_dnn=np.asarray(X_test_dnn)


y_train_cnn=filtered_train_data["label"]
X_train_cnn = filtered_train_data[cnn_x_name]
y_train_cnn=np.asarray(y_train_cnn).reshape(-1,1)
y_train_cnn=onehot(y_train_cnn)
X_train_cnn=np.asarray(X_train_cnn).reshape((-1,128,1))
y_test_cnn=filtered_val_data["label"]
X_test_cnn = filtered_val_data[cnn_x_name]
y_test_cnn=np.asarray(y_test_cnn).reshape(-1,1)
y_test_cnn=onehot(y_test_cnn)
X_test_cnn=np.asarray(X_test_cnn).reshape((-1,128,1))



y_train_rnn=filtered_train_data["label"]
X_train_rnn = filtered_train_data[rnn_x_name]
y_train_rnn=np.asarray(y_train_rnn).reshape(-1,1)
y_train_rnn=onehot(y_train_rnn)
X_train_rnn=np.asarray(X_train_rnn)
y_test_rnn=filtered_val_data["label"]
X_test_rnn = filtered_val_data[rnn_x_name]
y_test_rnn=np.asarray(y_test_rnn).reshape(-1,1)
y_test_rnn=onehot(y_test_rnn)
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
kernel_size=8

dnn_inp=Input(shape=(18,))
d=Dense(32,activation='relu',kernel_regularizer=l1_l2(l1=0.01,l2=0.01))(dnn_inp)
d=Dense(32,activation='relu',kernel_regularizer=l1_l2(l1=0.01,l2=0.01))(dnn_inp)

cnn_input_shape=(128,1)
cnn_inp = Input(shape=cnn_input_shape, dtype='float32', name='cnn')
# 两层卷积操作
c = Conv1D(nb_filters, kernel_size=kernel_size,padding='same',strides=1)(cnn_inp)
c = MaxPooling1D()(c)
c = Conv1D(nb_filters, kernel_size=kernel_size,padding='same',strides=1)(c)
c = Flatten()(c)
c = Dense(32, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(c)

rnn_inp= Input(shape=(128,))
r=Embedding(257,16,input_length=128)(rnn_inp)
r=SimpleRNN(128,return_sequences=True)(r)
r=SimpleRNN(128)(r)
r=Dense(32,activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(r)

dcr_inp = concatenate([d, c, r])
# 加入注意力机制
attention_probs=Dense(96,activation='softmax',name="attention_probs")(dcr_inp)
dcr_inp=Multiply()([dcr_inp,attention_probs])
# wide特征和deep特征拼接，wide特征直接和输出节点相连
dcr = Dense(32,activation='relu')(dcr_inp)
dcr_out = Dense(config.HTTPS_CONFIG["num_class"], activation='softmax', name='dnn_cnn_rnn')(dcr)

# 模型网络的入口和出口
dcr = Model(inputs=[dnn_inp,cnn_inp, rnn_inp], outputs=dcr_out)
dcr.compile(optimizer=Adam(lr=0.01),loss="categorical_crossentropy",metrics=["accuracy"])
# 以下输入数据进行wide and deep模型的训练
print(dcr.summary())

X_tr = [X_train_dnn, X_train_cnn, X_train_rnn]
Y_tr = y_train_cnn
# 测试集
X_te = [X_test_dnn, X_test_cnn, X_test_rnn]
Y_te = y_test_cnn
dcr.fit(X_tr, Y_tr, epochs=100, batch_size=128)

results = dcr.evaluate(X_te, Y_te)
print("\n", results)
results = dcr.evaluate(X_te, Y_te)
predicts= dcr.predict(X_te)
y_pre=[np.argmax(i) for i in predicts]
y_ture=[np.argmax(i) for i in Y_te]
print("\n", results)
figures.plot_confusion_matrix(y_ture, y_pre,alphabet, "./")