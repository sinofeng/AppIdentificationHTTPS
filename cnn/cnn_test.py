import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Input,Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.models import Model
from keras.utils import np_utils
from keras import backend as K
from keras.optimizers import Adam
# 全局变量
batch_size = 128
nb_classes = 10
epochs = 12
# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 根据不同的backend定下不同的格式
if K.image_dim_ordering() == 'th':
    # 训练集reshape
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    # 测试集reshape
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# 转换为one_hot类型
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

#构建模型
deep_inp = Input(shape=input_shape, dtype='float32', name='deep')
d=Convolution2D(nb_filters, (kernel_size[0], kernel_size[1]),padding='same',input_shape=input_shape)(deep_inp)
d=Activation('relu')(d) #激活层
d=Convolution2D(nb_filters, (kernel_size[0], kernel_size[1]))(d) #卷积层2
d=Activation('relu')(d) #激活层
d=MaxPooling2D(pool_size=pool_size)(d) #池化层
d=Dropout(0.25)(d) #神经元随机失活
d=Flatten()(d) #拉成一维数据
d=Dense(128)(d) #全连接层1
d=Activation('relu')(d) #激活层
d=Dropout(0.5)(d) #随机失活
d=Dense(nb_classes)(d) #全连接层2
d=Activation('softmax')(d) #Softmax评分

fit_param = dict()
fit_param['logistic'] = ('sigmoid', 'binary_crossentropy', 'accuracy')
fit_param['regression'] = (None, 'mse', None)
fit_param['multiclass'] = ('softmax', 'categorical_crossentropy', 'accuracy')
activation, loss, metrics = fit_param['multiclass']
model = Model(deep_inp, d)
model.compile(optimizer='adadelta', loss=loss, metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=64, nb_epoch=10,verbose=1,validation_data=(X_test,Y_test))
results = model.evaluate(X_test, Y_test)

print('Test score:', results[0])
print('Test accuracy:', results[1])