from model import CNN
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

import pandas as pd
import numpy as np




params = {
    'epoch': 100,
    'batch_size': 32
}

models = CNN(**params)

train = pd.read_csv('input/payload_train.csv')
val = pd.read_csv('input/payload_val.csv')

y_train = train['label'].values
x_train = train.drop('label', axis=1).values / 16
x_train = np.reshape(x_train, [-1, 64, 64, 1])


y_val = val['label'].values
x_val = val.drop('label', axis=1).values / 16
x_val = np.reshape(x_val, [-1, 64, 64, 1])
models.fit(x_train, y_train, x_val, y_val)


