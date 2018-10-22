import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
param = {
    'ngram_range' : (1, 1),
    'decode_error' : 'ignore',
    'token_pattern' : r'\b\w+\b',
    'analyzer' : 'char',
}
c = CountVectorizer(**param)
characters="abcdefghijklmnopqrstuvwxyz0123456789"
c.fit([characters])

output_train=pd.read_csv('../data/train.csv')
output_val=pd.read_csv('../data/val.csv')
train_labels=np.asarray(output_train.pop('label'),dtype=np.int32)
eval_labels=np.asarray(output_val.pop('label'),dtype=np.int32)

train_data=np.asarray(output_train["extension_servername_indication"])
train_data=c.transform(train_data)
eval_data=np.asarray(output_val["extension_servername_indication"])
eval_data=c.transform(eval_data)
