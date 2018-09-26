#!/usr/bin/env python
# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pandas as pd
import numpy as np
HTTPS_SIZE = 32

# Global constants describing the HTTPS data set.
NUM_CLASSES = 3
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 200
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 100


# Load trainning and eval data
def inputs():
    payload_train=pd.read_csv('./processed_data/payload_train.csv')
    payload_val=pd.read_csv('./processed_data/payload_val.csv')
    train_labels=np.asarray(payload_train.pop('label'),dtype=np.int32)
    train_data=np.asarray(payload_train/16,dtype=np.float32)
    eval_labels=np.asarray(payload_val.pop('label'),dtype=np.int32)
    eval_data=np.asarray(payload_val/16,dtype=np.float32)
    return train_data,train_labels,eval_data,eval_labels