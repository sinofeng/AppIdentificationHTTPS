import config
from classification import input_w
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

from save_result import fit_and_save_result
import numpy as np
import os


n_gram = 3
estimator = RandomForestClassifier
# 参数范围
kw = {
    'n_estimators' : [290],
    'max_depth' : [90],
    'max_leaf_nodes' : [70],

}

path = config.HTTPS_CONFIG["result"]+"rf/"
if not os.path.exists(path): os.makedirs(path)

x_train, y_train, x_test, y_test = input_w.inputs()

clf = GridSearchCV(
        estimator(),
        kw,
        scoring='f1_macro',
        cv=5,
        verbose=1,
        n_jobs=5)


fit_and_save_result(x_train, y_train, x_test, y_test, clf, path)
