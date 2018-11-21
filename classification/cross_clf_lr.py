import config
from classification import input_w
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

import os

estimator = LogisticRegression

path = config.HTTPS_CONFIG["result"]+"lr/"
if not os.path.exists(path): os.makedirs(path)

x_train, y_train, x_test, y_test = input_w.inputs()
# 参数范围
kw = {
    'C' : [1000000000],
    'penalty' : ['l1']
}
clf = GridSearchCV(
        estimator(),
        kw,
        scoring='f1_macro',
        cv=2,
        verbose=1,
        n_jobs=5)

from save_result import fit_and_save_result
fit_and_save_result(x_train, y_train, x_test, y_test, clf, path)