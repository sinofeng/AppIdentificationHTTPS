import config
from classification import input_w
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

import os

estimator = KNeighborsClassifier
# 参数范围
kw = {'n_neighbors':[1,2,3]}

path = config.HTTPS_CONFIG["result"]+"knn/"
if not os.path.exists(path): os.makedirs(path)

x_train, y_train, x_test, y_test = input_w.inputs()

clf = GridSearchCV(
        estimator(),
        kw,
        scoring='f1_macro',
        cv=5,
        verbose=1,
        n_jobs=5)

from save_result import fit_and_save_result
fit_and_save_result(x_train, y_train, x_test, y_test, clf, path)