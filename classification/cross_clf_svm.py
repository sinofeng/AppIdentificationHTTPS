import config
from classification import input_w
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import os
estimator = SVC
# 参数范围
# kw = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
#                      'C': [1, 10, 100, 1000]},
#      {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

kw = [{'kernel': ['linear'], 'C': [1]}]

path =config.HTTPS_CONFIG["result"]+"svm/"
if not os.path.exists(path): os.makedirs(path)

x_train, y_train, x_test, y_test = input_w.inputs()

clf = GridSearchCV(
        estimator(probability=True),
        kw,
        scoring='f1_macro',
        cv=5,
        verbose=1,
        n_jobs=5)

from save_result import fit_and_save_result
fit_and_save_result(x_train, y_train, x_test, y_test, clf, path)