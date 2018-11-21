import config
from classification import input_w
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier

estimator = GradientBoostingClassifier

x_train, y_train, x_test, y_test=input_w.inputs()
path=config.HTTPS_CONFIG["result"]+"gbdt/"
if not os.path.exists(path): os.makedirs(path)

# 参数范围
kw = {
    'max_depth':[5],
    'n_estimators':[500],
    'learning_rate':[0.1]
}
clf = GridSearchCV(
        estimator(),
        kw,
        scoring='f1_macro',
        cv=5,
        verbose=1,
        n_jobs=5)
from save_result import fit_and_save_result
fit_and_save_result(x_train, y_train, x_test, y_test, clf, path)