from sklearn.model_selection import KFold
from sklearn.svm.classes import SVC
from sklearn.metrics.classification import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict






X = X_sel
y = y_sel

cv = KFold(n_splits=5)

clf = GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
             learning_rate=0.01, loss='ls', max_depth=7, max_features=0.2,
             max_leaf_nodes=None, min_impurity_decrease=0.0,
             min_impurity_split=None, min_samples_leaf=20,
             min_samples_split=2, min_weight_fraction_leaf=0.0,
             n_estimators=550, presort='auto', random_state=None,
             subsample=1.0, verbose=0, warm_start=False)

ypred_all = np.zeros(shape=(len(y)))
i = 1
for train_index, test_index in cv.split(X):
    print("iteration", i, ":")
    print("train indices:", train_index)
    print("train data:", X[train_index])
    print("test indices:", test_index)
    print("test data:", X[test_index])
    clf.fit(X[train_index], y[train_index])
    ypred = clf.predict(X[test_index])
    print("predicted labels for data of indices", test_index, "are:", ypred)
    ypred_all[test_index] = ypred
    print("merged predicted labels:", ypred_all)
    i = i+1
    print("=====================================")
y_cross_val_predict = cross_val_predict(clf, X, cv=cv)
print("predicted labels by cross_val_predict:", y_cross_val_predict)


