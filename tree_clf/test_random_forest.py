from tree_clf.random_forest import ForestBuilder
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_curve


forest1 = ForestBuilder('000001.XSHE')
dict1 = forest1.get_eligible_rf('2012-01-01', '2017-01-01', train_precision_threshold=0.9, test_precision_threshold=0.65)

forest1_data = forest1.data('2013-01-01', '2019-02-13')

temp = DataFrame(forest1_data[0])
temp1 = DataFrame(forest1_data[1])

# param_grid = {"max_depth": [6],
#               "max_features": [None],
#               "min_samples_split": [0.01, 0.001, 0.03],
#               "bootstrap": [False],
#               "criterion": ["entropy"]}
#
# result = forest1.param_tunning(param_grid, forest1_data[0], forest1_data[1], cv=5)
# best_rf = result.best_estimator_
best_rf = RandomForestClassifier(bootstrap=False, class_weight=None,
                             criterion='entropy', max_depth=6, max_features=None,
                             max_leaf_nodes=None, min_impurity_decrease=0.0,
                             min_impurity_split=None, min_samples_leaf=1,
                             min_samples_split=0.01, min_weight_fraction_leaf=0.0,
                             n_estimators=500, n_jobs=-1, oob_score=False,
                             random_state=None, verbose=0, warm_start=False)
# clf = RandomForestClassifier(bootstrap=False, class_weight=None,
#                              criterion='entropy', max_depth=10, max_features=None,
#                              max_leaf_nodes=None, min_impurity_decrease=0.0,
#                              min_impurity_split=None, min_samples_leaf=1,
#                              min_samples_split=0.005, min_weight_fraction_leaf=0.0,
#                              n_estimators=500, n_jobs=-1, oob_score=False,
#                              random_state=None, verbose=0, warm_start=False)
X_train, X_test, y_train, y_test = forest1.split_scale(temp, temp1)
best_rf.fit(X_train, y_train)
train_prediction = best_rf.predict(X_train)
test_prediction = best_rf.predict(X_test)
accuracy_score(y_train, train_prediction)
accuracy_score(y_test, test_prediction)
dict(zip(temp.columns, best_rf.feature_importances_))

test_prob_pred = best_rf.predict_proba(X_test)
confusion_matrix(y_test, test_prediction)
test_threshold_pred = test_prob_pred[:, 1] > 0.9
confusion_matrix(y_test, test_threshold_pred)
# # split and scale X, y
# X_train, X_test, y_train, y_test = forest1.split_scale(temp, temp1)
# best_rf.fit(X_train, y_train)
# prediction = best_rf.predict(X_test)
# from sklearn.metrics import accuracy_score
# accuracy_score(y_test, prediction)
# accuracy_score(y_train, best_rf.predict(X_train))
