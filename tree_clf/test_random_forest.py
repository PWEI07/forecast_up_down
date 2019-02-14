from tree_clf.random_forest import ForestBuilder
from sklearn.ensemble import RandomForestClassifier

forest1 = ForestBuilder('000001.XSHE')
forest1_data = forest1.data('2013-01-01', '2019-02-13')

temp=forest1_data[0]
temp1=forest1_data[1]

param_grid = {"max_depth": [5, 7, 9, None],
              "max_features": [0.5, 'auto', None],
              "min_samples_split": [0.01, 0.025, 0.005],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

result = forest1.param_tunning(param_grid, forest1_data[0], forest1_data[1], n_iter=50)
best_rf = result.best_estimator_

# split and scale X, y
X_train, X_test, y_train, y_test = forest1.split_scale(temp, temp1)
best_rf.fit(X_train, y_train)
prediction = best_rf.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, prediction)
accuracy_score(y_train, best_rf.predict(X_train))
