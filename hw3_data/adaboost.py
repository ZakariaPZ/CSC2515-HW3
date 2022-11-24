from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

import data

train_data, train_labels, test_data, test_labels = data.load_all_data('data')

param_grid = {'n_estimators': [50, 100, 120, 150, 180], 'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3, 0.4]} 

optClf = GridSearchCV(AdaBoostClassifier(), param_grid, refit = True, verbose=3)  

optClf.fit(train_data, train_labels)

train_pred = optClf.predict(train_data)
test_pred = optClf.predict(test_data)

print('Train accuracy: ', str(metrics.accuracy_score(y_pred=train_pred, y_true=train_labels)))
print('Test accuracy: ', str(metrics.accuracy_score(y_pred=test_pred, y_true=test_labels)))

print(optClf.best_params_)
print(optClf.best_estimator_)

# clf = AdaBoostClassifier(n_estimators=20, random_state=0, base_estimator=DecisionTreeClassifier(max_depth=1, random_state=0))
# train_pred = clf.predict(train_data)
# test_pred = clf.predict(test_data)

# print('Train accuracy: ', str(metrics.accuracy_score(y_pred=train_pred, y_true=train_labels)))
# print('Test accuracy: ', str(metrics.accuracy_score(y_pred=test_pred, y_true=test_labels)))