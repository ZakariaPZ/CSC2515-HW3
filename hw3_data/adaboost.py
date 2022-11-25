from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from plot_roc import plot_roc

import data

train_data, train_labels, test_data, test_labels = data.load_all_data('data')

param_grid = {'n_estimators': [50, 100, 120, 150, 180], 
              'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3, 0.4]} 

optClf = GridSearchCV(AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=10)), param_grid)  

print('Grid search complete...')

optClf.fit(train_data, train_labels)

print(train_data.shape)

train_pred = optClf.predict(train_data)
test_pred = optClf.predict(test_data)
proba = optClf.decision_function(test_data)

print(optClf.best_params_)
print(optClf.best_estimator_)

print('Train accuracy: ', str(metrics.accuracy_score(y_pred=train_pred, y_true=train_labels)))
print('Test accuracy: ', str(metrics.accuracy_score(y_pred=test_pred, y_true=test_labels)))
print('Confusion matrix: ' + str(metrics.confusion_matrix(test_labels, test_pred)))
print('Precision: ' + str(metrics.precision_score(test_labels, test_pred, average='macro')))
print('Recall: ' + str(metrics.recall_score(test_labels, test_pred, average='macro')))
plot_roc(test_labels, proba, 'Adaboost')