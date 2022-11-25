from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import data
from plot_roc import plot_roc

train_data, train_labels, test_data, test_labels = data.load_all_data('data')

param_grid = {'C': [1, 3, 5, 7], 
              'degree' : [2, 3, 4],
              'gamma': [1, 0.1, 0.01],
              'kernel': ["linear", "poly", "rbf"]} 

optClf = GridSearchCV(SVC(), param_grid, refit = True)  
optClf.fit(train_data, train_labels)
y_pred = optClf.predict(test_data)

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
plot_roc(test_labels, proba, 'SVM')