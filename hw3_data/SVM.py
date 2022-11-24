from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import data


def build_SVM(X_train, y_train, X_test, y_test, kernel='poly', degree=3):
    
    svm_clf = SVC(kernel=kernel, degree=degree)
    svm_clf.fit(X_train, y_train)

    y_pred = svm_clf.predict(X_test)

    print(metrics.accuracy_score(y_pred=y_pred, y_true=y_test))


    return svm_clf


train_data, train_labels, test_data, test_labels = data.load_all_data('data')
clf = build_SVM(train_data, train_labels, test_data, test_labels)

param_grid = {'C': [0.01, 0.1, 1, 10, 100], 
              'degree' : [2, 3, 4],
              'gamma': [1, 0.1, 0.01],
              'kernel': ["linear", "poly", "rbf"]} 

optClf = GridSearchCV(SVC(), param_grid, refit = True)  
optClf.fit(train_data, train_labels)
y_pred = optClf.predict(test_data)

print(metrics.accuracy_score(y_pred=y_pred, y_true=test_labels))

print(optClf.best_params_)
print(optClf.best_estimator_)