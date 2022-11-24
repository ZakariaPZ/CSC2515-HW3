'''
Question 4.1 Skeleton Code

Here you should implement and evaluate the k-NN classifier.
'''

import data
import numpy as np
# Import pyplot - plt.imshow is useful!
import matplotlib.pyplot as plt

from collections import Counter
from sklearn.model_selection import KFold
from sklearn import metrics


class KNearestNeighbor(object):
    '''
    K Nearest Neighbor classifier
    '''

    def __init__(self, train_data, train_labels):
        self.train_data = train_data
        self.train_norm = (self.train_data**2).sum(axis=1).reshape(-1,1)
        self.train_labels = train_labels

    def l2_distance(self, test_point):
        '''
        Compute L2 distance between test point and each training point
        
        Input: test_point is a 1d numpy array
        Output: dist is a numpy array containing the distances between the test point and each training point
        '''
        # Process test point shape
        test_point = np.squeeze(test_point)
        if test_point.ndim == 1:
            test_point = test_point.reshape(1, -1)
        assert test_point.shape[1] == self.train_data.shape[1]

        # Compute squared distance
        test_norm = (test_point**2).sum(axis=1).reshape(1,-1)
        dist = self.train_norm + test_norm - 2*self.train_data.dot(test_point.transpose())
        return np.squeeze(dist)

    def query_knn(self, test_point, k):
        '''
        Query a single test point using the k-NN algorithm

        You should return the digit label provided by the algorithm
        '''

        inds = np.argsort(self.l2_distance(test_point))
        k_max_inds = inds[:k]
        digits = self.train_labels[k_max_inds]
        votes = Counter(digits).most_common(2)

        # votes = None
        # maxDig = 11
        # for element, digit in zip(list(Counter([3, 7, 3, 7, 2, 2, 4, 6, 6]).values()), list(Counter([3, 7, 3, 7, 2, 2, 4, 6, 6]).keys())):
        #     if element == max(list(Counter([3, 1, 3, 1, 2, 2, 4]).values())) and digit < maxDig:
        #         votes = element 
        #         maxDig = digit 

        # votes = np.argmax(np.bincount(digits.astype('int64')))

        if k > 1 and len(votes) > 1:

            # print(votes[0][1], votes[1][1])

            # if k becomes 1, len(votes) == 1 - no need to check for k == 1
            while (len(votes) > 1 and votes[0][1] == votes[1][1]):
                k -= 1
                k_max_inds = inds[:k]
                digits = self.train_labels[k_max_inds]
                votes = Counter(digits).most_common(2)

        return votes[0][0]
        # return votes

def cross_validation(train_data, train_labels, k_range=np.arange(1,16)):
    '''
    Perform 10-fold cross validation to find the best value for k

    Note: Previously this function took knn as an argument instead of train_data,train_labels.
    The intention was for students to take the training data from the knn object - this should be clearer
    from the new function signature.
    '''
    train_accs = []
    val_accs = []

    for k in k_range:
        # Loop over folds
        # Evaluate k-NN
        # ...
        train_k_accs = []
        val_k_accs = []

        KF_splits = KFold(n_splits=10)

        for train_inds, val_inds in KF_splits.split(train_data):
            X_train, X_val = train_data[train_inds], train_data[val_inds]
            y_train, y_val = train_labels[train_inds], train_labels[val_inds]
    
            knn = KNearestNeighbor(train_data=X_train, train_labels=y_train)

            train_k_accs.append(classification_accuracy(knn, k, X_train, y_train)[0])
            val_k_accs.append(classification_accuracy(knn, k, X_val, y_val)[0])
        
        train_accs.append(np.mean(train_k_accs))
        val_accs.append(np.mean(val_k_accs))

        print('Validation accuracy for k={}'.format(k), val_accs[-1])
    
    k = np.argmax(val_accs) + 1
    
    return k
        
def classification_accuracy(knn, k, eval_data, eval_labels):
    '''
    Evaluate the classification accuracy of knn on the given 'eval_data'
    using the labels
    '''
    hit_rate = 0
    N = eval_data.shape[0]

    y_pred = []
    # Loop through all test points
    for test_point, test_label in zip(eval_data, eval_labels):
        pred = knn.query_knn(test_point, k)

        y_pred.append(pred)

        # Accumulate loss
        if pred == test_label:
            hit_rate += 1
    
    y_pred = np.array(y_pred)
    acc = metrics.accuracy_score(y_pred=y_pred, y_true=eval_labels)

    # return hit_rate/N
    return acc, y_pred

def main():
    train_data, train_labels, test_data, test_labels = data.load_all_data('data')
    knn = KNearestNeighbor(train_data, train_labels)

    # # 4.1 a)
    # train_err_k1 = classification_accuracy(knn, 1, train_data, train_labels)[0]
    # test_err_k1 = classification_accuracy(knn, 1, test_data, test_labels)[0]

    # # 4.1 b)
    # train_err_k15 = classification_accuracy(knn, 15, train_data, train_labels)[0]
    # test_err_k15 = classification_accuracy(knn, 15, test_data, test_labels)[0]

    # print('K=1 - train accuracy: ' + str(train_err_k1) + ', test accuracy: ' + str(test_err_k1))
    # print('K=15 - train accuracy: ' + str(train_err_k15) + ', test accuracy: ' + str(test_err_k15))

    # 4.2 - Implemented 
    4.3 
    # optK = cross_validation(train_data, train_labels)
    optK = 3
    test_acc, preds = classification_accuracy(knn, optK, test_data, test_labels)

    print('The optimal k is {}'.format(optK) + '.')
    print('Test accuracy: ' + str(test_acc))
    print('Confusion matrix: ' + str(metrics.confusion_matrix(test_labels, preds)))
    print('Precision: ' + str(metrics.precision_score(test_labels, preds)))
    print('Recall: ' + str(metrics.recall_score(test_labels, preds)))
    

if __name__ == '__main__':
    main()