from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt 

def plot_roc(y_true, y_probs, clf):
    y_true = label_binarize(y_true, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    plt.figure()
    for i in range(10):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_probs[:, i])

        plt.plot(fpr, tpr, label='Class No. {}'.format(i))
        plt.plot([0, 1], [0, 1], 'k--')
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for {}'.format(clf))
        plt.legend(loc="lower right")
    
    plt.show()