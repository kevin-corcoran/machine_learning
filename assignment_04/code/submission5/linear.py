import numpy as np
from utils import *

class LinearClassifier:
    """ Boosting for binary classification.
    Please build an boosting model by yourself.

    Examples:
    The following example shows how your boosting classifier will be used for evaluation.
    >>> X_train, y_train = load_train_dataset() # we ignore the process of loading datset
    >>> X_test, y_test = load_test_dataset()
    >>> clf = LinearClassifier().fit(X_train, y_train)
    >>> y_pred =  clf.predict(X_test) # this is how you get your predictions
    >>> evaluation_score(y_pred, y_test) # this is how we get your final score for the problem.

    """
    def __init__(self):
        # initialize the parameters here
        pass
    

    def fit(self, X, y):
        """ Fit the boosting model.

        Parameters
        ----------
        X : { numpy.ndarray } of shape (n_samples, n_features)
            The input samples with dtype=np.float32.
        
        y : { numpy.ndarray } of shape (n_samples,)
            Target values. By default, the labels will be in {-1, +1}.

        Returns
        -------
        self : object
        """

        """ attempt 0 """

        # calculate discriminant
        (m,n) = X.shape
        
        # center data so we don't need to find t
        # X = X - np.mean(X, axis = 0)

        # least squares
        self.w = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T), y)

        """ attempt 1 """
        # # calculate discriminant
        # (m,n) = X.shape
        # 
        # # center data
        # X = X - np.mean(X, axis = 0)

        # # weights
        # # w = np.ones((m,))/m
        # # c = np.matmul(w,X)/sum(w)
        # c = np.mean(X, axis = 0)

        # y = y - np.matmul(X, c)

        # # least squares
        # self.w = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T), y)

        """ attempt 2 """

        # (m,n) = X.shape

        # # center data
        # X = X - np.mean(X, axis = 0)

        # index_class0 = np.where(y == 1)
        # X0 = X[index_class0]
        # y0 = y[index_class0] # labels 1

        # index_class1 = np.where(y == -1)
        # X1 = X[index_class1]
        # y1 = y[index_class1] # labels -1

        # centroid0 = np.mean(X0, axis = 0)
        # centroid1 = np.mean(X1, axis = 0)
        # c = (centroid1 - centroid0)/2

        # y = y - np.matmul(X, c)

        # self.w = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T), y)

        return self

    def predict(self, X):
        """ Predict binary class for X.

        Parameters
        ----------
        X : { numpy.ndarray } of shape (n_samples, n_features)

        Returns
        -------
        y_pred : { numpy.ndarray } of shape (n_samples)
                 In this sample submission file, we generate all ones predictions.
        """

        (m,_) = X.shape
        X = X - np.mean(X, axis = 0)

        prediction = np.zeros((m, 1), dtype=np.int16)
        yhat = np.matmul(X,self.w)

        # index of prediction
        # predicted class label0
        index0 = [i for i in range(m) if yhat[i] >= 0]
        class0 = X[index0]
        # predicted class label1
        index1 = [i for i in range(m) if yhat[i] < 0]
        class1 = X[index1]

        # update prediction
        prediction[index0] = 1
        prediction[index1] = -1

        return prediction

if __name__ == "__main__":
    X_train, y_train = load_train_dataset() # we ignore the process of loading datset
    X_test, y_test = load_test_dataset()

    # train_input_dir = 'data/training1.txt'
    # train_label_dir = 'data/training1_label.txt'
    # X_train = np.loadtxt(train_input_dir, skiprows=0)
    # y_train = np.loadtxt(train_label_dir, skiprows=0)

    clf = LinearClassifier().fit(X_train, y_train)
    y_pred = clf.predict(X_test)
