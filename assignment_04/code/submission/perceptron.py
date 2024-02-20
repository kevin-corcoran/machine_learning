import numpy as np
from utils import *

class PerceptronClassifier:
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

    def fit(self, X_train, Y_train):
        eta = 0.01
        epochs = 12

        # # Split data
        # index_class0 = np.where(Y_train == 0)
        # Y_train[index_class0] = -1

        (m,n) = X_train.shape
        # X = np.hstack((X_train, np.ones((m,1))))
        X = X_train - np.mean(X_train, axis = 0)

        y = Y_train.reshape((m,1))

        # attempt 3: update w
        k = 0; w_k = np.zeros((n,)); alph = [0]; t = 0; T = epochs
        w_k = w_k.reshape(-1,1)
        # w = [w_k]
        converged = False
        Xy = list(zip(X,y))
        while not converged and t <= T:
            converged = True
            for (x_i,y_i) in Xy:
                x_i = x_i.reshape(-1,1)
                y_i = y_i.reshape(-1,1)
                # yhat = y[i,0]*np.sign(np.dot(x_i.T, w[k]))
                yhat = y_i*np.sign(np.dot(x_i.T, w_k))
                if yhat <= 0:
                    w_k = w_k + eta * (y_i * x_i)
                    # w.append(w[k] + eta * (y[i,0] * x_i))
                    # alph.append(1)
                    # k += 1
                    converged = False
                else:
                    pass
                    # alph[k] += 1
            t += 1

        self.w = w_k

        return self

        # t = w_k[-1]
        # w = w_k[:-1]
        # return w, t
    


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

