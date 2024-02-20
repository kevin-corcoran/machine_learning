import numpy as np
from utils import *
from linear import LinearClassifier as Classifier

class BoostingClassifier:
    """ Boosting for binary classification.
    Please build an boosting model by yourself.

    Examples:
    The following example shows how your boosting classifier will be used for evaluation.
    >>> X_train, y_train = load_train_dataset() # we ignore the process of loading datset
    >>> X_test, y_test = load_test_dataset()
    >>> clf = BoostingClassifier().fit(X_train, y_train)
    >>> y_pred =  clf.predict(X_test) # this is how you get your predictions
    >>> evaluation_score(y_pred, y_test) # this is how we get your final score for the problem.

    """
    def __init__(self):
        # initialize the parameters here
        self.alg = Classifier
    
    def boosting(self, X, y, T):

        # # split data into training and validation
        # np.random.seed(31)
        # np.random.shuffle(X)
        # np.random.shuffle(y)

        # split = int(0.30 * X.shape[0])
        # # validation data
        # X_test = X[:split,:]
        # y_test = y[:split]

        # # training data
        # X = X[split:,:]
        # y = y[split:]

        # weights
        (m,_) = X.shape
        w = np.ones((m,))/m

        # confidence
        alpha = np.zeros((T,))
        # M = np.zeros((T,))
        M = []

        for t in range(T):
            c = np.matmul(w,X)/sum(w)
            y_i = y - np.matmul(X, c)

            clf = Classifier().fit(X, y_i)
            y_pred = clf.predict(X)

            total_correct = sum([1 for (p,t) in zip(y_pred,y) if p == t])
            acc = total_correct/m
            error = 1 - acc

            if error >= 1/2:
                T = t - 1
                break

            for i, (p,t) in enumerate(zip(y_pred,y)):
            # for i in range(m):
                if p == t:
                    # total_correct += 1
                    w[i] = w[i]/(2*error)
                else:
                    w[i] = w[i]/(2*(1-error))

            alpha[t] = 1/2*np.log(1-error)/error
            M.append(y_pred)
            # M.append(lambda X: clf.predict(X))
            # M[t] = lambda X: clf.predict

        # ensemble of binary classifiers
        M = sum([alpha[t]*M[t] for t in range(T)])
        # M = lambda X: sum([alpha[t]*M[t](X) for t in range(T)])
        return w, M
        # return w


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
        # calculate discriminant
        
        # center data
        X = X - np.mean(X, axis = 0)

        # hyper parameter 
        T = 10
        # weights
        w, self.M = self.boosting(X, y, T)
        # c = np.matmul(w,X)/sum(w)
        # y = y - np.matmul(X, c)

        # # least squares
        # self.w = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T), y)
        # self.w = wt[:-1]
        # self.t = wt[-1]

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
        prediction = self.M
        index0 = [i for i in range(m) if self.M[i] < 0]
        class0 = X[index0]
        # predicted class label1
        index1 = [i for i in range(m) if self.M[i] >= 0]
        class1 = X[index1]

        # update prediction
        prediction[index0] = 1
        prediction[index1] = -1
        return prediction

        # (m,_) = X.shape
        # prediction = np.zeros((m, 1), dtype=np.int16)
        # yhat = np.matmul(X,self.w)

        # # index of prediction
        # # predicted class label0
        # index0 = [i for i in range(m) if yhat[i] >= 0]
        # class0 = X[index0]
        # # predicted class label1
        # index1 = [i for i in range(m) if yhat[i] < 0]
        # class1 = X[index1]

        # # update prediction
        # prediction[index0] = 1
        # prediction[index1] = -1

        # return prediction

if __name__ == "__main__":
    X_train, y_train = load_train_dataset() # we ignore the process of loading datset
    X_test, y_test = load_test_dataset()
