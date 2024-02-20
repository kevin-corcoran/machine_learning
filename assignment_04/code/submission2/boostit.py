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
        x_mean = np.mean(X, axis = 0) 
        X = X - x_mean

        # hyper parameter 
        T = 8

        """ boosting alg add ensemble of classifiers to this model """

        # weights
        (m,_) = X.shape
        w = np.ones((m,))/m

        # confidence
        alpha = []

        # binary classifiers
        M = []

        for t in range(T):
            # class exemplar
            c = np.matmul(w,X)/sum(w)
            y_i = y - np.matmul(X, c)

            clf = Classifier().fit(X, y_i)
            y_pred = clf.predict(X)

            total_correct = sum([1 for (p,q) in zip(y_pred,y) if p == q])
            acc = total_correct/y.shape[0]
            error = 1 - acc

            if error >= 1/2:
                T = t - 1 
                break

            for i, (p,q) in enumerate(zip(y_pred,y)):
                if p == q: # correctly classified
                    w[i] = w[i]/(2*(1-error)) # decrease weight
                else:
                    w[i] = w[i]/(2*error) # increase weight

            # add classifier and confidence
            # M.append(y_pred)
            M.append(clf.predict)
            alpha.append(1/2*np.log(1-error)/error)

        # ensemble of binary classifiers
        # self.M = sum([alpha[t]*M[t] for t in range(T)])
        self.M = sum([alpha[t]*M[t](X) for t in range(T)])

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
        # prediction = self.M(X)

        index0 = np.where(prediction < 0)
        class0 = X[index0]

        index1 = np.where(prediction >= 0)
        class1 = X[index1]

        # update prediction
        prediction[index0] = 1
        prediction[index1] = -1
        return prediction


if __name__ == "__main__":
    X_train, y_train = load_train_dataset() # we ignore the process of loading datset
    X_test, y_test = load_test_dataset()
