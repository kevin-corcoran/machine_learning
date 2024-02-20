import numpy as np
# from Train import

def classifier(X_test, w, alph):
    (m,n) = X_test.shape
    X = X_test
    prediction = np.zeros((m,))

    # (k,) = alph.shape
    for i, x_i in enumerate(X):
        x_i = x_i.reshape(-1,1)
        # s = sum([alph[j]*np.sign(np.dot(x_i.T, w[j][:-1])) for j in range(k)])
        # yhat = np.sign(s)
        yhat = np.sign(np.dot(x_i.T, w))
        prediction[i] = yhat

    i0 = np.where(prediction == -1)
    prediction[i0] = 0

    return prediction
    # total_correct = sum([1 for (p,t) in zip(prediction,Y_test) if p == t])
    # print(total_correct)


if __name__ == "__main__":
    pass

