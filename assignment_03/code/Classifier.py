import numpy as np
# from Train import

def classifier(X_test, w, alph):
    (m,n) = X_test.shape
    X = X_test
    prediction = np.zeros((n,))

    w = w[-1][:-1].reshape(-1,1)
    (k,) = alph.shape
    for i, x_i in enumerate(X):
        x_i = x_i.reshape(-1,1)
        yhat = np.sign(x_i.T @ w)
        prediction[i] = yhat

    i0 = np.where(prediction == -1)
    prediction[i0] = 0

    return prediction
    # total_correct = sum([1 for (p,t) in zip(prediction,Y_test) if p == t])
    # print(total_correct)


if __name__ == "__main__":
    test_input_dir = 'data/testing1.txt'
    test_label_dir = 'data/testing1_label.txt'
    pred_file = 'result'
    train_input_dir = 'data/training1.txt'
    train_label_dir = 'data/training1_label.txt'
    w, t = Train(train_input_dir, train_label_dir)
    prediction1 = Classifier(test_input_dir, w, t, pred_file)

    test_data = np.loadtxt(test_input_dir, skiprows=0)
    test_labels = np.loadtxt(test_label_dir, skiprows=0)
    [num, _] = test_data.shape
    prediction = np.zeros((num, 1), dtype=np.int16)

    test_data = np.loadtxt(test_input_dir, skiprows=0)


