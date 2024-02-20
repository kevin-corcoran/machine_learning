import numpy as np
from Train import Train

def Classify(w, t, data, prediction, label0, label1, sgn):
    m0 = np.zeros((1,3))
    i0 = []
    m1 = np.zeros((1,3))
    i1 = []
    [num, _] = data.shape
    for i in range(num):
        margin = sgn*(np.dot(w,data[i,:]) - t)/np.linalg.norm(w,2)

        if margin <= 0:
            m0 = np.vstack((m0, data[i,:]))
            i0.append(i)
            prediction[i] = label0
        else:
            m1 = np.vstack((m1, data[i,:]))
            i1.append(i)
            prediction[i] = label1
    return (m0[1:,:], i0), (m1[1:,:], i1)

def Classify1(w, data, prediction, label0, label1):
    X = data
    w0 = w
    # yhat = X @ w0
    yhat = np.matmul(X,w0)
    (m,n) = yhat.shape
    # index of prediction
    # predicted class label0
    index0 = [i for i in range(m) if yhat[i] >= 0]
    class0 = X[index0]
    # predicted class label1
    index1 = [i for i in range(m) if yhat[i] < 0]
    class1 = X[index1]

    # update prediction
    prediction[index0] = label0
    prediction[index1] = label1

    return (class0, index0), (class1, index1)



def Classifier(test_input_dir, w, t, pred_file = "result"):
    test_data = np.loadtxt(test_input_dir, skiprows=0)
    [num, _] = test_data.shape
    prediction = np.zeros((num, 1), dtype=np.int16)

    X = test_data
    (m,n) = X.shape
    X = np.hstack((X, np.ones((m,1))))

    # check boundary 0 1
    w0 = np.vstack((w[0].reshape((n,1)), t[0].reshape((1,1))))
    w1 = np.vstack((w[1].reshape((n,1)), t[1].reshape((1,1))))
    w2 = np.vstack((w[2].reshape((n,1)), t[2].reshape((1,1))))
    (class0, indices0), (class1, indices1) = Classify1(w0, X, prediction, 0, 1)
    # (class0, indices0), (class1, indices1) = Classify(ws[0], ts[0], test_data, prediction, 0, 1, sgn)

    # find class 2 in class labeled 1
    pred_class1 = prediction[indices1]
    (temp1, tempi1), (class2, tempi2) = Classify1(w1, class1, pred_class1, 1, 2)
    # (temp1, tempi1), (class2, tempi2) = Classify(ws[1], ts[1], class1, pred_class1, 1, 2, sgn)
    # update prediction
    prediction[indices1] = pred_class1

    # # find class 2 in class labeled 0
    pred_class0 = prediction[indices0]
    (t2, t2i), (t0, t0i) = Classify1(w2, class0, pred_class0, 2, 0)
    # (t2, t2i), (t0, t0i) = Classify(ws[2], ts[2], class0, prediction, 2, 0, sgn)

    # # update prediction
    prediction[indices0] = pred_class0

    return prediction

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


