import numpy as np
# from sklearn.decomposition import PCA

def perceptron(X_train, Y_train, eta, epochs):

    # reduce feature space
    # pca = PCA(n_components = 0.95)
    # pca.fit(X_train)
    # X_train = pca.transform(X_train)

    # Split data
    index_class0 = np.where(Y_train == 0)
    # index_class1 = np.where(Y_train == 1)
    Y_train[index_class0] = -1

    # X0 = X_train[index_class0]
    # X1 = X_train[index_class1]

    # (m0,n0) = X0.shape
    # X0 = np.hstack((X0, np.ones((m0,1))))
    # y0 = np.ones((m0,1)) # label 0 classified as +1

    # (m1,n1) = X1.shape
    # X1 = np.hstack((X1, np.ones((m1,1))))
    # y1 = -1*np.ones((m1,1)) # label 1 classified as -1

    # X = np.vstack((X0, X1))
    # y = np.vstack((y0, y1))
    (m,n) = X_train.shape
    X = np.hstack((X_train, np.ones((m,1))))
    y = Y_train.reshape((m,1))

    # attempt 3
    k = 0; w_0 = np.zeros((n+1,)); alph = [0]; t = 0; T = epochs
    w = [w_0]
    converged = False
    while not converged and t <= T:
        converged = True
        for i in range(m):
            yhat = y[i,0]*np.sign(w[k] @ X[i,:])
            if yhat <= 0:
                # breakpoint()
                w.append(w[k] + eta * (y[i,0] * X[i,:]))
                # w = w + y[i,0] * X[i,:]
                alph.append(1)
                k += 1
                converged = False
            else:
                alph[k] += 1
        t += 1
    # t = w[-1]
    # w = w[:-1]
    # breakpoint()
    # return w, t
    return w, np.array(alph)

    # # attempt 2
    # alph = np.zeros((m,1)); converged = False; T = 5; t = 0
    # while not converged and t <= T:
    #     converged = True
    #     for i in range(m):
    #         if y[i,0]*sum([alph[j,0]*y[j,0]*X[i,:].T @ X[j,:] for j in range(m)]) <= 0:
    #             alph[i,0] += 1
    #             converged = False
    #     t += 1
    # 
    # ia = np.where(alph != 0)
    # alph = alph[ia]
    # y = y[ia]
    # X = X[ia[0],:]
    # (k,) = alph.shape
    # w = sum([alph[i]*y[i]*X[i,:] for i in range(k)])
    # t = w[-1]
    # w = w[:-1]
    # breakpoint()
    # return w, t


    # attempt 1
    # G = X.T @ X
    # k = 1; c_k = 0; w_k = np.zeros((n,1)); t = 0; T = 100; eta = 0.5
    # # wt0_1 = np.linalg.inv(X.T @ X) @ X.T @ y
    # while t < T:
    #     for i in range(m):
    #     # for x in X:
    #         breakpoint()
    #         if y[i,0] * (w_k.T @ X[i,:]) <= 0:
    #             w_k = w_k + eta*y[i,0]*X[i,:]
    #             c_k = 1
    #             k += 1
    #         else:
    #             c_k += 1
    #     t += 1


if __name__ == "__main__":
    train_input_dir = 'data/training1.txt'
    train_label_dir = 'data/training1_label.txt'
    w, t = Train(train_input_dir, train_label_dir)

    train_data = np.loadtxt(train_input_dir, skiprows=0)
    train_labels = np.loadtxt(train_label_dir, skiprows=0)
    index_class0 = np.where(train_labels == 0)
