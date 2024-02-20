import numpy as np
# from sklearn.decomposition import PCA

def perceptron(X_train, Y_train, eta, epochs):

    # Split data
    index_class0 = np.where(Y_train == 0)
    Y_train[index_class0] = -1

    (m,n) = X_train.shape
    X = np.hstack((X_train, np.ones((m,1))))
    y = Y_train.reshape((m,1))

    # attempt 3: update w
    k = 0; w_k = np.zeros((n+1,)); alph = [0]; t = 0; T = epochs
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
    t = w_k[-1]
    w = w_k[:-1]
    return w, t
    
    # return w, np.array(alph)

def dual_perceptron(X_train, Y_train, eta, epochs):
    index_class0 = np.where(Y_train == 0)
    Y_train[index_class0] = -1

    (m,n) = X_train.shape
    X = np.hstack((X_train, np.ones((m,1))))
    y = Y_train.reshape((m,1))

    # attempt 3: alpha weights
    # alph = (index, value)
    k = 0; w_k = np.zeros((n+1,)); alph = [[0, 0]]; t = 0; T = epochs
    w_k = w_k.reshape(-1,1)
    w = [w_k]
    converged = False
    Xy = list(zip(X,y))
    while not converged and t <= T:
        converged = True
        for i, (x_i,y_i) in enumerate(Xy):
            x_i = x_i.reshape(-1,1)
            y_i = y_i.reshape(-1,1)
            # w_k = sum([a[1]*y[a[0],0]*X[a[0],:] for a in alph])*eta
            # w_k = w_k.reshape(-1,1)
            yhat = y_i*np.sign(np.dot(x_i.T, w_k))
            # yhat = y_i*np.sign(np.dot(x_i.T, w[k]))
            if yhat <= 0:
                w_k = w_k + eta * (y_i * x_i)
                # w.append(w[k] + eta * (y_i * x_i))
                alph.append([i,1])
                k += 1
                converged = False
            else:
                alph[k][1] += 1
        t += 1
    # w_k = sum([a[1]*y[a[0],0]*X[a[0],:] for a in alph])
    return w_k[:-1], alph
    
    # return w, np.array(alph)

def k_means(X, y, K):
    (m,n) = X.shape
    klusters = []
    np.random.seed(2)

    klusters = np.random.rand(K,n)
    for i in range(K):
        for j in range(n):
            m_ = X[:,j].min()
            M_ = X[:,j].max()
            klusters[i,j] = (M_ - m_) * klusters[i,j] + m_

    # klusters = (M_ - m_) * np.random.rand(K, n) + m_

    labels = np.zeros((m,))

    while True:
        count_changes = 0
        # assign x to cluster and count changes
        for i, x in enumerate(X):
            l = np.argmin([np.linalg.norm(x-k) for k in klusters]) + 1
            # ith data point in X labels with index of cluster + 1
            if l != labels[i]:
                labels[i] = l
                count_changes += 1

        for i, k in enumerate(klusters):
            index_i = np.where(labels == i+1)
            X_i = X[index_i]
            klusters[i,:] = np.mean(X_i,axis=0)

        if count_changes == 0:
            break

# def kNN(X, y, K):




if __name__ == "__main__":
    pass
