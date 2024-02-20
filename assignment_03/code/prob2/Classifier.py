import numpy as np
from collections import Counter

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

def euclidean_distance(X_train, X_test):
    """
    Create list of all euclidean distances between the given
    feature vector and all other feature vectors in the training set
    """
    return [np.linalg.norm(X - X_test) for X in X_train]

def k_nearest(X, Y, k):
    """
    Get the indices of the nearest feature vectors and return a
    list of their classes
    """
    idx = np.argpartition(X, k)
    return np.take(Y, idx[:k])

def predict(X_train, Y_train, X_test, Y_test, k):
    """
    For each feature vector get its predicted class
    """
    distances_list = [euclidean_distance(X_train, X) for X in X_test]
    distances = np.array(distances_list)
    breakpoint()
    prediction = np.array([Counter(k_nearest(distances, Y_train, k)).most_common()[0][0]])
    return prediction

def k_nearest_neighbors(X_train, Y_train, X_valid, K):
    (m, n) = X_train.shape
    (mv, nv) = X_valid.shape
    prediction = np.zeros((mv,))

    Y_train = Y_train.reshape(-1,1)
    distances = []
    for xv in X_valid:
        xv_xt_dist = np.array([np.linalg.norm(xv - xt) for xt in X_train]).reshape((m,1))
        # stack labels with distances
        main = np.hstack((xv_xt_dist, Y_train))
        # sort distances from lowest to highest
        distances.append(main[main[:,0].argsort()])

    for i, d in enumerate(distances):
        nn = np.zeros((K,))
        nn = d[:K,1] # labels of K nearest neighbors
        # count number of times labels occur
        neighbors, c = np.unique(nn, return_counts = True)
        # assign prediction to label that occurs most frequently 
        prediction[i] = neighbors[np.argmax(c)]

    # total_correct = sum([1 for (p,t) in zip(prediction,Y_valid) if p == t])
    return prediction



def KNN(x, train_label, y,k):
    x = x - np.mean(x, axis=0)
    y = y - np.mean(y, axis=0)
    dist = [] 
    #Computing Euclidean distance
    dist_ind = np.sqrt(np.sum((x-y)**2, axis=1)) 
    #Concatinating the label with the distance
    main_arr = np.column_stack((train_label,dist_ind))
    #Sorting the distance in ascending order
    main = main_arr[main_arr[:,1].argsort()] 
    #Calculating the frequency of the labels based on value of K
    count = Counter(main[0:k,0])
    keys, vals = list(count.keys()), list(count.values())
    breakpoint()
    if len(vals)>1:
        if vals[0]>vals[1]:
            return int(keys[0])
        else:
            return int(keys[1])
    else:
        return int(keys[0])

def kNN(X_train, Y_train, X_valid, Y_valid, X_test, K):
    # labs = np.unique(Y_train)
    # lab_indices = []
    (m,n) = X_train.shape
    # X = np.zeros((len(labs),n))
    # X = []
    # for i, l in enumerate(labs):
    #     lab_indices.append(np.where(Y_train == l))
    #     X.append((X_train[np.where(Y_train == l)], l))
    #     # X[i,:] = X_train[np.where(Y_train == l)]

    X_train = X_train - np.mean(X_train, axis = 0)
    X_valid = X_valid - np.mean(X_valid, axis = 0)

    total_correct = 0
    (mv, nv) = X_valid.shape
    correct_ = []
    c_k_ = []
    p_k_ = []

    for k in range(1,10):
        prediction = predict_neighbor(X_train, Y_train, X_valid, Y_valid, p=2, K = k)
        total_correct = sum([1 for (p,t) in zip(prediction,Y_valid) if p == t])
        correct_.append(total_correct)
    breakpoint()
    # prediction, K = find_k(X_train, Y_train, X_valid, Y_valid, p=1, K = 6)
    # total_correct = sum([1 for (p,t) in zip(prediction,Y_valid) if p == t])


    # # choose p
    # for p in [1, 2, 3, 4, np.inf]:
    #     # choose K
    #     prediction, K = find_k(X_train, Y_train, X_valid, Y_valid, p)

    #     total_correct = sum([1 for (p,t) in zip(prediction,Y_valid) if p == t])
    #     ## correct[K] = total_correct
    #     correct_.append(total_correct)
    #     c_k_.append(K)
    #     p_k.append(p)

    # find max p value
    # max_ = 0
    # count = 0
    # for k, tc in zip(c_k_, correct_):
    #     if tc > max_:
    #         K = k
    #         max_ = tc
    #     count += 1

    # p = p_k[count-1]
    # breakpoint()

            # for j, label in Y_train:

            # for (Di, label) in X_copy:
            #     l = np.argmin([np.linalg.norm(xv - di) for di in Di])
            #     di_removed = np.delete(Di, (l), axis = 0) # delete lth row
            #     breakpoint()
def predict_neighbor(X_train, Y_train, X_valid, Y_valid, p = 2, K = 1):
    (m, n) = X_train.shape
    (mv, nv) = X_valid.shape
    prediction = np.zeros((mv,))
    for i, xv in enumerate(X_valid):
        X_copy = X_train
        Y_copy = Y_train
        nn = np.zeros((K,)) # list of nearest neighbors labels for xv
        for k in range(K):
            index_t = np.argmin([np.linalg.norm(xv - xt, p) for xt in X_copy])
            nn[k] = Y_copy[index_t]
            X_copy = np.delete(X_copy, (index_t), axis = 0) # delete row of nn
            Y_copy = np.delete(Y_copy, (index_t), axis = 0) # delete row of nn

        neighbors, c = np.unique(nn, return_counts = True)
        prediction[i] = neighbors[np.argmax(c)]
    return prediction

def find_k(X_train, Y_train, X_valid, Y_valid, p = 2, K = 1):
    (m, n) = X_train.shape
    (mv, nv) = X_valid.shape
    prediction = np.zeros((mv,))
    pred = []
    correct = []
    c_k = []
    total_correct = 0
    while total_correct/nv <= 0.8 and K <= m:
        for i, xv in enumerate(X_valid):
            X_copy = X_train
            Y_copy = Y_train
            nn = np.zeros((K,)) # list of nearest neighbors labels for xv
            for k in range(K):
                index_t = np.argmin([np.linalg.norm(xv - xt, p) for xt in X_copy])
                nn[k] = Y_copy[index_t]
                X_copy = np.delete(X_copy, (index_t), axis = 0) # delete row of nn
                Y_copy = np.delete(Y_copy, (index_t), axis = 0) # delete row of nn

            u, c = np.unique(nn, return_counts = True)
            prediction[i] = u[np.argmax(c)]

        total_correct = sum([1 for (p,t) in zip(prediction,Y_valid) if p == t])
        # correct[K] = total_correct
        correct.append(total_correct)
        c_k.append(K)
        pred.append(prediction)
        K += 1
    
    breakpoint()
    # find max K value
    if K == m + 1: # Not enough training data
        max_ = 0
        for k, tc in zip(c_k, correct):
            if tc > max_:
                K = k
                max_ = tc
    else:
        breakpoint()

    return pred[K-1], K



if __name__ == "__main__":
    pass

