import numpy as np

def Discriminant(centroid0, centroid1):
    diff = centroid0 - centroid1
    t = np.sum(centroid0**2 - centroid1**2)/(2*diff[0,2])
    # normal to plane
    w = np.array([diff[0, 0]/diff[0, 2], diff[0, 1]/diff[0,2], 1])
    # z0_1(x,y) = t .- (w[1]*x + w[2]*y)
    return w, t

def Discriminant1(class0, class1):
    # Boundary 0-1

    X0 = class0
    (m0,n0) = X0.shape
    X0 = np.hstack((X0, np.ones((m0,1))))
    y0 = np.ones((m0,1))

    X1 = class1
    (m1,n1) = X1.shape
    X1 = np.hstack((X1, np.ones((m1,1))))
    y1 = -1*np.ones((m1,1))

    X = np.vstack((X0, X1))
    y = np.vstack((y0, y1))

    centroid0 = np.mean(X0, axis = 0)
    centroid1 = np.mean(X1, axis = 0)
    c = (centroid1 - centroid0)/2
    # c = np.mean(X, axis = 0)
    y = y.reshape((m0+m1,))
    y = y - np.matmul(X,c)

    # trained on 0 1 (least squares solution)
    # wt0_1 = np.linalg.inv(X.T @ X) @ X.T @ y
    wt0_1 = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T), y)

    w = wt0_1[:-1]
    t = wt0_1[-1]
    # yhat = X @ wt0_1
    # labs = [0 if i >= 0 else 1 for i in yhat]
    # print("i")
    # breakpoint()

    return w,t 
    

def Train(train_input_dir, train_label_dir):
    train_data = np.loadtxt(train_input_dir, skiprows=0)
    train_labels = np.loadtxt(train_label_dir, skiprows=0)
    
    train_data = train_data - np.mean(train_data, axis = 0)
    index_class0 = np.where(train_labels == 0)
    index_class1 = np.where(train_labels == 1)
    index_class2 = np.where(train_labels == 2)

    train_class0 = train_data[index_class0, :]
    train_class1 = train_data[index_class1, :]
    train_class2 = train_data[index_class2, :]
    class0 = train_class0[0,:,:]
    class1 = train_class1[0,:,:]
    class2 = train_class2[0,:,:]


    centroid0 = np.mean(class0, axis = 1)
    centroid1 = np.mean(class1, axis = 1)
    centroid2 = np.mean(class2, axis = 1)
    # class0 = class0 - centroid0
    # class1 = class1 - centroid1
    # class2 = class2 - centroid2

    # breakpoint()

    # w0_1_, t0_1_ = Discriminant(centroid0, centroid1)
    # w1_2_, t1_2_ = Discriminant(centroid1, centroid2)
    # w2_0_, t2_0_ = Discriminant(centroid2, centroid0)

    # t = np.array([t0_1_, t1_2_, t2_0_])
    # w = np.array([w0_1_, w1_2_, w2_0_])
    

    (m,n) = train_data.shape
    # class0_labels = train_labels[index_class0]
    # class1_labels = train_labels[index_class1]
    w0_1, t0_1 = Discriminant1(class0, class1) 
    w0_1 = w0_1.reshape((n,))

    # class1_labels = train_labels[index_class0]
    # class2_labels = train_labels[index_class1]
    w1_2, t1_2 = Discriminant1(class1, class2) 
    w1_2 = w1_2.reshape((n,))

    # class2_labels = train_labels[index_class0]
    # class0_labels = train_labels[index_class1]
    w2_0, t2_0 = Discriminant1(class2, class0)
    w2_0 = w2_0.reshape((n,))

    ts = np.array([t0_1, t1_2, t2_0])
    ws = np.array([w0_1, w1_2, w2_0])

    return ws, ts
    

if __name__ == "__main__":
    train_input_dir = 'data/training1.txt'
    train_label_dir = 'data/training1_label.txt'
    w, t = Train(train_input_dir, train_label_dir)

    train_data = np.loadtxt(train_input_dir, skiprows=0)
    train_labels = np.loadtxt(train_label_dir, skiprows=0)
    index_class0 = np.where(train_labels == 0)
