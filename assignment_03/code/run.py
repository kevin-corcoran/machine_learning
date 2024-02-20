import numpy as np
from Train import perceptron
from Classifier import classifier

'''
train_input_dir : the directory of training dataset txt file. For example 'training1.txt'.
train_label_dir :  the directory of training dataset label txt file. For example 'training1_label.txt'
test_input_dir : the directory of testing dataset label txt file. For example 'testing1.txt'
pred_file : output directory 
'''

def run (Xtrain_file,Ytrain_file,test_data_file,pred_file):
    # Reading data
    Xtrain_data = np.loadtxt(Xtrain_file, delimiter=",", skiprows=0)
    Ytrain_data = np.loadtxt(Ytrain_file, delimiter=",", skiprows=0)
    # split 70% training, 15% validation, 15% test if test_data_file = ''
    # np.random.seed(31)
    # np.random.shuffle(Xtrain_data)
    # np.random.shuffle(Ytrain_data)
    if not test_data_file:
        split = int(0.30 * Xtrain_data.shape[0])
        X_train = Xtrain_data[split:,:]
        Y_train = Ytrain_data[split:]

        X_test = Xtrain_data[:split,:]
        Y_test = Ytrain_data[:split]

        # split = int(0.15 * Xtrain_data.shape[0])
        # X_train = Xtrain_data[split*2:,:]
        # Y_train = Ytrain_data[split*2:]

        # X_valid = Xtrain_data[:split,:]
        # Y_valid = Ytrain_data[:split]

        # X_test = Xtrain_data[split:split*2,:]
        # Y_test = Ytrain_data[split:split*2]
    # else 80% training, 20% validation
    else:
        #split = int(0.20 * Xtrain_data.shape[0])
        #X_train = Xtrain_data[split:,:]
        #Y_train = Ytrain_data[split:]
        X_train = Xtrain_data
        Y_train = Ytrain_data

        #X_valid = Xtrain_data[:split,:]
        #Y_valid = Ytrain_data[:split]
        X_test = np.loadtxt(test_data_file, delimiter=",", skiprows=0)

    w, alph = perceptron(X_train, Y_train, 0.01, 100)
    prediction = classifier(X_test, w, alph)

    # total_correct = sum([1 for (p,t) in zip(prediction,Y_test) if p == t])

    # Saving you prediction to pred_file directory (Saving can't be changed)
    np.savetxt(pred_file, prediction, fmt='%1d', delimiter=",")

    
if __name__ == "__main__":
    Xtrain_file = './data/Xtrain.csv'
    Ytrain_file = './data/Ytrain.csv'
    test_data_file = ''
    pred_file = 'result'
    run(Xtrain_file,Ytrain_file,test_data_file,pred_file)

    # Testing
    # Xtrain_data = np.loadtxt(Xtrain_file, delimiter=',', skiprows=0)
    # Ytrain_data = np.loadtxt(Ytrain_file, delimiter=',', skiprows=0)

    # w, t = Train(Xtrain_file, Ytrain_file)
    # prediction = Classifier(test_data_file, w, t, pred_file)

    # test_label_file = 'data/testing1_label.txt'
    # test_labels = np.loadtxt(test_label_file, skiprows=0)

    # total_correct = sum([1 for (p,t) in zip(prediction,test_labels) if p == t])
