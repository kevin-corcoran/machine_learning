#!/usr/bin/env python3
import numpy as np
from Train import perceptron, dual_perceptron
from Classifier import classifier

'''
train_input_dir : the directory of training dataset txt file. For example 'training1.txt'.
train_label_dir :  the directory of training dataset label txt file. For example 'training1_label.txt'
test_input_dir : the directory of testing dataset label txt file. For example 'testing1.txt'
pred_file : output directory 
'''

def run (Xtrain_file,Ytrain_file,test_data_file,pred_file):
    # local tests
    # Reading data
    Xtrain_data = np.loadtxt(Xtrain_file, delimiter=",", skiprows=0)
    Ytrain_data = np.loadtxt(Ytrain_file, delimiter=",", skiprows=0)

    # split 70% training, 30% test if test_data_file = ''
    if not test_data_file:
        split = int(0.30 * Xtrain_data.shape[0])
        X_train = Xtrain_data[split:,:]
        Y_train = Ytrain_data[split:]

        X_test = Xtrain_data[:split,:]
        Y_test = Ytrain_data[:split]
    else:
        X_train = Xtrain_data
        Y_train = Ytrain_data
        X_test = np.loadtxt(test_data_file, delimiter=",", skiprows=0)

    # w, alph = dual_perceptron(X_train, Y_train, 0.01, 20)
    # prediction = classifier(X_test, w, alph)
    w, alph = perceptron(X_train, Y_train, 0.01, 20)
    prediction = classifier(X_test, w, alph)

    total_correct = sum([1 for (p,t) in zip(prediction,Y_test) if p == t])
    print(total_correct)
    breakpoint()

    # Saving you prediction to pred_file directory (Saving can't be changed)
    np.savetxt(pred_file, prediction, fmt='%1d', delimiter=",")

    
if __name__ == "__main__":
    Xtrain_file = './data/Xtrain.csv'
    Ytrain_file = './data/Ytrain.csv'
    test_data_file = ''
    pred_file = 'result'
    run(Xtrain_file,Ytrain_file,test_data_file,pred_file)
