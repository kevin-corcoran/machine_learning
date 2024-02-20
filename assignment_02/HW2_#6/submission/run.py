import numpy as np
from Train import Train
from Classifier import Classifier

'''
train_input_dir : the directory of training dataset txt file. For example 'training1.txt'.
train_label_dir :  the directory of training dataset label txt file. For example 'training1_label.txt'
test_input_dir : the directory of testing dataset label txt file. For example 'testing1.txt'
pred_file : output directory 
'''
# from julia.api import Julia
# jpath = "/home/computer/julia-1.6.3/bin/julia" # path to Julia, from current directory (your path may be slightly different)
# jl = Julia(runtime=jpath, compiled_modules=False) # compiled_modules=True may work for you; it didn't for me

# # Import Julia Modules
# from julia import Main
# Main.include("trainclassify.jl")

def run (train_input_dir,train_label_dir,test_input_dir,pred_file):
    # Reading data

    # (w, t) = Main.Train(train_input_dir, train_label_dir)
    # prediction = Main.Classifier(test_input_dir, w, t)
    w, t = Train(train_input_dir, train_label_dir)
    prediction = Classifier(test_input_dir, w, t, pred_file)

    # Saving you prediction to pred_file directory (Saving can't be changed)
    np.savetxt(pred_file, prediction, fmt='%1d', delimiter=",")

    
if __name__ == "__main__":
    train_input_dir = 'data/training1.txt'
    train_label_dir = 'data/training1_label.txt'
    test_input_dir = 'data/testing1.txt'
    pred_file = 'result'
    # run(train_input_dir,train_label_dir,test_input_dir,pred_file)

    test_label_dir = 'data/testing1_label.txt'
    w, t = Train(train_input_dir, train_label_dir)
    prediction = Classifier(test_input_dir, w, t, pred_file)
    test_labels = np.loadtxt(test_label_dir, skiprows=0)

    total_correct = sum([1 for (p,t) in zip(prediction,test_labels) if p == t])
