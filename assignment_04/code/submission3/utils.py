from numpy import load
from os.path import join

dataset_dir = './dataset1'
train_set = join(dataset_dir, 'train.npy')
test_set = join(dataset_dir, 'test.npy')

def load_train_dataset():
    # load dataset
    with open(train_set, 'rb') as f:
        X_train = load(f)
        y_train = load(f)

    return X_train, y_train

def load_test_dataset():
    with open(test_set, 'rb') as f:
        X_test = load(f)
        y_test = load(f)

    return X_test, y_test


