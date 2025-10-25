import numpy as np
from data_preprocessor import preprocess_mnist

def train_test_split(X, Y, train_size=.80, shuffle=True, random_state=None):
    """"
    Splits into train and test split
    Args:
        X: Input
        Y: Output
        train_size: (0,1) 
        shuffe: Shuffles partition
        random_state: seed for shuffling
    Returns:
        X_train, X_test, Y_train, Y_test
    """
    
    
    if random_state is not None:
        np.random.seed(random_state)

    n_samples = X.shape[0]
    indicies = np.arange(n_samples)

    if shuffle:
        np.random.shuffle(indicies)


    n_train = int(n_samples * train_size)

    train_indicies = indicies[:n_train]
    test_indicies = indicies[n_train:]


    X_train = X[train_indicies]
    X_test = X[test_indicies]
    Y_train = Y[train_indicies]
    Y_test = Y[test_indicies]

    return X_train, X_test, Y_train, Y_test


