from load_mnist_data import load_mnist_data
from data_preprocessor import preprocess_mnist
from train_test_split import train_test_split
import numpy as np

def init_weights(n_features, n_classes, xavier_norm = True):
    if xavier_norm:
        xavier_norm = np.sqrt(6 / (n_features + n_classes))
        W = np.random.uniform(-xavier_norm, xavier_norm, (n_classes, n_features))
        b = np.zeros((n_classes,1))
    else:
        W = np.zeros((n_classes, n_features))
        b = np.zeros((n_classes,1))
    return W, b


def one_hot_encode(Y, n_classes):
    """
    One hot encodes for classes.
    """
    n_samples = Y.shape[0]
    y_one_hot = np.zeros((n_samples, n_classes))
    y_one_hot[np.arange(n_samples), Y] = 1
    return y_one_hot

def forward(X, W, b):
    # Y = WX + B =>  XW^T + B.T
    return np.matmul(X, W.T) + b.T


def compute_loss(y_pred, y_true):
    return  np.sum((y_true-y_pred) ** 2) / (2 * y_pred.shape[0])

def backward(X, y_pred, y_true):
    n_samples = X.shape[0]

    #dL/dY (n_samples, n_classes) => d_loss
    #d_loss.T : (n_classes, n_samples) 
    #X: (n_samples, n_features) 

    d_loss =  (y_pred - y_true) / n_samples # (n_samples, n_classes)
   
    dW = np.matmul(d_loss.T, X) #Gradient w.r.t. W: dL/dW = dL/dy * dy/dW = d_loss.T @ X

    # dl/db = sum(dL/dY)
    db = np.sum(d_loss, axis=0, keepdims=True).T

    return dW, db

def update_weights(W, b, dW, db, learning_rate=0.01):
    W -= learning_rate * dW
    b -= learning_rate * db
    return W, b

def fit(X_train, Y_train, n_classes, epochs=100, batch_size=64, xavier = True, _learning_rate=0.01, verbose=True):
    n_samples = X_train.shape[0]
    y_train_one_hot = one_hot_encode(Y_train, n_classes)

    W, b = init_weights(X_train.shape[1], n_classes, xavier)

    epoch_loss = 0
    n_batches = 0

    # Training loop
    print("Starting Training for Linear Classifier")

    for epoch in range(epochs):
        indicies = np.random.permutation(n_samples)
        X_shuffled = X_train[indicies]
        Y_shuffled = y_train_one_hot[indicies]

        for i in range(0, n_samples, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            Y_batch = Y_shuffled[i:i+batch_size]

            
            Y_pred = forward(X_batch, W, b)

            loss = compute_loss(Y_pred, Y_batch)
            epoch_loss += loss
            n_batches += 1
            
            dW, db = backward(X_batch, Y_pred, Y_batch)

            W, b = update_weights(W, b, dW, db, learning_rate=_learning_rate)
        
        #If verbose
        if verbose and (epoch%10==0):
            print(f"Epoch: {epoch} avg loss: {epoch_loss/n_batches}")

    print("Training complete")
    return W, b

def predict(X, W, b):
    logits = forward(X, W, b)
    return np.argmax(logits, axis=1)

# Linear Classifier standalone test
if __name__ == "__main__":
    images, labels = load_mnist_data()
    images_flat, images_normalized, labels = preprocess_mnist()
    X_train, X_test, Y_train, Y_test = train_test_split(
        images_normalized, labels, train_size=0.8, random_state=42
    )
    classes = np.unique(Y_train)
    n_classes = len(classes)

    W, b = fit(X_train, Y_train, n_classes, xavier=False)

    linear_pred = predict(X_test, W, b)
    linear_acc = np.mean(linear_pred == Y_test)

    print(f"Linear Acc {linear_acc*100:.2f}%")