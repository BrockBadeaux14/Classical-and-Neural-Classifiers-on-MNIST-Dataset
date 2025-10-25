from load_mnist_data import load_mnist_data
from data_preprocessor import preprocess_mnist
from train_test_split import train_test_split
import numpy as np


def binarize(X_train, threshold):
    return (X_train > threshold).astype(int)


def fit(X_train, Y_train, threshold, alpha):
    X_binary = binarize(X_train, threshold)  # binarize data

    classes = np.unique(Y_train)
    n_classes = len(classes)

    n_features = X_train.shape[1]

    priors = np.zeros(n_classes)
    feature_probs = np.zeros((n_classes, n_features))

    for idx, c in enumerate(classes):
        X_c = X_binary[Y_train == c]
        priors[idx] = X_c.shape[0] / X_train.shape[0]
        feature_probs[idx] = (X_c.sum(axis=0) + alpha) / (X_c.shape[0] + 2 * alpha)

    return {
        "classes": classes,
        "threshold": threshold,
        "priors": priors,
        "feature_probs": feature_probs,
    }
    
        
def predict(X_test, params):
    threshold = params["threshold"]
    classes = params["classes"]
    priors = params["priors"]
    feature_probs = params["feature_probs"]

    X_binary = binarize(X_test, threshold)

    log_priors = np.log(priors)
    log_probs_on = np.log(feature_probs)
    log_probs_off = np.log(1 - feature_probs)

    n_samples = X_test.shape[0]
    log_posteriors = np.zeros((n_samples, len(classes)))

    for idx in range(len(classes)):
        log_posteriors[:, idx] = (
            log_priors[idx]
            + (X_binary * log_probs_on[idx]).sum(axis=1)
            + ((1 - X_binary) * log_probs_off[idx]).sum(axis=1)
        )

    return classes[np.argmax(log_posteriors, axis=1)]



if __name__ == "__main__":
    images, imagesN, labels = preprocess_mnist()
    X_train, X_test, Y_train, Y_test = train_test_split(imagesN, labels, random_state=67)

    params = fit(X_train, Y_train, threshold=0.5, alpha=1)
    nb_pred = predict(X_test, params)
    nb_acc = np.mean(nb_pred == Y_test)

    print(f"acc: {nb_acc}")