import numpy as np

from data_preprocessor import preprocess_mnist
from train_test_split import train_test_split


def multi_distance(X_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    # ||X_train - X_test|| = sqrt(||X_train||^2 + ||X_test||^2 - 2*X_train^T*X_test)
    
    # X_train: (N_tr, D), X_test: (N_te, D)
    train_norms_sq = (X_train ** 2).sum(axis=1).reshape(-1, 1)   # (N_tr, 1)
    test_norms_sq  = (X_test  ** 2).sum(axis=1).reshape(1, -1)   # (1, N_te)
    products = X_train @ X_test.T                                 # (N_tr, N_te)
    d2 = train_norms_sq + test_norms_sq - 2 * products            # (N_tr, N_te)
    return np.sqrt(np.maximum(d2, 0.0))

#Cosine distance
def cosine_distance(X_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    # Cosine distance = 1 - (X_train . X_test) / (||X_train|| * ||X_test||)
    train_norms = np.linalg.norm(X_train, axis=1).reshape(-1, 1)  # (N_tr, 1)
    test_norms  = np.linalg.norm(X_test, axis=1).reshape(1, -1)   # (1, N_te)
    products = X_train @ X_test.T                                 # (N_tr, N_te)
    cosine_similarities = products / (train_norms * test_norms + 1e-10)  # (N_tr, N_te)
    return 1.0 - cosine_similarities


def knn_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    *,
    k: int = 3,
    distance_metric: str = "euclidean"
) -> np.ndarray:
    if distance_metric == "euclidean":
        distances = multi_distance(X_train, X_test) #(N_tr, N_te)
        knn_indices = np.argpartition(distances, kth=k - 1, axis=0)[:k, :]
    elif distance_metric == "cosine":
        distances = cosine_distance(X_train, X_test) #(N_tr, N_te)
        knn_indices = np.argpartition(distances, kth=k - 1, axis=0)[:k, :]
    else:
        raise ValueError(f"Unsupported distance metric: {distance_metric}")
    preds = []
    for column in range(X_test.shape[0]):
        votes = y_train[knn_indices[:, column]]
        preds.append(np.bincount(votes, minlength=10).argmax())
    return np.array(preds, dtype=np.int64)


if __name__ == "__main__":
    _, images_normalized, labels = preprocess_mnist()
    X_train, X_test, Y_train, Y_test = train_test_split(
        images_normalized, labels, random_state=67, train_size=0.8
    )

    predictions = knn_predict(X_train, Y_train, X_test, k=3)
    accuracy = np.mean(predictions == Y_test)
    print(f"KNN Acc: {accuracy * 100:.2f}%")