from data_preprocessor import preprocess_mnist
from train_test_split import train_test_split

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def create_mlp_model() -> nn.Sequential:
    layers = [
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    ]
    return nn.Sequential(*layers)

def fit_mlp_model(
    X: np.ndarray,
    y: np.ndarray,
    *,
    learning_rate: float,
    momentum: float = 0.9,
    epochs: int = 5,
    batch_size: int = 128,
    device: torch.device | None = None,
    weight_decay: float = 1e-4,
    loss_function: str = "cross_entropy",
    optimizer_type: str = "sgd",
) -> nn.Module:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_mlp_model().to(device)
    dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    if optimizer_type == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                              momentum=momentum, weight_decay=weight_decay)
    elif optimizer_type == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate,
                               weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer_type: {optimizer_type}")

    if loss_function == "cross_entropy":
        criterion = nn.CrossEntropyLoss()
    elif loss_function == "mse":
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"Unknown loss_function: {loss_function}")

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)  # [N, C]

            if loss_function == "cross_entropy":
                # targets shape [N] (class indices)
                loss = criterion(logits, batch_y)
            else:
                # MSE needs same shape: one-hot targets vs probabilities
                probs = torch.softmax(logits, dim=1)                     # [N, C]
                y_onehot = nn.functional.one_hot(batch_y, num_classes=logits.shape[1]).float()  # [N, C]
                loss = criterion(probs, y_onehot)

            loss.backward()
            optimizer.step()
            running += loss.item() * batch_x.size(0)

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running/len(dataset):.6f}")

    return model

def predict_mlp(model: nn.Module, X: np.ndarray, *, device: torch.device | None = None, batch_size: int = 256) -> np.ndarray:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, X.shape[0], batch_size):
            batch = torch.from_numpy(X[start : start + batch_size]).float().to(device)
            preds.append(model(batch).argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)

if __name__ == "__main__":
    _, images_normalized, labels = preprocess_mnist()
    X_train, X_test, Y_train, Y_test = train_test_split(
        images_normalized, labels, train_size=0.8, random_state=42
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = fit_mlp_model(X_train, Y_train, epochs=10, batch_size=128, device=device, learning_rate=0.01, dropout=True, loss_function="mse")
    predictions = predict_mlp(model, X_test, device=device)
    test_acc = (predictions == Y_test).mean()
    print(f"Test accuracy: {test_acc:.4f}")