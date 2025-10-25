import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Tuple

from load_mnist_data import load_mnist_data
from train_test_split import train_test_split


class CNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),

            nn.Flatten(),
            nn.Linear(3136, 512),    
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.25),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.5),
            nn.Linear(1024, 10),
        )
       
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _default_device(device: torch.device | None) -> torch.device:
    if device is not None:
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fit_cnn(
    X: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int = 20,
    batch_size: int = 256,
    device: torch.device | None = None,
    weight_decay: float = 1e-4,
    lr: float = 2e-3,
    optimizer: str = "SGD",
    scheduler: str = "cosine",

) -> Tuple[nn.Module, float, float]:
    device = _default_device(device)

    mean = float(X.mean())
    std = float(X.std() + 1e-7)
    X_norm = (X - mean) / std

    dataset = TensorDataset(torch.from_numpy(X_norm).float(), torch.from_numpy(y).long())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = CNN().to(device)

    criterion =  nn.CrossEntropyLoss()

    if optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for _ in range(epochs):
        model.train()
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        if scheduler == "cosine":
            scheduler.step()
        print(f"Epoch {_ + 1}/{epochs}, Loss: {loss.item()}")

    return model, mean, std


def predict_cnn(
    model: nn.Module,
    X: np.ndarray,
    *,
    mean: float,
    std: float,
    device: torch.device | None = None,
    batch_size: int = 256,
) -> np.ndarray:
    device = _default_device(device)
    model.eval()
    preds: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, X.shape[0], batch_size):
            batch = torch.from_numpy((X[start : start + batch_size] - mean) / std).float().to(device)
            preds.append(model(batch).argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)



if __name__ == "__main__":
    images, labels = load_mnist_data()

    images = images.astype(np.float32) / 255.0
    images = np.expand_dims(images, axis=1)

    X_train, X_test, Y_train, Y_test = train_test_split(
        images, labels, train_size=0.8, random_state=42
    )

    train_mean = X_train.mean()
    train_std = X_train.std() + 1e-7
    X_train_norm = (X_train - train_mean) / train_std
    X_test_norm = (X_test - train_mean) / train_std

    device = _default_device(None)
    print(f"Using device: {device}")

    X_train_tensor = torch.from_numpy(X_train_norm).float()
    Y_train_tensor = torch.from_numpy(Y_train).long()
    X_test_tensor = torch.from_numpy(X_test_norm).float()
    Y_test_tensor = torch.from_numpy(Y_test).long()

    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    model = CNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

    #Use fit function to fit
    epochs = 5
    model, train_mean, train_std = fit_cnn(
        X_train,
        Y_train,
        epochs=epochs,
        batch_size=256,
        device=device,
        weight_decay=1e-4,
        lr=2e-3,
        optimizer="AdamW",
        scheduler="cosine",
    )
    #Predict on test set
    final_test_preds = predict_cnn(
        model,
        X_test,
        mean=train_mean,
        std=train_std,
        device=device,
        batch_size=256,
    )

    final_test_acc = (final_test_preds == Y_test).mean()
    print(f"Final test accuracy: {final_test_acc:.4f}")
