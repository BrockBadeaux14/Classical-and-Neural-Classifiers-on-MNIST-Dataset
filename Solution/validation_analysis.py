import random
from dataclasses import dataclass
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

from load_mnist_data import load_mnist_data
from data_preprocessor import preprocess_mnist
from train_test_split import train_test_split

from CNN import fit_cnn, predict_cnn
from knn_classifier import knn_predict
from linear_classifier import fit as linear_fit, predict as linear_predict
from naive_bayes import fit as nb_fit, predict as nb_predict
from mlp import fit_mlp_model, predict_mlp


NUM_CLASSES = 10
SEED = 42


@dataclass
class ModelReport:
    name: str
    test_acc: float
    y_true: np.ndarray
    y_pred: np.ndarray


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_linear(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, xavier: bool, _learning_rate: float, epochs: int) -> ModelReport:
    W, b = linear_fit(X_train, y_train, NUM_CLASSES, epochs=epochs, batch_size=128, xavier=xavier, _learning_rate=_learning_rate, verbose=False)
    preds = linear_predict(X_test, W, b)
    acc = np.mean(preds == y_test) 
    return ModelReport("Linear", float(acc), y_test, preds)


def evaluate_naive_bayes(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, threshold: float, alpha: float) -> ModelReport:
    params = nb_fit(X_train, y_train, threshold=threshold, alpha=alpha)
    preds = nb_predict(X_test, params)
    acc = np.mean(preds == y_test)
    return ModelReport("Naive Bayes", float(acc), y_test, preds)


def evaluate_knn(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, k_neighbors: int, distance_metric: str) -> ModelReport:
    preds = knn_predict(X_train, y_train, X_test, k=k_neighbors, distance_metric=distance_metric)
    acc = np.mean(preds == y_test)
    return ModelReport(f"KNN (k={k_neighbors}, metric={distance_metric})", float(acc), y_test, preds)


def evaluate_mlp(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, learning_rate: float, momentum: float, epochs: int, device: torch.device, weight_decay: float, loss_function: str) -> ModelReport:
    model = fit_mlp_model(X_train, y_train, learning_rate=learning_rate, momentum=momentum, epochs=epochs, batch_size=256, device=device, weight_decay=weight_decay, loss_function=loss_function)
    preds = predict_mlp(model, X_test, device=device)
    acc = np.mean(preds == y_test)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return ModelReport("MLP", float(acc), y_test, preds)


def evaluate_cnn(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, device: torch.device, epochs: int, weight_decay: float, lr: float, optimizer: str, scheduler: str) -> ModelReport:
    model, mean, std = fit_cnn(X_train, y_train, epochs=epochs, batch_size=256, 
                               device=device, weight_decay=weight_decay, lr=lr,
                               optimizer=optimizer, scheduler=scheduler
                               )
    preds = predict_cnn(model, X_test, mean=mean, std=std, device=device)
    acc = np.mean(preds == y_test)
    print(f"CNN Test Accuracy: {acc*100:.4f}%")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return ModelReport("CNN", float(acc), y_test, preds)


def visualize_confusion_matrix(report: ModelReport) -> None:
    cm = confusion_matrix(report.y_true, report.y_pred, labels=list(range(NUM_CLASSES)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(NUM_CLASSES)))
    disp.plot(cmap="Blues", colorbar=False)
    plt.title(f"{report.name} Confusion Matrix")
    plt.tight_layout()
    outfile = report.name.lower().replace(" ", "_") + "_cm.png"
    plt.savefig(outfile)
    plt.close()
    print(f"\n{report.name} classification report:\n{classification_report(report.y_true, report.y_pred)}")
    print(f"Saved confusion matrix figure to {outfile}")


def run_analysis() -> None:
    set_seed()

    images, labels = load_mnist_data()
    _, normalized_flat, _ = preprocess_mnist(images, labels)

    normalized_images = normalized_flat.reshape(images.shape[0], 28, 28)
    tensor = np.expand_dims(normalized_images, 1)

    X_train_flat, X_test_flat, y_train, y_test = train_test_split(normalized_flat, labels, train_size=0.8, random_state=SEED)
    X_train_cnn, X_test_cnn, _, _ = train_test_split(tensor, labels, train_size=0.8, random_state=SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results: List[ModelReport] = []
    cnn_results: list[tuple[int, float, str, str, float]] = []

    '''
    for k in np.arange(1, 21):
        for metric in ["euclidean", "cosine"]:
            print(f"Evaluating KNN with k={k} and distance metric={metric}")
            knn_report = evaluate_knn(X_train_flat, y_train, X_test_flat, y_test, k_neighbors=k, distance_metric=metric)
            knn_report.name = f"KNN (k={k}, metric={metric})"
            results.append(knn_report)

    ks = np.arange(1, 21)
    euclidean_accuracies = [report.test_acc for report in results if report.name.startswith("KNN") and "euclidean" in report.name]
    cosine_accuracies = [report.test_acc for report in results if report.name.startswith("KNN") and "cosine" in report.name]
    plt.plot(ks, euclidean_accuracies, marker='o', label='Euclidean Distance')
    plt.plot(ks, cosine_accuracies, marker='s', label='Cosine Distance')
    plt.title('KNN Accuracy vs K Value')
    plt.xlabel('K Value')
    plt.ylabel('Accuracy')
    plt.xticks(ks)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
    plt.close()
            
    for lr in [0.001, 0.01]:
        for epochs in [10, 50, 100, 200, 500]:
            for xavier in [True, False]:
                linear_report = evaluate_linear(X_train_flat, y_train, X_test_flat, y_test, xavier=xavier, _learning_rate=lr, epochs=epochs)
                linear_report.name = f"Linear (xavier={xavier}, lr={lr}, epochs={epochs})"
                results.append(linear_report)
    
    for xavier in [True, False]:
        for lr in [0.001, 0.01]:
            epochs_list = [10, 50, 100, 200, 500]
            accuracies = [report.test_acc for report in results if report.name.startswith("Linear") and f"xavier={xavier}" in report.name and f"lr={lr}" in report.name]
            plt.plot(epochs_list, accuracies, marker='o', label=f'LR={lr}, Xavier={xavier}')
    plt.title('Linear Classifier Accuracy vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.xticks(epochs_list)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
    plt.close()
    
    

    for threshold in [0.3, 0.4, 0.45, 0.5, 0.55, 0.6]:
        for alpha in [0.000001, 0.001, 0.1, 0.5]:
            nb_report = evaluate_naive_bayes(X_train_flat, y_train, X_test_flat, y_test, threshold=threshold, alpha=alpha)
            nb_report.name = f"Naive Bayes (threshold={threshold}, alpha={alpha})"
            results.append(nb_report)

    #Create graph for Naive Bayes accuracy vs threshold vs alpha using results collected above
    for alpha in [0.000001, 0.001, 0.1, 0.5]:
        thresholds = [0.3, 0.4, 0.45, 0.5, 0.55, 0.6]
        accuracies = [report.test_acc for report in results if report.name.startswith("Naive Bayes") and f"alpha={alpha}" in report.name]
        plt.plot(thresholds, accuracies, marker='o', label=f'Alpha={alpha}')
    plt.title('Naive Bayes Accuracy vs Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('Accuracy')
    plt.xticks(thresholds)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
    plt.close()
    
    
    for lr in [0.1, 0.01]:
        for epochs in [1, 3, 5, 10, 25, 50, 100]:
            for weight_decay in [0.0, 1e-5]:
                for loss_function in ["cross_entropy", "mse"]:
                    print(f"Evaluating MLP with learning rate={lr}, momentum={0.9}, and epochs={epochs}")
                    mlp_report = evaluate_mlp(X_train_flat, y_train, X_test_flat, y_test, learning_rate=lr, momentum=0.9, epochs=epochs, device=device, weight_decay=weight_decay, loss_function=loss_function)
                    mlp_report.name = f"MLP (lr={lr}, momentum={0.9}, epochs={epochs}, weight_decay={weight_decay}, loss_function={loss_function})"
                    results.append(mlp_report)

    
    for lr in [0.1, 0.01]:
        for weight_decay in [0.0, 1e-5]:
            epochs_list = [1, 3, 5, 10, 25, 50, 100]
            accuracies = [report.test_acc for report in results if report.name.startswith("MLP") and f"lr={lr}" in report.name and f"weight_decay={weight_decay}" in report.name]
            plt.plot(epochs_list, accuracies, marker='o', label=f'LR={lr}, Weight Decay={weight_decay}')
    plt.title("MLP Accuracy vs Epochs")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.xticks(epochs_list)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"MLP_accuracy_vs_Epochs(loss_function).png")
    plt.close()
    

    
    mlp_report = evaluate_mlp(X_train_flat, y_train, X_test_flat, y_test, device)
    results.append(mlp_report)
    

    
    for epochs in [1, 3, 5, 10, 25, 50, 100, 200]:
        for weight_decay in [1e-3]:
            for loss_function in ["cross_entropy"]:
                for optimizer in ["SGD"]:
                    for scheduler in ["cosine", None]:
                        print(f"Evaluating CNN with epochs={epochs}, weight_decay={weight_decay}, optimizer={optimizer}, scheduler={scheduler}")
                        cnn_report = evaluate_cnn(
                            X_train_cnn,
                            y_train,
                            X_test_cnn,
                            y_test,
                            device,
                            epochs=epochs,
                            weight_decay=weight_decay,
                            lr=2e-3,
                            optimizer=optimizer,
                            scheduler=scheduler,
                        )
                        cnn_report.name = f"CNN (epochs={epochs}, weight_decay={weight_decay}, optimizer={optimizer}, scheduler={scheduler})"
                        results.append(cnn_report)
                        cnn_results.append((epochs, weight_decay, optimizer, scheduler, cnn_report.test_acc))
    # Create graph for CNN's accuracy vs epochs, weight_decay, optimizer, scheduler using results collected above
    cnn_epoch_ticks = sorted({epoch for epoch, *_ in cnn_results})

    for weight_decay in [1e-3]:
        for optimizer in ["SGD"]:
            for scheduler in ["cosine"]:
                subset = [
                    (epoch, acc)
                    for epoch, wd, opt, sch, acc in cnn_results
                    if wd == weight_decay and opt == optimizer and sch == scheduler
                ]
                if not subset:
                    continue
                subset.sort(key=lambda item: item[0])
                epoch_values, accuracy_values = zip(*subset)
                plt.plot(epoch_values, accuracy_values, marker='o', label=f'Weight Decay={weight_decay}, Optimizer={optimizer}, Scheduler={scheduler}')
    plt.title("CNN Accuracy vs Epochs")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    if cnn_epoch_ticks:
        plt.xticks(cnn_epoch_ticks)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"CNN.png")
    plt.close()


    '''
    
    #for report in results:
    #    visualize_confusion_matrix(report)



if __name__ == "__main__":
    run_analysis()
