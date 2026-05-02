import os
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from dataset import get_data_loaders
from model import create_pneumonia_model

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Index of OTHER in alphabetized ImageFolder classes: NORMAL, OTHER, PNEUMONIA
OTHER_CLASS_IDX = 1


def _class_weights_from_dataset(train_dataset, num_classes, device):
    counts = Counter(train_dataset.targets)
    total = len(train_dataset)
    weights = [total / (num_classes * counts[i]) for i in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float, device=device)


def train_model():
    train_loader, val_loader, test_loader, classes = get_data_loaders()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    model = create_pneumonia_model().to(device)

    train_dataset = train_loader.dataset
    num_classes = len(classes)
    class_weights = _class_weights_from_dataset(train_dataset, num_classes, device)

    count_normal = sum(1 for t in train_dataset.targets if t == 0)
    count_other = sum(1 for t in train_dataset.targets if t == OTHER_CLASS_IDX)
    count_pneumonia = sum(1 for t in train_dataset.targets if t == 2)

    print("\n--- Class counts (train) ---")
    print(f"NORMAL: {count_normal} | OTHER: {count_other} | PNEUMONIA: {count_pneumonia}")
    print(f"Total train: {len(train_dataset)}")

    print("\n--- Loss function class weights ---")
    print(
        f"NORMAL: {class_weights[0].item():.3f} | OTHER: {class_weights[1].item():.3f} | "
        f"PNEUMONIA: {class_weights[2].item():.3f}\n"
    )

    criterion_ce = nn.CrossEntropyLoss(weight=class_weights)
    criterion_gate = nn.BCEWithLogitsLoss()
    gate_loss_weight = 0.5

    trainable = list(model.classifier.parameters()) + list(model.xray_gate.parameters())
    optimizer = optim.SGD(trainable, lr=0.001, momentum=0.9)

    num_epochs = 3

    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logits_3, xray_logit = model(images)
            is_xray = (labels != OTHER_CLASS_IDX).float()
            loss_ce = criterion_ce(logits_3, labels)
            loss_gate = criterion_gate(xray_logit, is_xray)
            loss = loss_ce + gate_loss_weight * loss_gate

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits_3, xray_logit = model(images)
                loss_ce = criterion_ce(logits_3, labels)
                is_xray = (labels != OTHER_CLASS_IDX).float()
                loss_gate = criterion_gate(xray_logit, is_xray)
                loss = loss_ce + gate_loss_weight * loss_gate
                val_loss += loss.item()

                _, predicted = torch.max(logits_3.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_train_loss = running_loss / len(train_loader)
        epoch_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total

        print(f"Epoch [{epoch + 1}/{num_epochs}] "
              f"Train Loss: {epoch_train_loss:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} | "
              f"Val Accuracy: {val_accuracy:.2f}%")

    save_path = os.path.join(_PROJECT_ROOT, "pneumonia_model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"\nTraining complete. Model saved to {save_path}")


if __name__ == "__main__":
    train_model()
