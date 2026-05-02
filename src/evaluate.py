import os
import torch
from sklearn.metrics import confusion_matrix, classification_report
from dataset import get_data_loaders
from model import create_pneumonia_model

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODEL_PATH = os.path.join(_PROJECT_ROOT, "pneumonia_model.pth")


def evaluate_model():
    print("Initializing Evaluation Protocol...")

    _, _, test_loader, classes = get_data_loaders(batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")

    model = create_pneumonia_model(pretrained_backbone=False).to(device)
    try:
        try:
            ckpt = torch.load(_MODEL_PATH, map_location=device, weights_only=True)
        except TypeError:
            ckpt = torch.load(_MODEL_PATH, map_location=device)
        model.load_state_dict(ckpt)
        print(f"Successfully loaded trained weights from {_MODEL_PATH}")
    except FileNotFoundError:
        print(f"ERROR: model weights not found at {_MODEL_PATH}. Please run train.py first.")
        return

    model.eval()

    all_preds = []
    all_labels = []

    print("\nRunning test data through the network. Please wait...")

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            logits_3, _ = model(images)
            _, predicted = torch.max(logits_3, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("\n" + "=" * 50)
    print("FINAL MODEL PERFORMANCE")
    print("=" * 50)

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=classes))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)

    row_format = "{:>15}" * (len(classes) + 1)
    print(row_format.format("", *classes))
    for i, row in enumerate(cm):
        print(row_format.format(classes[i], *row))

    print("\nEvaluation complete. Use these metrics for your final project documentation.")


if __name__ == "__main__":
    evaluate_model()
