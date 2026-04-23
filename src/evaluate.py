import torch
from sklearn.metrics import confusion_matrix, classification_report
from dataset import get_data_loaders
from model import create_pneumonia_model

def evaluate_model():
    print("Initializing Evaluation Protocol...")
    
    # 1. Load the test data
    # We only care about the test_loader, so we ignore the train and val loaders
    _, _, test_loader, classes = get_data_loaders(batch_size=32)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on device: {device}")

    # 2. Load the trained model
    model = create_pneumonia_model().to(device)
    try:
        model.load_state_dict(torch.load("pneumonia_model.pth", map_location=device))
        print("Successfully loaded trained weights from pneumonia_model.pth")
    except FileNotFoundError:
        print("ERROR: pneumonia_model.pth not found. Please run train.py first.")
        return

    model.eval() # CRITICAL: Turn off dropout and batch normalization layers

    # 3. Storage for our predictions and the actual truth
    all_preds = []
    all_labels = []

    print("\nRunning test data through the network. Please wait...")
    
    # 4. The Inference Loop
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 5. Calculate and Display Metrics
    print("\n" + "="*50)
    print("FINAL MODEL PERFORMANCE")
    print("="*50)
    
    # Classification Report (Precision, Recall, F1-Score)
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=classes))

    # Confusion Matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    
    # Formatting the matrix for clean terminal reading
    row_format = "{:>15}" * (len(classes) + 1)
    print(row_format.format("", *classes))
    for i, row in enumerate(cm):
        print(row_format.format(classes[i], *row))
        
    print("\nEvaluation complete. Use these metrics for your final project documentation.")

if __name__ == "__main__":
    evaluate_model()