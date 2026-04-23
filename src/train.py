import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_data_loaders
from model import create_pneumonia_model

def train_model():
    # 1. Setup Data
    train_loader, val_loader, test_loader, classes = get_data_loaders()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # 2. Initialize Model
    model = create_pneumonia_model().to(device)

    # 3. Address the 3-Class Imbalance
    # Counts explicitly verified from your dataset.py output
    count_normal = 1341
    count_other = 700
    count_pneumonia = 3875
    total_images = count_normal + count_other + count_pneumonia

    # Formula: total_samples / (num_classes * class_samples)
    weight_normal = total_images / (3 * count_normal)       # ~1.47
    weight_other = total_images / (3 * count_other)         # ~2.81 (Highest penalty)
    weight_pneumonia = total_images / (3 * count_pneumonia) # ~0.50
    
    print("\n--- Loss Function Weights ---")
    print(f"Normal Penalty: {weight_normal:.2f}")
    print(f"Other Penalty: {weight_other:.2f}")
    print(f"Pneumonia Penalty: {weight_pneumonia:.2f}\n")

    # Order matches the alphabetized classes: 0: NORMAL, 1: OTHER, 2: PNEUMONIA
    class_weights = torch.tensor([weight_normal, weight_other, weight_pneumonia], dtype=torch.float).to(device)
    
    # 4. Define Loss and Optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)

    # 5. Training and Validation Loop
    num_epochs = 3 
    
    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        # --- Validation Phase ---
        model.eval() 
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        epoch_train_loss = running_loss / len(train_loader)
        epoch_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {epoch_train_loss:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} | "
              f"Val Accuracy: {val_accuracy:.2f}%")

    # 6. Save the Trained Weights
    save_path = "pneumonia_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\nTraining complete. Model saved to {save_path}")

if __name__ == "__main__":
    train_model()