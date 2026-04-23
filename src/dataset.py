import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

def get_data_loaders(data_dir=None, batch_size=32):
    # Dynamically find the data directory if not explicitly provided
    if data_dir is None:
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(BASE_DIR, "data")

    # Standard transforms for ResNet architectures
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # These specific mean and std values are required when using pre-trained PyTorch models
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load datasets
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform)
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=transform)

    # Double-check class distribution dynamically
    train_targets = train_dataset.targets
    
    print(f"--- Data Verification ---")
    print(f"Classes found: {train_dataset.classes}")
    
    # Loop through the classes PyTorch found and count them safely
    for class_name, idx in train_dataset.class_to_idx.items():
        count = train_targets.count(idx)
        print(f"{class_name} Images: {count}")
        
    print(f"Total Train Images: {len(train_dataset)}")

    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, train_dataset.classes

if __name__ == "__main__":
    try:
        # Running the script directly will trigger the verification printouts
        train_loader, val_loader, test_loader, classes = get_data_loaders()
        print("\nData Loaders created successfully. Ready for modeling.")
    except FileNotFoundError as e:
        print(f"ERROR: Could not find data directory. Check your folder structure. Details: {e}")