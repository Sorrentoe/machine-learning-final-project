import torch
import torch.nn as nn
import torchvision.models as models

def create_pneumonia_model():
    # 1. Load the pre-trained ResNet18 backbone
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    # 2. Freeze the feature extraction layers to preserve pre-trained knowledge
    for param in model.parameters():
        param.requires_grad = False

    # 3. Replace the classification head
    num_features = model.fc.in_features
    
    # We now output 3 values (Logits for NORMAL, OTHER, and PNEUMONIA)
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3), # Essential for preventing overfitting
        nn.Linear(512, 3) # UPDATED: Changed from 2 to 3
    )

    return model

if __name__ == "__main__":
    # Rigorous testing: verify the tensor shapes before training
    test_model = create_pneumonia_model()
    dummy_input = torch.randn(1, 3, 224, 224) 
    output = test_model(dummy_input)
    print(f"Model output shape: {output.shape} (Expected: 1, 3)")