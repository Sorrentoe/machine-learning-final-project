import torch
import torch.nn as nn
import torchvision.models as models


class PneumoniaDetector(nn.Module):
    """
    Frozen ResNet18 backbone with (1) a 3-class head for NORMAL / OTHER / PNEUMONIA
    and (2) a binary x-ray gate (chest X-ray vs not-OTHER) trained jointly.
    """

    def __init__(self):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT
        backbone = models.resnet18(weights=weights)
        for param in backbone.parameters():
            param.requires_grad = False

        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        num_features = backbone.fc.in_features

        self.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 3),
        )
        self.xray_gate = nn.Linear(num_features, 1)

    def forward(self, x):
        features = self.backbone(x)
        features = torch.flatten(features, 1)
        logits_3 = self.classifier(features)
        xray_logit = self.xray_gate(features).squeeze(-1)
        return logits_3, xray_logit


def create_pneumonia_model():
    return PneumoniaDetector()


if __name__ == "__main__":
    test_model = create_pneumonia_model()
    dummy_input = torch.randn(1, 3, 224, 224)
    logits_3, xray_logit = test_model(dummy_input)
    print(f"Classifier logits shape: {logits_3.shape} (expected 1, 3)")
    print(f"X-ray gate shape: {xray_logit.shape} (expected 1)")
