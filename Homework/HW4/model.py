import torch.nn as nn

class CustomViTClassifier(nn.Module):
    def __init__(self, vit_model, num_labels=2):
        super().__init__()
        self.vit = vit_model
        self.config = self.vit.config
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_labels)
    
    def forward(self, pixel_values):
        # 只接受 pixel_values 參數
        outputs = self.vit(pixel_values=pixel_values)
        logits = self.classifier(outputs.pooler_output)
        return logits