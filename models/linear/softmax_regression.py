import torch
import torch.nn as nn

class SoftmaxRegression(nn.Module):
    """Softmax回归（用于多分类）"""
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        # PyTorch的CrossEntropyLoss自动包含softmax
        return self.linear(x)