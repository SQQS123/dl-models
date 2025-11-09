import torch
import torch.nn as nn

class MLP(nn.Module):
    """多层感知机
    参数:
        input_dim: 输入特征维度
        hidden_dims: 隐藏层维度列表
        output_dim: 输出维度
        dropout: Dropout比率
        activation: 激活函数类型
    """
    def __init__(self, input_dim, hidden_dims, output_dim, dropout=0.1, activation='relu'):
        super().__init__()
        
        self.activation_map = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid()
        }
        self.activation = self.activation_map.get(activation, nn.ReLU())
        
        layers = []
        dims = [input_dim] + hidden_dims
        
        # 构建隐藏层
        for i in range(len(dims)-1):
            layers.extend([
                nn.Linear(dims[i], dims[i+1]),
                self.activation,
                nn.Dropout(dropout)
            ])
        
        # 输出层
        layers.append(nn.Linear(dims[-1], output_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)