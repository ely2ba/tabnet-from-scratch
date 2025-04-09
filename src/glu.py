# src/glu.py

import torch
import torch.nn as nn

class GatedLinearUnit(nn.Module):
    """
    A Gated Linear Unit (GLU) as used in TabNet.
    It applies two linear transformations:
        - One for the output
        - One for the gate (followed by sigmoid)
    Then returns their element-wise product.
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.fc_gate = nn.Linear(input_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.fc(x) * self.sigmoid(self.fc_gate(x))
