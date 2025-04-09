# src/tabnet.py

import torch
import torch.nn as nn
from src.tabnet_step import TabNetStep



class TabNet(nn.Module):
    """
    Full TabNet model composed of multiple decision steps.

    Args:
        input_dim (int): Number of input features.
        n_d (int): Feature dimension used in transformers and decision steps.
        n_steps (int): Number of decision steps.
        output_dim (int): Final prediction output dimension (e.g., num classes or 1).
    """

    def __init__(self, input_dim, n_d, n_steps, output_dim):
        super().__init__()

        self.input_dim = input_dim
        self.n_d = n_d
        self.n_steps = n_steps
        self.output_dim = output_dim

        # Shared transformer layers (used across all decision steps)
        self.shared_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim if i == 0 else n_d, n_d),
                nn.BatchNorm1d(n_d),
                nn.ReLU()
            ) for i in range(2)  # You can make this configurable
        ])

        # List of decision steps
        self.steps = nn.ModuleList([
            TabNetStep(
                input_dim=input_dim,
                n_d=n_d,
                output_dim=n_d,
                shared_transformer=self.shared_layers
            ) for _ in range(n_steps)
        ])

        # Final output projection
        self.final_projection = nn.Linear(n_d, output_dim)

    def forward(self, x):
        """
        Forward pass through the TabNet model.

        Args:
            x (Tensor): Input tensor of shape (batch_size, input_dim)

        Returns:
            logits (Tensor): Output logits of shape (batch_size, output_dim)
            aggregated_mask (Tensor): Summed attention masks from all steps, shape (batch_size, input_dim)
        """
        batch_size = x.size(0)
        prior = torch.ones_like(x)
        aggregated_mask = torch.zeros_like(x)
        output = 0

        for step in self.steps:
            decision_out, next_feat, prior, mask = step(x, prior)
            output += decision_out
            aggregated_mask += mask

        logits = self.final_projection(output)
        return logits, aggregated_mask

