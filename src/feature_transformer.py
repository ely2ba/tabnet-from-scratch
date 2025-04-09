import torch
import torch.nn as nn

class FeatureTransformer(nn.Module):
    """
    Implements the Feature Transformer block in TabNet, consisting of:
        - Shared layers (optionally passed in from outside)
        - Step-specific independent layers

    This module processes the input through the shared layers and then
    through a unique stack of independent layers for each decision step.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input features.
    n_d : int
        Hidden dimension size used throughout the transformer.
    n_steps : int
        Number of decision steps (each with its own independent block).
    n_shared : int
        Number of shared layers (ignored if shared_layers is passed).
    n_independent : int
        Number of step-specific (independent) layers for each step.
    shared_layers : nn.ModuleList or None
        If provided, uses these shared layers instead of creating new ones.
    """
    def __init__(self, input_dim, n_d, n_steps, n_shared=2, n_independent=2, shared_layers=None):
        super().__init__()

        # Use externally passed shared layers or initialize fresh ones
        if shared_layers is not None:
            self.shared = shared_layers
        else:
            self.shared = nn.ModuleList()
            for i in range(n_shared):
                self.shared.append(nn.Sequential(
                    nn.Linear(input_dim if i == 0 else n_d, n_d),
                    nn.BatchNorm1d(n_d),
                    nn.ReLU()
                ))

        # Build a list of step-specific blocks (one per step)
        self.step_specific = nn.ModuleList()
        for step in range(n_steps):
            layers = []
            for _ in range(n_independent):
                layers.append(nn.Sequential(
                    nn.Linear(n_d, n_d),
                    nn.BatchNorm1d(n_d),
                    nn.ReLU()
                ))
            self.step_specific.append(nn.Sequential(*layers))

    def forward(self, x, step_idx):
        """
        Forward pass for a given decision step.

        Parameters
        ----------
        x : torch.Tensor, shape (batch_size, input_dim)
            Input to the feature transformer.
        step_idx : int
            Index of the current decision step (used to pick the independent block).

        Returns
        -------
        torch.Tensor, shape (batch_size, n_d)
            Transformed feature output after shared and step-specific layers.
        """
        out = x
        for layer in self.shared:
            out = layer(out)
        out = self.step_specific[step_idx](out)
        return out
