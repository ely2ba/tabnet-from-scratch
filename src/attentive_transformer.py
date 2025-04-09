import torch
import torch.nn as nn
from src.sparsemax import Sparsemax
from src.glu import GatedLinearUnit

class AttentiveTransformer(nn.Module):
    """
    Attentive Transformer module used in TabNet for soft feature selection.
    It outputs a sparse attention mask using Sparsemax, conditioned on prior masks.

    Args:
        input_dim (int): Dimensionality of input features.
        output_dim (int): Dimensionality of the output mask (should match input_dim).
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.glu = GatedLinearUnit(input_dim, output_dim)
        self.sparsemax = Sparsemax(dim=-1)

    def forward(self, x, prior):
        """
        Forward pass.

        Args:
            x (Tensor): Input of shape (batch_size, input_dim)
            prior (Tensor): Prior mask of shape (batch_size, input_dim), values in [0, 1]

        Returns:
            Tensor: Sparse attention mask of shape (batch_size, input_dim)
        """
        # Element-wise multiplication with prior
        x = x * prior

        # Project + GLU + Sparsemax
        x = self.glu(x)
        mask = self.sparsemax(x)

        return mask

