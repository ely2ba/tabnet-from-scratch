# src/tabnet_step.py

import torch
import torch.nn as nn
from src.feature_transformer import FeatureTransformer
from src.attentive_transformer import AttentiveTransformer

class TabNetStep(nn.Module):
    """
    A single decision step in the TabNet architecture.

    Each step:
    - Applies an attention mask to the input features.
    - Passes masked features through a shared + step-specific FeatureTransformer.
    - Outputs a decision contribution (decision_out).
    - Updates the prior to encourage exploration of new features.
    """

    def __init__(self, input_dim, n_d, output_dim, shared_transformer=None):
        """
        Parameters
        ----------
        input_dim : int
            Dimensionality of input features.
        n_d : int
            Dimension of decision output and intermediate features.
        output_dim : int
            Final output dimension per decision step (usually n_d).
        shared_transformer : nn.ModuleList, optional
            Predefined shared layers across steps (for weight sharing).
        """
        super().__init__()

        # Feature transformer: shared + step-specific block
        self.feature_transformer = FeatureTransformer(
            input_dim=input_dim,
            n_d=n_d,
            n_steps=1,  # only 1 step-specific block for this step
            n_shared=len(shared_transformer) if shared_transformer else 0,
            n_independent=2
        )

        # Replace shared layers if provided (external shared weights)
        if shared_transformer:
            self.feature_transformer.shared = shared_transformer

        # Attentive transformer: generates sparse feature selection mask
        self.attentive_transformer = AttentiveTransformer(
           input_dim=input_dim,
            output_dim=input_dim
        )
        # Linear projection to decision output
        self.decision_layer = nn.Linear(n_d, output_dim)

    def forward(self, x, prior):
        """
        Forward pass for one TabNet decision step.

        Args:
            x: input features (batch_size, input_dim)
            prior: feature mask prior (batch_size, input_dim)

        Returns:
            decision_out: Output of this decision step (batch_size, output_dim)
            next_feat: Features to carry forward (batch_size, feature_dim)
            updated_prior: Updated feature prior (batch_size, input_dim)
            mask: Attention mask used for this step (batch_size, input_dim)
        """
        # Compute sparse attention mask
        mask = self.attentive_transformer(x, prior)  # (B, input_dim)
        masked_x = x * mask                          # (B, input_dim)

        # Run through feature transformer (shared + step-specific)
        feat = self.feature_transformer(masked_x, step_idx=0)

        # Split into decision and features for next step
        decision_out = self.decision_layer(feat)  # (B, output_dim)
        next_feat = feat                          # (B, feature_dim)

        # Update prior for next step
        updated_prior = prior * (1 - mask)

        return decision_out, next_feat, updated_prior, mask
