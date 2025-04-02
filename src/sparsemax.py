import torch
import torch.nn as nn

class Sparsemax(nn.Module):
    """
    PyTorch implementation of the Sparsemax activation function from:
    
        "From Softmax to Sparsemax: A Sparse Model of Attention"
        (Martins & Astudillo, 2016)
    
    This activation produces a probability distribution along the specified
    dimension, but allows some outputs to become exactly zero (sparsity).
    
    Parameters
    ----------
    dim : int, optional (default=-1)
        The dimension along which to apply the Sparsemax function.
        
    Usage:
    ------
    >>> import torch
    >>> from sparsemax import Sparsemax
    >>> # Example with a small input tensor
    >>> logits = torch.tensor([[1.0, 2.0, 3.0],
    ...                        [1.0, -1.0, 0.0]], dtype=torch.float)
    >>> sparsemax = Sparsemax(dim=1)
    >>> probs = sparsemax(logits)
    >>> print(probs)
    """
    
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Sparsemax activation.
        
        Applies the Sparsemax transformation along the specified dimension.
        
        Parameters
        ----------
        logits : torch.Tensor
            Input tensor of any shape.
        
        Returns
        -------
        torch.Tensor
            Tensor of the same shape as `logits`, containing
            non-negative values that sum to 1 along `self.dim`.
            Some elements may be exactly zero.
        """
        # Sort logits along the specified dim in descending order
        # and keep track of the sorted values and original indices
        sorted_logits, sorted_indices = torch.sort(logits, dim=self.dim, descending=True)
        
        # Compute the cumulative sum of the sorted logits
        cumsum_sorted = torch.cumsum(sorted_logits, dim=self.dim)
        
        # Create an index (1-based) for each element along self.dim
        # e.g., if shape along self.dim is K, then we create [1, 2, ..., K]
        rhos = torch.arange(start=1, end=logits.size(self.dim) + 1, device=logits.device)
        # Reshape rhos for proper broadcasting along self.dim
        # The shape is [1, 1, ..., K, ..., 1] with K in the self.dim position
        dim_shape = [1] * logits.dim()
        dim_shape[self.dim] = -1
        rhos = rhos.view(dim_shape)
        
        # Compute the threshold function: t_j = (cumsum(z)_j - 1) / j
        # for j = 1..K (the size along self.dim)
        t = (cumsum_sorted - 1) / rhos
        
        # Compare sorted_logits to t to find the valid region (support)
        # support_ij = sorted_logits_ij > t_ij
        support = (sorted_logits > t)
        
        # The maximum number of elements (k) in the support is the sum of True
        # along self.dim (since we have a sorted sequence).
        # K is the largest index where sorted_logits_j > t_j.
        support_size = torch.sum(support, dim=self.dim, keepdim=True).clamp(min=1)
        
        # Gather the threshold for the k-th largest logit:
        # We pick the value of t at the (k)th position in each slice.
        # Because of 0-based indexing, we use (support_size - 1).
        # We'll gather from `t` using the final index in the support.
        range_indices = torch.arange(logits.size(self.dim), device=logits.device)
        range_shape = [1] * logits.dim()
        range_shape[self.dim] = -1
        range_indices = range_indices.view(range_shape)
        
        # Expand support_size to match shape
        # Then compute the gather index for each slice
        gather_index = (support_size - 1).expand_as(sorted_logits)
        gather_index = torch.where(gather_index < 0, torch.zeros_like(gather_index), gather_index)
        
        # We gather the threshold value (tau) for each "slice" along self.dim
        # The dimension we gather from is self.dim
        tau = torch.gather(t, self.dim, gather_index.long())
        
        # Scatter these tau values back to the original order of elements
        # Alternatively, we can compute the final output directly in sorted order,
        # then revert to the original order via gather. We'll do it in two steps:
        # Step 1: p_sorted = ReLU(sorted_logits - tau)
        p_sorted = torch.relu(sorted_logits - tau)
        
        # Step 2: revert to the original order
        # Create a tensor of zeros with the same shape as logits
        p = torch.zeros_like(logits)
        p.scatter_(self.dim, sorted_indices, p_sorted)
        
        return p

if __name__ == "__main__":
    # Minimal usage example
    logits = torch.tensor([[1.0, 2.0, 3.0],
                           [1.0, -1.0, 0.0]], dtype=torch.float)

    sparsemax = Sparsemax(dim=1)
    probs = sparsemax(logits)

    print("Input logits:")
    print(logits)

    print("\nSparsemax probabilities:")
    print(probs)

    print("\nCheck row sums (should be close to 1):")
    print(probs.sum(dim=1))
