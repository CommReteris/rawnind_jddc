"""
Generalized Divisive Normalization (GDN) implementation.

This module provides a PyTorch implementation of Generalized Divisive Normalization,
a normalization technique commonly used in image compression neural networks.
GDN has been shown to be effective for removing statistical dependencies in image data,
making it useful for tasks like image compression and enhancement.

The implementation is adapted from https://github.com/jorge-pessoa/pytorch-gdn under MIT license.

References:
    Ballé, J., Laparra, V., & Simoncelli, E. P. (2016). Density modeling of images using a
    generalized normalization transformation. arXiv preprint arXiv:1511.06281.
"""

import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.autograd import Function


class LowerBound(Function):
    """Custom autograd function that enforces a lower bound on tensor values.
    
    This function applies a lower bound constraint to input tensors during the forward pass,
    while maintaining proper gradient flow during backpropagation. It's used in the GDN
    implementation to ensure stability of the normalization parameters.
    """
    
    @staticmethod
    def forward(ctx, inputs, bound):
        """Apply a lower bound to input tensor values.
        
        Args:
            ctx: Context object for storing information for backward pass
            inputs: Input tensor that will be constrained
            bound: Scalar lower bound value to enforce
            
        Returns:
            Tensor with all values guaranteed to be >= bound
        """
        b = torch.ones(inputs.size(), device=inputs.device) * bound
        b = b.to(inputs.device)
        ctx.save_for_backward(inputs, b)
        return torch.max(inputs, b)

    @staticmethod
    def backward(ctx, grad_output):
        """Custom gradient computation for the lower bound operation.
        
        Implements straight-through estimator logic for the lower bound operation.
        Gradients flow through unmodified when:
        1. The input value was already above the bound (pass_through_1)
        2. The gradient is negative, pushing away from the constraint (pass_through_2)
        
        Args:
            ctx: Context with saved tensors from forward pass
            grad_output: Gradient flowing from subsequent operations
            
        Returns:
            Tuple containing (gradient for inputs, None for bound parameter)
        """
        inputs, b = ctx.saved_tensors

        pass_through_1 = inputs >= b
        pass_through_2 = grad_output < 0

        pass_through = pass_through_1 | pass_through_2
        return pass_through.type(grad_output.dtype) * grad_output, None


class GDN(nn.Module):
    """Generalized divisive normalization layer.
    
    This layer implements the GDN normalization operation, a form of divisive normalization
    that has been shown to be effective for image compression neural networks.
    
    The GDN operation is defined as:
        y[i] = x[i] / sqrt(beta[i] + sum_j(gamma[j, i] * x[j]^2))
    
    Where:
    - x[i] is the input feature map
    - beta and gamma are trainable parameters
    - beta provides a stabilizing term to prevent division by zero
    - gamma controls the normalization strength across channels
    
    The inverse GDN (if inverse=True) multiplies rather than divides:
        y[i] = x[i] * sqrt(beta[i] + sum_j(gamma[j, i] * x[j]^2))
    
    References:
        Ballé, J., Laparra, V., & Simoncelli, E. P. (2016).
        Density modeling of images using a generalized normalization transformation.
    """

    def __init__(
        self,
        ch,
        device,
        inverse=False,
        beta_min=1e-6,
        gamma_init=0.1,
        reparam_offset=2**-18,
    ):
        """Initialize the GDN layer.
        
        Args:
            ch: Number of channels in the input tensor
            device: Device to create tensors on (e.g., 'cuda:0', 'cpu')
            inverse: If True, implements inverse GDN (multiplication instead of division)
            beta_min: Minimum value for beta parameter to ensure stability
            gamma_init: Initial value for gamma parameters (diagonal entries)
            reparam_offset: Small constant for reparameterization to ensure stability
        """
        super(GDN, self).__init__()
        self.inverse = inverse
        self.beta_min = beta_min
        self.gamma_init = gamma_init
        self.reparam_offset = torch.tensor([reparam_offset], device=device)

        self.build(ch, torch.device(device))

    def build(self, ch, device):
        """Build the GDN layer parameters.
        
        This method initializes the beta and gamma parameters required for GDN.
        The parameters are initialized in a way that ensures stability during training
        by using appropriate bounds and reparameterization.
        
        Args:
            ch: Number of channels in the input tensor
            device: Device to create tensors on
        """
        self.pedestal = self.reparam_offset**2
        self.beta_bound = (self.beta_min + self.reparam_offset**2) ** 0.5
        self.gamma_bound = self.reparam_offset

        # Create beta param
        beta = torch.sqrt(torch.ones(ch, device=device) + self.pedestal)
        self.beta = nn.Parameter(beta)

        # Create gamma param
        eye = torch.eye(ch, device=device)
        g = self.gamma_init * eye
        g = g + self.pedestal
        gamma = torch.sqrt(g)
        self.gamma = nn.Parameter(gamma)

    def forward(self, inputs):
        """Apply GDN normalization to input tensor.
        
        The forward pass handles both standard 4D tensors (batch, channels, height, width)
        and 5D tensors, by unfolding 5D inputs and refolding them back after processing.
        
        Args:
            inputs: Input tensor with shape [batch_size, channels, height, width] or
                   [batch_size, channels, depth, width, height]
                   
        Returns:
            Normalized tensor with the same shape as the input
        """
        unfold = False
        if inputs.dim() == 5:
            unfold = True
            bs, ch, d, w, h = inputs.size()
            inputs = inputs.view(bs, ch, d * w, h)

        _, ch, _, _ = inputs.size()

        # Beta bound and reparam
        beta = LowerBound.apply(self.beta, self.beta_bound)
        beta = beta**2 - self.pedestal

        # Gamma bound and reparam
        gamma = LowerBound.apply(self.gamma, self.gamma_bound)
        gamma = gamma**2 - self.pedestal
        gamma = gamma.view(ch, ch, 1, 1)

        # Norm pool calc
        norm_ = nn.functional.conv2d(inputs**2, gamma, beta)
        norm_ = torch.sqrt(norm_)

        # Apply norm
        if self.inverse:
            outputs = inputs * norm_
        else:
            outputs = inputs / norm_

        if unfold:
            outputs = outputs.view(bs, ch, d, w, h)
        return outputs
