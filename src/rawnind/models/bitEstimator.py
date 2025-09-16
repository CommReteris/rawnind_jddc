"""Entropy model for neural image compression.

This module implements parametric models for estimating probability distributions
of latent representations in neural compression systems. These models are critical
for rate-distortion optimization during training and for entropy coding during 
actual compression.

Key components:
- MultiHeadBitEstimator: Main class that estimates cumulative distribution functions (CDFs)
  using a sequence of non-linear transformations
- MultiHeadBitparm: Building block that performs individual transformation steps
  in the CDF estimation process

The entropy model transforms continuous-valued latents into discrete symbols and
estimates their probability distribution, which determines their bitrate cost
according to information theory principles. This implementation uses a multi-head
approach where different distribution parameters can be learned for different channels
or contexts.

This implementation is derived from the ManyPriors architecture, which allows for
more flexible distribution modeling compared to simpler parametric approaches.

Technical note:
The models estimate CDFs (cumulative distribution functions) rather than PDFs
(probability density functions) because:
1. CDFs are needed for range coding during actual compression
2. CDF gradients are more numerically stable during training
3. CDF-based formulations avoid potential issues with zero probabilities
"""

import torch
from torch.nn import functional as F


class MultiHeadBitEstimator(torch.nn.Module):
    """Estimate cumulative distribution functions for entropy coding.
    
    This class implements a non-linear transform that maps input values to
    their estimated cumulative distribution functions (CDFs). It uses a sequence
    of four MultiHeadBitparm transforms to create a flexible, expressive CDF model
    that can capture complex distributions.
    
    The multi-head approach allows the model to learn different distribution
    parameters for different channels or contexts. This increases model capacity
    and enables more accurate probability estimation, which directly impacts
    compression performance.
    
    The model uses a composition of non-linear functions:
    1. Three intermediate transforms (f1, f2, f3) with tanh activations
    2. A final transform (f4) with sigmoid activation to ensure output in [0,1]
    
    This approach is more flexible than using simple parametric distributions
    (like Gaussian or Laplacian) and can adapt to the complex, multi-modal
    distributions often encountered in image compression latent spaces.
    """

    def __init__(
        self,
        channel: int,
        nb_head: int,
        shape=("g", "bs", "ch", "h", "w"),
        bitparm_init_mode="normal",
        bitparm_init_range=0.01,
        **kwargs,
    ):
        """Initialize the multi-head bit estimator.
        
        Creates a sequence of four MultiHeadBitparm transforms that together
        form a flexible cumulative distribution function estimator.
        
        Args:
            channel: Number of channels in the latent representation
            nb_head: Number of distribution heads (allows learning different
                    distribution parameters for different contexts)
            shape: Tensor dimension ordering, controls how parameters are shaped.
                  Options are:
                  - ("g", "bs", "ch", "h", "w"): Used in Balle2017ManyPriors_ImageCompressor
                  - ("bs", "ch", "g", "h", "w"): Alternative ordering
            bitparm_init_mode: Weight initialization method. Options are:
                              - "normal": Normal distribution initialization
                              - "xavier_uniform": Xavier uniform initialization
            bitparm_init_range: Parameter range for initialization (standard deviation
                               for normal, gain for xavier_uniform)
            **kwargs: Additional arguments (unused, for compatibility)
            
        Notes:
            - The first three transforms (f1, f2, f3) use non-final configurations
              that include tanh activations
            - The last transform (f4) uses final=True to apply sigmoid activation,
              ensuring output is a valid CDF in [0,1]
            - All transforms share the same channel count, head count, and initialization
        """
        super(MultiHeadBitEstimator, self).__init__()
        self.f1 = MultiHeadBitparm(
            channel,
            nb_head=nb_head,
            shape=shape,
            bitparm_init_mode=bitparm_init_mode,
            bitparm_init_range=bitparm_init_range,
        )
        self.f2 = MultiHeadBitparm(
            channel,
            nb_head=nb_head,
            shape=shape,
            bitparm_init_mode=bitparm_init_mode,
            bitparm_init_range=bitparm_init_range,
        )
        self.f3 = MultiHeadBitparm(
            channel,
            nb_head=nb_head,
            shape=shape,
            bitparm_init_mode=bitparm_init_mode,
            bitparm_init_range=bitparm_init_range,
        )
        self.f4 = MultiHeadBitparm(
            channel,
            final=True,
            nb_head=nb_head,
            shape=shape,
            bitparm_init_mode=bitparm_init_mode,
            bitparm_init_range=bitparm_init_range,
        )

    #        if bs_first:
    #            self.prep_input_fun = lambda x: x.unsqueeze(0)
    #        else:
    #            self.prep_input_fun = lambda x: x

    def forward(self, x):
        """Transform input values to their cumulative distribution values.
        
        Applies a sequence of four non-linear transforms to map input values
        to their estimated cumulative probabilities.
        
        Args:
            x: Input tensor containing latent values to be transformed
               Expected shape depends on the 'shape' parameter from initialization
            
        Returns:
            Tensor containing estimated cumulative distribution values in [0,1]
            for each input value, maintaining the input tensor's shape
            
        Notes:
            - Output values are in range [0,1] due to the final sigmoid activation
            - These CDF values are used for entropy coding and rate estimation
            - The transformation is differentiable, allowing gradient-based training
        """
        # x = self.prep_input_fun(x)
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        return self.f4(x)


class MultiHeadBitparm(torch.nn.Module):
    """Building block for the multi-head bit estimator.
    
    This class implements a single parametric transformation step in the
    cumulative distribution function estimation process. It applies an
    affine transformation followed by non-linear activation.
    
    The transformation has the form:
    - For non-final blocks: x + tanh(x) * tanh(a) where x = h*x + b
    - For final blocks: sigmoid(h*x + b)
    
    Each block learns three parameter tensors:
    - h: Multiplicative factor (passed through softplus for positivity)
    - b: Additive bias
    - a: Scaling factor for the tanh non-linearity (only in non-final blocks)
    
    The multi-head design allows the model to learn different parameter values
    for different channels or groups, enabling more flexible distribution modeling.
    """

    def __init__(
        self,
        channel,
        nb_head,
        final=False,
        shape=("g", "bs", "ch", "h", "w"),
        bitparm_init_mode="normal",
        bitparm_init_range=0.01,
    ):
        """Initialize a multi-head bit parameter transform block.
        
        Args:
            channel: Number of channels in the latent representation
            nb_head: Number of distribution heads (parameter sets)
            final: Whether this is the final transformation block.
                  If True, uses sigmoid activation instead of tanh,
                  and doesn't create the 'a' parameter
            shape: Tensor dimension ordering, determines how parameter tensors are shaped.
                  Options are:
                  - ("g", "bs", "ch", "h", "w"): Used in Balle2017ManyPriors_ImageCompressor
                  - ("bs", "ch", "g", "h", "w"): Alternative ordering
            bitparm_init_mode: Parameter initialization method. Options:
                              - "normal": Normal distribution with mean 0
                              - "xavier_uniform": Xavier uniform initialization
            bitparm_init_range: Parameter range for initialization (standard deviation
                               for normal, gain for xavier_uniform)
                               
        Notes:
            - Parameter shapes are carefully designed to allow broadcasting during forward pass
            - The parameter initialization is critical for training stability
            - Final blocks have different behavior from non-final blocks (sigmoid vs. tanh)
        """
        super(MultiHeadBitparm, self).__init__()
        self.final = final
        if shape == (
            "g",
            "bs",
            "ch",
            "h",
            "w",
        ):  # used in Balle2017ManyPriors_ImageCompressor
            params_shape = (nb_head, 1, channel, 1, 1)
        elif shape == ("bs", "ch", "g", "h", "w"):
            params_shape = (1, channel, nb_head, 1, 1)
        if bitparm_init_mode == "normal":
            init_fun = torch.nn.init.normal_
            init_params = 0, bitparm_init_range
        elif bitparm_init_mode == "xavier_uniform":
            init_fun = torch.nn.init.xavier_uniform_
            init_params = [bitparm_init_range]
        else:
            raise NotImplementedError(bitparm_init_mode)
        self.h = torch.nn.Parameter(
            init_fun(torch.empty(nb_head, channel).view(params_shape), *init_params)
        )
        self.b = torch.nn.Parameter(
            init_fun(torch.empty(nb_head, channel).view(params_shape), *init_params)
        )
        if not final:
            self.a = torch.nn.Parameter(
                init_fun(torch.empty(nb_head, channel).view(params_shape), *init_params)
            )
        else:
            self.a = None

    def forward(self, x):
        """Apply the transformation to input tensor.
        
        Applies either:
        - For final blocks: sigmoid(x * softplus(h) + b)
        - For non-final blocks: x + tanh(x) * tanh(a) where x = x * softplus(h) + b
        
        The softplus on h ensures positivity of the multiplicative factor,
        which is important for monotonicity of the cumulative distribution function.
        
        Args:
            x: Input tensor to transform
            
        Returns:
            Transformed tensor with the same shape as the input
            
        Notes:
            - Final blocks ensure output is in [0,1] via sigmoid activation
            - Non-final blocks use a residual-like connection with tanh non-linearities
            - The parameters h, b, and a are automatically broadcast to match input dimensions
            - Using softplus for h ensures the transformation is invertible
        """
        if self.final:
            return torch.sigmoid(x * F.softplus(self.h) + self.b)
        else:
            x = x * F.softplus(self.h) + self.b
            return x + torch.tanh(x) * torch.tanh(self.a)
