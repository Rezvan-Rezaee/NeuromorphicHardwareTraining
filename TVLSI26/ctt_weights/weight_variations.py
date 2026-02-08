
import torch
import torch.nn as nn
import torch.nn.functional as F
from TVLSI26.configs.config import modelConstants

def maskW(w):
    w = torch.clamp(w, min=modelConstants.min_weight, max=modelConstants.max_weight)
    step = (modelConstants.max_weight - modelConstants.min_weight) / (modelConstants.num_w_levels - 1)
    return ((w - modelConstants.min_weight) / step).round() * step + modelConstants.min_weight

def add_retention_noise(model_weights, std_ret_high=modelConstants.std_ret_high, std_ret_low=modelConstants.std_ret_low, threshold = 0.06):
    """
    Adds noise to trained weights based on a threshold and clamps the weights within a given range.
    
    Args:
        model_weights (torch.Tensor): The tensor of trained weights.
        std_ret_high (float): The standard deviation for noise applied to weights >= threshold.
        std_ret_low (float): The standard deviation for noise applied to weights < threshold.
        threshold (float): The threshold to separate high and low noise application.
        min_weight (float): The minimum value to clamp weights to.
        max_weight (float): The maximum value to clamp weights to.
    
    Returns:
        torch.Tensor: The modified weights after noise addition and clamping.
    """
    # Create a mask for weights below the threshold
    mask_low = model_weights < threshold

    # Generate noise for weights below and above the threshold
    noise_low = torch.abs(torch.normal(0, std_ret_low, size=model_weights.size()))
    noise_high = torch.abs(torch.normal(0, std_ret_high, size=model_weights.size()))
    
    # Select the appropriate noise based on the mask
    ret_noise = torch.where(mask_low, noise_low, noise_high)
    
    return model_weights + ret_noise

def apply_quant_noise(model_weights, low0 = 0.24, high0 = 0.3):
    """
    Applies noise to the trained weights based on specified ranges, keeping them part of the computational graph.
    
    Args:
        model_weights (torch.Tensor): The trained weights to which noise will be applied.
        low0 (float): Lower bound for the first range of noise application.
        high0 (float): Upper bound for the first range of noise application.    
    Returns:
        torch.Tensor: The updated weights with noise applied, part of the computational graph.
    """
    # First range: [low0, high0]
    mask0 = (model_weights > low0) & (model_weights < high0)
    random_values0 = low0 + (high0 - low0) * torch.abs(torch.randn_like(model_weights))
    random_values0 = torch.where(mask0, random_values0, torch.zeros_like(model_weights))
    
    # Second range: [high0, max_weight + 0.01]
    low1, high1 = high0, modelConstants.max_weight + 0.01
    mask1 = (model_weights > low1) & (model_weights < high1)
    random_values1 = low1 + (high1 - low1) * torch.abs(torch.randn_like(model_weights))
    random_values1 = torch.where(mask1, random_values1, torch.zeros_like(model_weights))
    
    return model_weights + random_values0 + random_values1

class WeightDropoutLinear(nn.Module):
    """Defined for testing dropout on weights. Not currently used in the main codebase, but can be used for future experiments.
    """
    def __init__(self, num_inputs, num_outputs, p=modelConstants.drop_probability, bias=False):
        super(WeightDropoutLinear, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.p = p  # Probability of weight dropout
        self.linear = nn.Linear(num_inputs, num_outputs, bias=bias)

    def forward(self, x):
        if self.training:
            # Create a dropout mask for the weights
            weight_mask = torch.bernoulli((1 - self.p) * torch.ones_like(self.linear.weight))
            weight_mask = weight_mask.to(self.linear.weight.device)  # Ensure it's on the correct device
            # Apply the mask to the weights
            dropped_weights = self.linear.weight * weight_mask
        else:
            # Use the full weights during evaluation
            dropped_weights = self.linear.weight

        # Perform the linear transformation with the (possibly dropped) weights
        return F.linear(x, dropped_weights, self.linear.bias)