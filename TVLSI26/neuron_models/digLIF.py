from __future__ import annotations
import torch
from typing import Any, Optional
import snntorch as snn  # type: ignore
from configs.config import modelConstants, noiseRangeConfig
import torch.nn as nn

# Quantized weights for inference
class Square(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight)
        return nn.functional.linear(input, weight)

    @staticmethod
    def backward(ctx, grad_out):
        input, weight = ctx.saved_tensors
        return grad_out @ weight, (input.t() @ grad_out).swapaxes(0, 1)


class digLIF(snn.Leaky):
    """This model implements a digital LIF neuron with noise, based on the neuron hardware described in the paper [Hardware-Aware Offline Training of CTT-Based Neuromorphic Hardware]. 
    The noise can be turned on or off using the `noise_on` argument in the constructor. 
    When noise is on, it affects both the amplitude of the input and the input voltage, based on the configuration parameters defined in `noiseRangeConfig`.

    Args:
        noise_on (bool, optional): completely turns on/off noise (from all sources in the neuron). Defaults to False.
    """

    def __init__(self, noise_on: bool = False, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.noise_on = bool(noise_on)   # Master switch: if False, ALL noise injections below are skipped.
        self.cfg = modelConstants()         # Configuration container holding constants and noise magnitudes + mode flags.
        self.nVin: Optional[torch.Tensor] = None        # Stores the last sampled input-voltage noise multiplier for debugging/analysis.

    def _base_state_function(self, input_: torch.Tensor) -> torch.Tensor:
        noisy_A1 = self.cfg.A1          # Start from nominal A1 (gain/slope term used to scale input_ into voltage [based on hardware implementation]).

        if self.noise_on:
            eps = torch.randn(1, device=input_.device, dtype=input_.dtype)  # Gaussian(0,1) sample used to perturb A1; one scalar per call.
            if self.cfg.training_noise_ctt:
                noisy_A1 = self.cfg.A1 * (1 + noiseRangeConfig.amp_noise * eps)
            elif self.cfg.inference_noise_ctt:
                with torch.no_grad():
                    noisy_A1 = self.cfg.A1 * (1 + noiseRangeConfig.amp_noise * eps) # Same A1 perturbation rule during inference mode. Disabled gradient calculation during inference.

        in_voltage = (noisy_A1 * input_) + self.cfg.A2  # Convert scaled input to voltage, adding offset A2.

        if self.noise_on:
            self.nVin = torch.normal(
                torch.ones_like(in_voltage), noiseRangeConfig.in_vol_noise_pr
            )
            if self.cfg.training_noise_ctt:
                in_voltage *= self.nVin
            if self.cfg.inference_noise_ctt:
                with torch.no_grad():
                    in_voltage *= self.nVin

        base_fn = in_voltage + self.mem - self.cfg.leakage_vol
        return torch.clamp(base_fn, min=0)

    def _base_state_function_hidden(self, input_: torch.Tensor) -> torch.Tensor:
        if self.noise_on:
            eps = torch.randn(1, device=input_.device)
            if self.cfg.training_noise_ctt:
                noisy_A1 = self.cfg.A1 * (1 + noiseRangeConfig.amp_noise * eps)
            if self.cfg.inference_noise_ctt:
                with torch.no_grad():
                    noisy_A1 = self.cfg.A1 * (1 + noiseRangeConfig.amp_noise * eps)
        else:
            noisy_A1 = self.cfg.A1

        in_voltage = (noisy_A1 * input_) + self.cfg.A2

        if self.noise_on:
            self.nVin = torch.normal(
                torch.ones_like(in_voltage), noiseRangeConfig.in_vol_noise_pr
            )
            if self.cfg.training_noise_ctt:
                in_voltage *= self.nVin
            if self.cfg.inference_noise_ctt:
                with torch.no_grad():
                    in_voltage *= self.nVin

        base_fn = in_voltage + self.mem - self.cfg.leakage_vol
        return torch.clamp(base_fn, min=0)
