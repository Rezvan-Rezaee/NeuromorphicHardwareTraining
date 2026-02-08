import torch
import torch.nn as nn
from TVLSI26.configs.config import modelConstants, noiseRangeConfig
from TVLSI26.ctt_weights.weight_variations import WeightDropoutLinear, apply_quant_noise, add_retention_noise
from TVLSI26.neuron_models.digLIF import digLIF, Square

class populationEncodedBinaryClassifier(nn.Module):
    def __init__(self, num_inputs=modelConstants.num_inputs):
        super().__init__()
        self.num_inputs = num_inputs

        self.fc1 = WeightDropoutLinear(self.num_inputs, modelConstants.num_outputs, bias=False)
        self.lif1 = digLIF(beta=1, reset_mechanism="zero")

        self.initialize_weights()

    def initialize_weights(
        self,
    ):
        nn.init.normal_(
            self.fc1.linear.weight,
            mean=(modelConstants.max_weight - modelConstants.min_weight) / 2,
            std=(modelConstants.max_weight - modelConstants.min_weight) / 8,
        )
        with torch.no_grad():
            self.fc1.linear.weight.clamp_(modelConstants.min_weight, modelConstants.max_weight)

    def initialize_lif_thresholds(self, device, V_th = modelConstants.Threshold_voltage):
        self.lif1.threshold.data = torch.tensor(V_th)
        self.lif1.threshold.data = self.lif1.threshold.data.to(device)
        
    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        self.nW1 = torch.normal(torch.zeros_like(self.fc1.linear.weight), noiseRangeConfig.std_ctt)

        if self.training and modelConstants.training_noise_ctt:
            noisy_weights = self.fc1.linear.weight
            if modelConstants.retention_noise:
                noisy_weights = add_retention_noise(
                    self.fc1.linear.weight, modelConstants.std_ret_high, modelConstants.std_ret_low
                )
            if modelConstants.quantization_noise:
                noisy_weights = apply_quant_noise(noisy_weights)
        else:
            noisy_weights = self.fc1.linear.weight

        # Record the final layer
        spk_rec = []
        mem_rec = []

        for step in range(x.shape[0]):
            spk0 = x[step].flatten(1)
            cur1 = Square.apply(x[step].flatten(1), noisy_weights)

            with torch.no_grad():
                noise1 = nn.functional.linear(spk0, self.nW1)

            if self.training and modelConstants.general_noise:
                cur1 = cur1 + noise1
            spk1, mem1 = self.lif1.forward(cur1, mem1)

            spk_rec.append(spk1)
            mem_rec.append(mem1)

        return torch.stack(spk_rec, dim=0), torch.stack(mem_rec, dim=0)

# The following functions are used for population coding and prediction checking in the populationEncodedBinaryClassifier model.
# both are copied and lightly modified from "snntorch" library source code.
def _prediction_check(spk_out):
    device = spk_out.device

    num_steps = spk_out.size(0)
    num_outputs = spk_out.size(-1)

    return device, num_steps, num_outputs

def _population_code(spk_out, num_classes, num_outputs):
    """Count up spikes sequentially from output classes."""
    if not num_classes:
        raise Exception(
            "``num_classes`` must be specified if ``population_code=True``."
        )
    if num_outputs % num_classes:
        raise Exception(
            f"``num_outputs {num_outputs} must be a factor of num_classes "
            f"{num_classes}."
        )

    device = spk_out.device
    pop_code = torch.zeros(tuple([spk_out.size(1)] + [num_classes])).to(device)
    for idx in range(num_classes):
        pop_code[:, idx] = (
            spk_out[
                :,
                :,
                int(num_outputs * idx / num_classes) : int(
                    num_outputs * (idx + 1) / num_classes
                ),
            ]
            .sum(-1)
            .sum(0)
        )
    return pop_code