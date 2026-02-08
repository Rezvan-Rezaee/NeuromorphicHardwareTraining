from dataclasses import dataclass

from pyparsing import Any

@dataclass(frozen=True)
class modelConstants:
    def __new__(cls, *args, **kwargs):
        raise TypeError("CFG is static; do not instantiate")
    
    A1: float = 240e-3
    A2: float = 0.0
    leakage_vol: float = 0.0

    amp_noise: float = 0.0         # formerly amp_noise_
    in_vol_noise_pr: float = 0.0   # std for multiplicative noise
    drop_probability: float = 0.0

    training_noise_ctt: bool = True
    inference_noise_ctt: bool = False
    population_code: bool = True
    pretrained_weights_used: bool = False
    inference_noise_ctt: bool = False
    training_noise_ctt: bool = True
    noise_hardware_ctt: bool = True
    generate_processed_data: bool = False
    general_noise: bool = True
    retention_noise: bool = True #always increases current
    quantization_noise: bool = True
    std_ret_low: float = 0.04
    std_ret_high: float = 0.025

    Threshold_voltage: float = 0.3
    leakage_vol: float = .03

    Vdd: float = 0.8

    min_weight: float = 0.02
    max_weight: float = 0.386
    if general_noise:
        std_ctt = 0.2
    else:
        std_ctt = 0

    num_w_levels: int = 12
    image_size: int = 10
    num_steps: int = 16

    if population_code:
        pop: float = 1
    else:
        pop: float = 1/2

    num_classes: int = 2
    num_inputs: int = image_size ** 2
    num_outputs: int = int(num_classes*pop)

    data_path='/data'

class noiseRangeConfig:
    """The values in this class describe noise ranges. 
    These values are used by the digLIF neuron model to determine how much noise to inject during training and inference.
    """
    def __new__(cls, *args, **kwargs):
        raise TypeError("CFG is static; do not instantiate")

    std_ctt = modelConstants.std_ctt
    amp_noise = modelConstants.amp_noise
    in_vol_noise_pr = modelConstants.in_vol_noise_pr

    def set_noise_range(self, range_percent: float) -> None:
        noiseRangeConfig.std_ctt = modelConstants.std_ctt * range_percent
        noiseRangeConfig.amp_noise = modelConstants.amp_noise * range_percent
        noiseRangeConfig.in_vol_noise_pr = modelConstants.in_vol_noise_pr * range_percent

    



