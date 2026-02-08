import seaborn as sns
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Constants:
    k: float = 1.38e-23   # Boltzmann constant (J/K)
    q: float = 1.602176634e-19
    T_r: float = 300
    n: float = 1.2
    k_mu: float = 1.5
    k_vt: float = 1e-3 # Typical value for threshold voltage temperature dependence (in V/K)

    @property
    def vt_r(self):
        return self.k * self.T_r / self.q

    @property
    def vt_nom(self):
        return self.k * self.T / self.q
    
class NoisePool:
    def __init__(self):
        self.ideal_Ids = [0.020000, 0.05327272727272728, 0.08654545454545455, 0.11981818181818182, 0.15309090909090908, 0.18636363636363637, 0.21963636363636363, 0.2529090909090909, 0.2861818181818182, 0.3194545454545455, 0.3527272727272728, 0.386000]
        self.cor_thr_v = [110, 75, 55, 51, 43, 35, 23, 18, 14, 10, 7, 0]
        self.varying_params = ['W', 'L', 'mu_n', 'Cox', 'Vth', 'T']
        self.consts = Constants()
        self.T = 300        # Temperature in Kelvin
        self.vt_r = self.consts.k * self.consts.T_r / self.consts.q  # Thermal voltage at room temperature
        self.vt_nom = self.consts.k * self.T / self.consts.q  # Thermal voltage
        self.Mt1 = 243/300
        self.Mt2 = 398/300

        self.pool_of_weight_dist = {}
        for weight_index in range (len(self.ideal_Ids)):
            Id_nom = self.ideal_Ids[weight_index]  # Nominal Id value (in Amps)
            Vth_nom = 0.15 + (self.cor_thr_v[weight_index] * 1e-3)  # Nominal threshold voltage (V)
            all_Id_values, max_shifts = self.compute_Id_variations(Id_nom, Vth_nom, num_samples=1000)
            self.pool_of_weight_dist[Id_nom] = all_Id_values

    def get_Ideal_Ids(self):
        return self.ideal_Ids
    def calculate_new_Id_table(self, parameter, M, Id_nom, Vth_nom):
        if parameter == 'W':
            return M * Id_nom  # ID' = M * ID
        elif parameter == 'L':
            return (1 / M) * Id_nom  # ID' = (1/M) * ID
        elif parameter == 'mu_n':
            return M * Id_nom  # ID' = M * ID
        elif parameter == 'Cox':
            return M * Id_nom  # ID' = M * ID
        elif parameter == 'Vth':
            return Id_nom * np.exp(-((M - 1) * Vth_nom) / (self.consts.n * self.vt_nom))  # ID' = ID * exp(-((M - 1) * Vth) / (n * vt))
        elif parameter == 'T':  # Temperature variation case
                T = M * self.consts.T_r  # M represents the ratio of T/T_r
                vt_T = self.consts.k * T / self.consts.q  # Thermal voltage at new temperature T
                return Id_nom * (self.consts.T_r / T) ** (self.consts.k_mu + 2) * np.exp((self.consts.k_vt * (T - self.consts.T_r)) / (self.consts.n * vt_T))  # ID formula with temperature variation  
        else:
            raise ValueError("Invalid parameter name")

    def compute_Id_range_table(self, parameter, M_min, M_max, Id_nom, Vth_nom, num_samples=1000, distribution="uniform"):
        if distribution == "uniform":
            M_values = np.random.uniform(M_min, M_max, num_samples)
        elif distribution == "normal":
            mean_M = (M_max + M_min) / 2
            std_M = (M_max - M_min) / 4  # Roughly within ±2σ range
            M_values = np.random.normal(mean_M, std_M, num_samples)
            M_values = np.clip(M_values, M_min, M_max)  # Ensure values remain in range
        else:
            raise ValueError("Invalid distribution type")

        Id_values = np.array([self.calculate_new_Id_table(parameter, M, Id_nom, Vth_nom) for M in M_values])

        return M_values, Id_values

    def compute_Id_variations(self, Id_nom, Vth_nom, num_samples=1000):
        all_Id_values = []
        max_shifts = {}

        for param in self.varying_params:
            if param == "T":
                M_values = np.linspace(self.Mt1, self.Mt2, num_samples)  # Temperature from -30°C to 125°C (converted to Kelvin)
            elif param == "Vth":
                M_values = np.linspace(0.9, 1.1, num_samples)  # Range from 1 to 1.3
            else:
                M_values = np.linspace(0.9, 1.1, num_samples)  # Range from 0.9 to 1.1

            Id_values = np.array([self.calculate_new_Id_table(param, M, Id_nom, Vth_nom) for M in M_values])

            # Find the value with the greatest shift from Id_nom
            max_shift_idx = np.argmax(np.abs(Id_values - Id_nom))
            max_shifts[param] = (M_values[max_shift_idx], Id_values[max_shift_idx])

            # Add all values (except the max shift) to the combined list
            all_Id_values.extend(np.delete(Id_values, max_shift_idx))

        return all_Id_values, max_shifts

    def sample_from_pool(
        self, pool, mode="uniform", tail_frac=0.1, p_static=0.0, original_weight=None
    ):
        """
        Sample a noise value from the pool based on mode, with a probability `p_static` to return the original weight unchanged.

        Parameters:
            pool (array-like): The distribution of values to sample from.
            mode (str): Sampling mode - "uniform", "tail_high", "tail_low", "mean", or "normal".
            tail_frac (float): Fraction of data to use for tail sampling.
            p_static (float): Probability of returning `original_weight` without noise.
            original_weight (float): The original unperturbed weight (required if p_static > 0).
        """
        if p_static > 0.0:
            assert (
                original_weight is not None
            ), "original_weight must be provided when using p_static > 0"
        if np.random.rand() < p_static:
            return original_weight

        pool = np.sort(pool)
        mu = np.mean(pool)
        sigma = np.std(pool)

        if mode == "uniform":
            return np.random.choice(pool)

        elif mode == "tail_high":
            cutoff = int((1 - tail_frac) * len(pool))
            return np.random.choice(pool[cutoff:])

        elif mode == "tail_low":
            cutoff = int(tail_frac * len(pool))
            return np.random.choice(pool[:cutoff])

        elif mode == "mean":
            return mu

        elif mode == "normal":
            val = np.random.normal(loc=mu, scale=sigma)
            return np.clip(val, pool[0], pool[-1])  # clip if needed

        elif mode == "near_mean":
            lower_bound = mu - tail_frac * sigma
            upper_bound = mu + tail_frac * sigma
            close_to_mean = pool[(pool >= lower_bound) & (pool <= upper_bound)]
            if len(close_to_mean) == 0:
                return mu  # fallback if no points in range
            return np.random.choice(close_to_mean)

        elif mode == "sigma_low":
            upper_bound = mu
            lower_bound = mu - tail_frac * sigma
            low_region = pool[(pool >= lower_bound) & (pool <= upper_bound)]
            if len(low_region) == 0:
                return mu  # fallback
            return np.random.choice(low_region)

        elif mode == "sigma_high":
            lower_bound = mu
            upper_bound = mu + tail_frac * sigma
            high_region = pool[(pool >= lower_bound) & (pool <= upper_bound)]
            if len(high_region) == 0:
                return mu  # fallback
            return np.random.choice(high_region)
        else:
            raise ValueError(f"Unsupported sampling mode: {mode}")
        

def get_noise_schedule(epoch):
    """
    Returns (noise_type, tail_frac) for given epoch,
    gradually increasing noise difficulty.

    Covers:
    - near_mean
    - sigma_low
    - sigma_high
    - tail_low
    - tail_high
    - mixed
    """

    if epoch <= 6:
        return "near_mean", 0.5  # ±0.5 std range

    elif epoch <= 8:
        return "near_mean", 1.0  # ±1 std range

    elif epoch <= 10:
        return "sigma_low", 1.0  # Focus on negative noise (low tail)

    elif epoch <= 12:
        return "sigma_high", 1.0  # Focus on positive noise (high tail)

    elif epoch <= 14:
        return "tail_low", 0.005  # Very sharp low-end noise

    elif epoch <= 16:
        return "tail_high", 0.005  # Very sharp high-end noise

    elif epoch <= 18:
        return "tail_low", 0.01  # Moderate low-end extremes

    elif epoch <= 19:
        return "tail_high", 0.01  # Moderate high-end extremes
    else:
        return "mixed", 0.5

        