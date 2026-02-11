import numpy as np
import torch
import torch.nn as nn

# Encoded with the autograd function.
# The hardware-based relationships between parameters do not to be manually derived, but can be computed numerically by autograd.
class _SubthresholdIdFn(torch.autograd.Function):
    # Physical constants and device parameters (can be adjusted for different technologies)
    k: float = 1.38e-23   # Boltzmann constant (J/K)
    q: float = 1.602176634e-19 # Charge (C)
    T_r: float = 300      # Reference temperature (K)
    n: float = 1.5
    k_mu: float = 1.5  # Typical value for mobility temperature dependence (in T^(-k_mu))
    k_vt: float = 1e-3 # Typical value for threshold voltage temperature dependence (in V/K)
    Tr = 300
    _exp_1p8 = 1.8

    @staticmethod
    def thermal_voltage(T, k = k, q = q):
        return k * T / q

    @staticmethod
    def Id_subthreshold(W, L, mu, Cox, Vth, VGS, VDS, T, n = n, k = k, q = q, _exp_1p8 = _exp_1p8):
        Vt = k * T / q
        Is0 = (W / L) * mu * Cox * (Vt**2) * np.exp(_exp_1p8)
        return Is0 * torch.exp((VGS - Vth) / (n * Vt)) * (1 - torch.exp(-VDS / Vt))

    @staticmethod
    def forward(ctx, W, L, mu, Cox, Vth, VGS, VDS, T, n, k, q, _exp_1p8):
        ctx.save_for_backward(W, L, mu, Cox, Vth, VGS, VDS, T, n, k, q, _exp_1p8)
        with torch.no_grad():
            out = _SubthresholdIdFn.Id_subthreshold(W, L, mu, Cox, Vth, VGS, VDS, T, n, k, q, _exp_1p8)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        W, L, mu, Cox, Vth, VGS, VDS, T, n, k, q, _exp_1p8 = ctx.saved_tensors

        # Recompute Id with grads enabled
        kB_ = k.detach()  # constants: no grad needed
        qE_ = q.detach()
        e18_ = _exp_1p8.detach()
        n_   = n.detach()

        W_   = W.detach().requires_grad_(True)
        L_   = L.detach().requires_grad_(True)
        mu_  = mu.detach().requires_grad_(True)
        Cox_ = Cox.detach().requires_grad_(True)
        Vth_ = Vth.detach().requires_grad_(True)
        VGS_ = VGS.detach().requires_grad_(True)
        VDS_ = VDS.detach().requires_grad_(True)
        T_   = T.detach().requires_grad_(True)

        Id = _SubthresholdIdFn.Id_subthreshold(
            W_, L_, mu_, Cox_, Vth_, VGS_, VDS_, T_, n_, kB_, qE_, e18_
        )

        gW, gL, gmu, gCox, gVth, gVGS, gVDS, gT = torch.autograd.grad(
            outputs=Id,
            inputs=(W_, L_, mu_, Cox_, Vth_, VGS_, VDS_, T_),
            grad_outputs=grad_out,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )

        # No gradients for constants (k, q, n, exp_1p8)
        return (gW, gL, gmu, gCox, gVth, gVGS, gVDS, gT, None, None, None, None)


# This class implements the sensitivity-based noise model for subthreshold operation, where the noise in the drain current (Id) is computed based on the variations in the device parameters and their sensitivities (partial derivatives of Id with respect to each parameter).
# This class was used to model the noise in the networks trained for MNIST and CIFAR10, 100 datasets
class SubthresholdPVTNoise(nn.Module):
    def __init__(self):
        super().__init__()
        """
        params: dict with
            W, L, mu, Cox, Vth, VGS, VDS, T, n
        sigmas: dict with
            W, L, mu, Cox, Vth, VDS (normal std)
            dT (uniform half-range)
        """
        self.W   = 44e-9
        self.L   = 22e-9
        self.mu  = 0.03
        self.Cox = 0.03
        self.Vth = 0.35
        self.VGS = 0.25
        self.VDS = 0.8
        self.T0  = 310.0
        self.n   = 1.5
        self.dId_dW_value = 0.0
        self.dId_dL_value = 0.0
        self.dId_dmu_value = 0.0
        self.dId_dCox_value = 0.0
        self.dId_dVth_value = 0.0
        self.dId_dVDS_value = 0.0
        self.dId_dT_value = 0.0
        self.s = {
            "W":   0.10 * self.W,
            "L":   0.10 * self.L,
            "mu":  0.10 * self.mu,
            "Cox": 0.10 * self.Cox,
            "Vth": 0.10 * self.Vth,
            "VDS": 0.10 * self.VDS,
            "T_min": 248,
            "T_max": 398,
        }

    # The following methods compute the partial derivatives of Id with respect to each parameter, evaluated at the nominal values.
    # The formulas are derived from the expression for Id in the subthreshold region, using the chain rule of differentiation.
    def dId_dW(self):
        Vt = _SubthresholdIdFn.thermal_voltage(self.T0)
        return (self.mu * self.Cox * Vt**2 / self.L) * np.exp(1.8) * \
            np.exp((self.VGS - self.Vth) / (self.n * Vt)) * (1 - np.exp(-self.VDS / Vt))

    def dId_dmu(self):
        Vt = _SubthresholdIdFn.thermal_voltage(self.T0)
        return (self.W * self.Cox * Vt**2 / self.L) * np.exp(1.8) * \
            np.exp((self.VGS - self.Vth) / (self.n * Vt)) * (1 - np.exp(-self.VDS / Vt))

    def dId_dCox(self):
        Vt = _SubthresholdIdFn.thermal_voltage(self.T0)
        return (self.W * self.mu * Vt**2 / self.L) * np.exp(1.8) * \
            np.exp((self.VGS - self.Vth) / (self.n * Vt)) * (1 - np.exp(-self.VDS / Vt))

    def dId_dL(self):
        Vt = _SubthresholdIdFn.thermal_voltage(self.T0)
        return -(self.W * self.mu * self.Cox * Vt**2 / self.L**2) * np.exp(1.8) * \
                np.exp((self.VGS - self.Vth) / (self.n * Vt)) * (1 - np.exp(-self.VDS / Vt))
    
    def dId_dVDS(self):
        Vt = _SubthresholdIdFn.thermal_voltage(self.T0)
        Is0 = (self.W / self.L) * self.mu * self.Cox * (Vt**2) * np.exp(1.8)
        return Is0 * np.exp((self.VGS - self.Vth) / (self.n * Vt)) * (np.exp(-self.VDS / Vt) / Vt)
    
    def dId_dVth(self): 
        Id = _SubthresholdIdFn.Id_subthreshold(self.W, self.L, self.mu, self.Cox, self.Vth, self.VGS, self.VDS, self.T0, self.n)
        Vt = _SubthresholdIdFn.thermal_voltage(self.T0)
        return (-1 / (self.n * Vt)) * Id * np.exp((self.VGS - self.Vth) / (self.n * Vt)) * \
            (1 - np.exp(-self.VDS / Vt))

    def dId_dT(self):
        Vt = _SubthresholdIdFn.thermal_voltage(self.T0)
        Id = _SubthresholdIdFn.Id_subthreshold(self.W, self.L, self.mu, self.Cox, self.Vth, self.VGS, self.VDS, self.T0, self.n)

        term1 = _SubthresholdIdFn.k_mu - (( _SubthresholdIdFn.k_vt * self.T0) / (self.n * Vt)) * Id
        term2 = (1 - np.exp(-self.VDS / Vt)) * _SubthresholdIdFn.Tr**(-_SubthresholdIdFn.k_mu) * self.T0**(_SubthresholdIdFn.k_mu - 1)
        term3 = np.exp((self.VGS - self.Vth * _SubthresholdIdFn.Tr - _SubthresholdIdFn.k_vt * (self.T0-_SubthresholdIdFn.Tr)) / (self.n * Vt))
        return Id * (term1 + term2 + term3)

    def update_params(self):
        self.dId_dW_value = self.dId_dW()
        self.dId_dL_value = self.dId_dL()
        self.dId_dmu_value = self.dId_dmu()
        self.dId_dCox_value = self.dId_dCox()
        self.dId_dVth_value = self.dId_dVth()
        self.dId_dVDS_value = self.dId_dVDS()
        self.dId_dT_value = self.dId_dT()

    def forward(self, Id0):
        """
        Id0: torch.Tensor (any shape)
        returns: noisy Id tensor (same shape)
        """
        device, dtype = Id0.device, Id0.dtype
        self.update_params()

        dW   = torch.normal(0.0, self.s["W"],   size=Id0.shape, device=device, dtype=dtype)
        dL   = torch.normal(0.0, self.s["L"],   size=Id0.shape, device=device, dtype=dtype)
        dmu  = torch.normal(0.0, self.s["mu"],  size=Id0.shape, device=device, dtype=dtype)
        dCox = torch.normal(0.0, self.s["Cox"], size=Id0.shape, device=device, dtype=dtype)
        dVth = torch.normal(0.0, self.s["Vth"], size=Id0.shape, device=device, dtype=dtype)
        dVDS = torch.normal(0.0, self.s["VDS"], size=Id0.shape, device=device, dtype=dtype)
        dT   = torch.empty_like(Id0).uniform_(self.s["T_min"], self.s["T_max"])
        dT  = dT - self.T0


        dId = (
            self.dId_dW_value           * dW   +
            self.dId_dL_value           * dL   +
            self.dId_dmu_value          * dmu  +
            self.dId_dCox_value         * dCox +
            self.dId_dVth_value         * dVth +
            self.dId_dVDS_value * dVDS +
            self.dId_dT_value * dT
        )

        return Id0 + dId
    
# The calculated derivitives are based on the behvior of CTT devices.
# For aplicability to other devices, the autograd functionality of PyTorch can be used to compute the sensitivities (partial derivatives) 
# numerically, by defining the Id function in terms of the device parameters and using torch.autograd.grad to compute the gradients with 
# respect to each parameter.
class SubthresholdPVTNoise_general(nn.Module):
    """
    Linearized PVT noise injection using autograd sensitivities from
    the subthreshold Id equation.

    dId is first-order Taylor:
      dId ≈ Σ (∂Id/∂p)|nominal * dp
    """
    def __init__(self):
        super().__init__()

        # Nominal operating point (buffers, not learnable)
        self.register_buffer("W0",   torch.tensor(44e-9))
        self.register_buffer("L0",   torch.tensor(22e-9))
        self.register_buffer("mu0",  torch.tensor(0.03))
        self.register_buffer("Cox0", torch.tensor(0.03))
        self.register_buffer("Vth0", torch.tensor(0.35))
        self.register_buffer("VGS0", torch.tensor(0.25))
        self.register_buffer("VDS0", torch.tensor(0.8))
        self.register_buffer("T0",   torch.tensor(310.0))
        self.register_buffer("n0",   torch.tensor(1.5))

        # Noise levels (std for normal, half-range for uniform)
        self.s = {
            "W":   0.10,     # fraction of W0
            "L":   0.10,     # fraction of L0
            "mu":  0.10,     # fraction of mu0
            "Cox": 0.10,     # fraction of Cox0
            "Vth": 0.10,     # fraction of Vth0
            "VDS": 0.10,     # fraction of VDS0
            "T_min": 248.0,
            "T_max": 398.0,
        }

    def _scalar_params(self, device, dtype):
        # constants as tensors (kept inside class)
        n = torch.tensor(self.n0, device=device, dtype=dtype)
        k = torch.tensor(self._k, device=device, dtype=dtype)
        q = torch.tensor(self._q, device=device, dtype=dtype)
        exp_1p8 = torch.tensor(self._exp_1p8, device=device, dtype=dtype)

        # nominal params as scalars on device/dtype
        W   = self.W0.to(device=device, dtype=dtype).detach().requires_grad_(True)
        L   = self.L0.to(device=device, dtype=dtype).detach().requires_grad_(True)
        mu  = self.mu0.to(device=device, dtype=dtype).detach().requires_grad_(True)
        Cox = self.Cox0.to(device=device, dtype=dtype).detach().requires_grad_(True)
        Vth = self.Vth0.to(device=device, dtype=dtype).detach().requires_grad_(True)
        VGS = self.VGS0.to(device=device, dtype=dtype).detach().requires_grad_(True)
        VDS = self.VDS0.to(device=device, dtype=dtype).detach().requires_grad_(True)
        T   = self.T0.to(device=device, dtype=dtype).detach().requires_grad_(True)

        return k, q, exp_1p8, W, L, mu, Cox, Vth, VGS, VDS, T, n

    def forward(self, Id0: torch.Tensor) -> torch.Tensor:
        device, dtype = Id0.device, Id0.dtype

        # 1) Autograd sensitivities at nominal point
        k, q, exp_1p8, W, L, mu, Cox, Vth, VGS, VDS, T, n = self._scalar_params(device, dtype)

        Id_nom = self._SubthresholdIdFn.apply(W, L, mu, Cox, Vth, VGS, VDS, T, n, k, q, exp_1p8)

        dW, dL, dmu, dCox, dVth, _, dVDS, dT= torch.autograd.grad(
            outputs=Id_nom,
            inputs=(W, L, mu, Cox, Vth, VGS, VDS, T),
            grad_outputs=torch.ones_like(Id_nom),
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )

        # 2) Sample perturbations per element (match Id0 shape)
        dW_s   = torch.normal(0.0, (self.s["W"]   * self.W0).to(device, dtype),   size=Id0.shape, device=device, dtype=dtype)
        dL_s   = torch.normal(0.0, (self.s["L"]   * self.L0).to(device, dtype),   size=Id0.shape, device=device, dtype=dtype)
        dmu_s  = torch.normal(0.0, (self.s["mu"]  * self.mu0).to(device, dtype),  size=Id0.shape, device=device, dtype=dtype)
        dCox_s = torch.normal(0.0, (self.s["Cox"] * self.Cox0).to(device, dtype), size=Id0.shape, device=device, dtype=dtype)
        dVth_s = torch.normal(0.0, (self.s["Vth"] * self.Vth0).to(device, dtype), size=Id0.shape, device=device, dtype=dtype)
        dVDS_s = torch.normal(0.0, (self.s["VDS"] * self.VDS0).to(device, dtype), size=Id0.shape, device=device, dtype=dtype)

        T_u = torch.empty_like(Id0).uniform_(self.s["T_min"], self.s["T_max"])
        dT_s = T_u - self.T0.to(device=device, dtype=dtype)

        # 3) Linearized current perturbation (scalar sensitivities broadcast)
        dId = (
            dW   * dW_s +
            dL   * dL_s +
            dmu  * dmu_s +
            dCox * dCox_s +
            dVth * dVth_s +
            dVDS * dVDS_s +
            dT   * dT_s
        )

        return Id0 + dId
