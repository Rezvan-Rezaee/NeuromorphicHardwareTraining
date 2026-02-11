import torch
import torch.nn as nn

# The timing dynamics of this surrogate function are explained in "softLIF_behavior.txt".
class SoftLIFRate(nn.Module):
    def __init__(self, theta=0.0, gain=1.0, tau_rc=0.02, tau_ref=0.002, eps=1e-4):
        super().__init__()
        self.theta = nn.Parameter(torch.tensor(float(theta)))
        self.gain  = nn.Parameter(torch.tensor(float(gain)))
        self.tau_rc  = float(tau_rc)
        self.tau_ref = float(tau_ref)
        self.eps = float(eps)

    def forward(self, z):
        # Force the critical math to FP32 to avoid AMP/FP16 inf->nan gradients
        z32 = z.float()
        u = (self.gain.float()) * (z32 - self.theta.float())

        # J = 1 + softplus(u) >= 1
        Jm1 = F.softplus(u)                       # this is (J-1)
        Jm1 = torch.clamp(Jm1, min=self.eps)      # prevents divide-by-0 / inf grads

        # log(1 + 1/(J-1))
        denom = self.tau_ref + self.tau_rc * torch.log1p(1.0 / Jm1)
        rate  = 1.0 / denom

        r = rate * self.tau_ref                   # normalize to [0,1]
        r = torch.clamp(r, 0.0, 1.0)

        return r.to(dtype=z.dtype)                # restore original dtype
