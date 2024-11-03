import torch
import numpy as np
from sksurv.nonparametric import kaplan_meier_estimator
from typing import List

class ExpActivation(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return torch.exp(x)

class Cox(torch.nn.Module):
    def __init__(self, device: torch.device, bias: bool = True):
        super().__init__()
        self.cox_model = None
        self.S_km = None
        self.bias = bias
        self.device = device
        
    def _lazy_init(self, X: torch.Tensor):
        self.cox_model = torch.nn.Sequential(
            torch.nn.Linear(X.shape[1], 1, bias=self.bias),
            ExpActivation()
        ).to(self.device)
        
    def _init_km(self, t: np.ndarray, d: np.ndarray):
        time, s = kaplan_meier_estimator(d.astype(bool), t)
        s[s < 1e-13] = 1e-13
        self.S_km = torch.log(torch.tensor(s, 
                        dtype=torch.get_default_dtype(),
                        device=self.device))
        
    def forward(self, c: List[torch.Tensor]) -> torch.Tensor:
        assert self.S_km is not None, "Use _init_km first"
        c_all = torch.cat(c, axis=-1)
        if self.cox_model is None:
            self._lazy_init(c_all)
        cox = self.cox_model(c_all)
        surv_func = torch.exp(self.S_km[None, :] * cox)
        surv_steps = surv_func[:, :-1] - surv_func[:, 1:]
        first_step = 1 - surv_func[:, 0]
        surv_steps = torch.concat((first_step[:, None], surv_steps), dim=-1)
        sum = torch.sum(surv_steps, dim=-1, keepdim=True).broadcast_to(surv_steps.shape).clone()
        bad_idx = sum < 1e-13
        sum[bad_idx] = 1
        surv_steps = surv_steps / sum
        surv_steps[bad_idx] = 0
        return surv_func, surv_steps
