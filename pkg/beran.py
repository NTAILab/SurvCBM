import torch
from typing import List


class SimpleGauss(torch.nn.Module):
    def __init__(self, device: torch.device, norm_axis: int=-2):
        super().__init__()
        self.bandwidth = torch.nn.parameter.Parameter(
                            torch.tensor([1.0],
                            dtype=torch.get_default_dtype(), device=device),
                            requires_grad=True)
        self.dim = norm_axis
        
    def calc_metric_(self, c_in: torch.Tensor,
                    c_p: List[torch.Tensor]) -> torch.Tensor:
        # l2 metric
        result = []
        c_in_e = c_in[None, ...].expand(c_p[0].shape[0], -1, -1)
        for i, c_tensor in enumerate(c_p):
            proba = torch.softmax(c_tensor, dim=-1)[:, None, ...].repeat(1, c_in.shape[0], 1)
            one_hot_t = -torch.ones_like(proba)
            proba.scatter_reduce_(-1, c_in_e[:, :, i, None], one_hot_t, reduce='sum')
            result.append(
                torch.sum(proba ** 2, dim=-1)
            )
        return torch.sum(torch.stack(result, dim=-1), dim=-1, keepdim=True)
    
    def forward(self, c_in: List[torch.Tensor], 
                c_p: torch.Tensor) -> torch.Tensor:
        metric = self.calc_metric_(c_in, c_p)
        bandwidth = torch.clamp(self.bandwidth, min=0.1, max=10)
        weights = torch.exp(-metric / bandwidth)
        sum = torch.sum(weights, dim=self.dim, keepdim=True).broadcast_to(weights.shape).clone()
        bad_idx = sum < 1e-13
        sum[bad_idx] = 1
        norm_weights = weights / sum
        norm_weights[bad_idx] = 0
        return norm_weights # (x_n_1, x_n_2, ..., 1)
        
class Beran(torch.nn.Module):
    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.kernel = SimpleGauss(device)

    def forward(self, c_in, delta_in, c_p):
        # n = c_in.shape[0]
        # x_p_repeat = c_p[:, None, :].expand(-1, n, -1)
        W = self.kernel(c_in, c_p)[..., 0] # (batch, n)
        w_cumsum = torch.cumsum(W, dim=1)
        shifted_w_cumsum = w_cumsum - W
        ones = torch.ones_like(shifted_w_cumsum)
        bad_idx = torch.isclose(shifted_w_cumsum, ones) | torch.isclose(w_cumsum, ones)
        shifted_w_cumsum[bad_idx] = 0.0
        w_cumsum[bad_idx] = 0.0

        xi = torch.log(1.0 - shifted_w_cumsum)
        xi -= torch.log(1.0 - w_cumsum)

        filtered_xi = delta_in * xi
        hazards = torch.cumsum(filtered_xi, dim=1)
        surv_func = torch.exp(-hazards)
        surv_steps = surv_func[:, :-1] - surv_func[:, 1:]
        first_step = 1 - surv_func[:, 0]
        surv_steps = torch.concat((first_step[:, None], surv_steps), dim=-1)
        sum = torch.sum(surv_steps, dim=-1, keepdim=True).broadcast_to(surv_steps.shape).clone()
        bad_idx = sum < 1e-13
        sum[bad_idx] = 1
        surv_steps = surv_steps / sum
        surv_steps[bad_idx] = 0
        return surv_func, surv_steps
