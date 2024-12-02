import torch
from torch.nn.functional import cross_entropy
from typing import List, Literal

class NNKernel(torch.nn.Module):
    INTERNAL_DIM = 32
    
    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.sparse_nn = None
        self.kernel_nn = None

    def _lazy_init(self, c_proba, c_bg):
        full_in_dim = sum([c.shape[-1] for c in c_proba])
        self.sparse_nn = torch.nn.Sequential(
                torch.nn.Linear(full_in_dim, self.INTERNAL_DIM),
                torch.nn.Tanh(),
                torch.nn.Linear(self.INTERNAL_DIM, self.INTERNAL_DIM),
                torch.nn.Tanh(),
            ).to(self.device)
        self.kernel_nn = torch.nn.Sequential(
            torch.nn.Linear(self.INTERNAL_DIM, self.INTERNAL_DIM),
            torch.nn.Tanh(),
            torch.nn.Linear(self.INTERNAL_DIM, self.INTERNAL_DIM),
            torch.nn.Tanh(),
            torch.nn.Linear(self.INTERNAL_DIM, 1),
            torch.nn.Softplus(),
        ).to(self.device)

    def forward(self, c_bg, c_proba):
        if self.kernel_nn is None:
            self._lazy_init(c_proba, c_bg)
        result = []
        c_bg_e = c_bg[None, ...].expand(c_proba[0].shape[0], -1, -1)
        full_proba = []
        full_labels = []
        for i, c_tensor in enumerate(c_proba):
            proba = torch.softmax(c_tensor, dim=-1)[:, None, ...].repeat(1, c_bg.shape[0], 1)
            full_proba.append(proba)
            one_hot_t_0 = torch.zeros_like(proba)
            one_hot_t_0.scatter_(-1, c_bg_e[:, :, i, None], torch.ones_like(proba))
            full_labels.append(one_hot_t_0)
        x_1 = self.sparse_nn(torch.cat(full_proba, dim=-1))
        x_2 = self.sparse_nn(torch.cat(full_labels, dim=-1))
        result.append(
            self.kernel_nn(torch.abs(x_1 - x_2))
        )
        weights = torch.sum(torch.stack(result, dim=-2), dim=-2)
        sum = torch.sum(weights, dim=1, keepdim=True).broadcast_to(weights.shape).clone()
        bad_idx = sum < 1e-13
        sum[bad_idx] = 1
        norm_weights = weights / sum
        norm_weights[bad_idx] = 0
        return norm_weights


class GaussKernel(torch.nn.Module):
    def __init__(self, device: torch.device, 
                 metric: Literal['l2', 'cross_entropy'] = 'l2',
                 norm_axis: int=-2):
        super().__init__()
        # self.bandwidth = torch.nn.parameter.Parameter(
        #                     torch.tensor([1.0],
        #                     dtype=torch.get_default_dtype(), device=device),
        #                     requires_grad=True)
        self.dim = norm_axis
        self.metric = getattr(self, f'calc_{metric}_metric_')
        self.bandwidth = torch.nn.parameter.Parameter(
                            torch.tensor([1.0],
                            dtype=torch.get_default_dtype(), device=device),
                            requires_grad=True)
        # self.device = device
        
    def calc_l2_metric_(self, c_in: torch.Tensor,
                    c_p: List[torch.Tensor]) -> torch.Tensor:
        # l2 metric
        c_in_e = c_in[None, ...].expand(c_p[0].shape[0], -1, -1)
        full_proba = []
        full_labels = []
        for i, c_tensor in enumerate(c_p):
            proba = torch.softmax(c_tensor, dim=-1)[:, None, ...].repeat(1, c_in.shape[0], 1)
            full_proba.append(proba)
            one_hot_1 = torch.ones_like(proba)
            one_hot_0 = torch.zeros_like(proba)
            one_hot_0.scatter_(-1, c_in_e[:, :, i, None], one_hot_1)
            full_labels.append(one_hot_0)
        cat_proba = torch.cat(full_proba, dim=-1)
        cat_labels = torch.cat(full_labels, dim=-1)
        return torch.sum((cat_proba - cat_labels) ** 2, dim=-1, keepdim=True)

    def calc_cross_entropy_metric_(self, c_in: torch.Tensor,
                    c_p: List[torch.Tensor]) -> torch.Tensor:
        result = []
        C_in = c_in[None, ...].expand(c_p[0].shape[0], -1, -1)
        for i, c in enumerate(c_p):
            c_rp = c[:, None, :].expand(-1, c_in.shape[0], -1).reshape(-1, c.shape[-1])
            ce = cross_entropy(c_rp, C_in[..., i].ravel(), reduction='none')
            result.append(ce.reshape(c_p[0].shape[0], c_in.shape[0]))
        return torch.sum(torch.stack(result, dim=-1), dim=-1, keepdim=True)
    
    def forward(self, c_in: List[torch.Tensor], 
                c_p: torch.Tensor) -> torch.Tensor:
        metric = self.metric(c_in, c_p)
        bandwidth = torch.clamp(self.bandwidth, min=0.1, max=10)
        weights = torch.exp(-metric / bandwidth)
        sum = torch.sum(weights, dim=self.dim, keepdim=True).broadcast_to(weights.shape).clone()
        bad_idx = sum < 1e-13
        sum[bad_idx] = 1
        norm_weights = weights / sum
        norm_weights[bad_idx] = 0
        return norm_weights # (x_n_1, x_n_2, ..., 1)
        
class Beran(torch.nn.Module):
    def __init__(self, device: torch.device, 
                 metric: Literal['l2', 'cross_entropy', 'nn'] = 'l2') -> None:
        super().__init__()
        self.device = device
        if metric == 'nn':
            self.kernel = NNKernel(device)
        else:
            self.kernel = GaussKernel(device, metric)

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
