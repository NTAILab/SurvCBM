import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sksurv.nonparametric import kaplan_meier_estimator
from torch.nn.functional import softmax
from typing import List, Literal

class ExpActivation(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return torch.exp(x)

class Cox(torch.nn.Module):
    def __init__(self, device: torch.device, f_proba: bool = False,
                 bias: bool = True, base: Literal['km', 'breslow'] = 'km'):
        super().__init__()
        self.cox_model = None
        self.f_proba = f_proba
        self.bias = bias
        self.device = device
        self.base_type = base
        self.base_func = None
        assert not(f_proba == False and base == 'breslow'), "Breslow estimator is available only with probabilities"
        
    def _lazy_init(self, X: torch.Tensor):
        self.force_init(X.shape[1])

    # C and d must be sorted according to t for breslow
    def force_init(self, inp_shape: int, C: np.ndarray, 
                   t: np.ndarray, d: np.ndarray):
        self.cox_model = torch.nn.Sequential(
            torch.nn.Linear(inp_shape, 1, bias=self.bias),
            ExpActivation()
        ).to(self.device)
        if self.base_type == 'km':
            time, s = kaplan_meier_estimator(d.astype(bool), t)
            if time.shape[0] != t.shape[0]:
                # t is sorted
                t_mask = (t[1:] - t[:-1]) < 1e-8
                t_idx = np.zeros_like(t, dtype=int)
                t_idx[1:] = ~t_mask
                idxes = np.cumsum(t_idx, dtype=int)
                S = np.empty_like(t)
                S = s[idxes]
                s = S
            s[s < 1e-13] = 1e-13
            S_km_log = torch.log(torch.tensor(s, 
                            dtype=torch.get_default_dtype(),
                            device=self.device))
            self.base_func = lambda: -S_km_log
        elif self.base_type == 'breslow':
            # args_s = np.argsort(t).ravel()
            C_oh_t = torch.tensor(self._to_oh(C), device='cpu', dtype=torch.get_default_dtype())
            dataset = TensorDataset(C_oh_t)
            self._C_dl = DataLoader(dataset, 1024, shuffle=False)
            self._d = torch.tensor(d, device=self.device)
            self.base_func = self._calc_breslow_hazard
        else:
            raise NotImplementedError('Unknown baseline function type')

    def _to_oh(self, C) -> np.ndarray:
        cards = C.max(axis=0) + 1
        C_oh_list = []
        for i, c in enumerate(cards):
            E = np.eye(c)
            C_oh_list.append(np.take_along_axis(E, C[:, i, None].astype(np.intp), axis=0))
        return np.concatenate(C_oh_list, axis=1)

    def _calc_breslow_hazard(self):
        cox_est_list = []
        with torch.no_grad():
            for c, in self._C_dl:
                cox_est_list.append(self.cox_model(c.to(self.device))[:, 0])
        cox_est = torch.cat(cox_est_list, dim=0)
        cox_cum_sum = torch.cumsum(cox_est, dim=0)
        cox_cum_sum = cox_est - cox_cum_sum + cox_cum_sum[-1] # reverse
        hazards = self._d / cox_cum_sum
        breslow = torch.cumsum(hazards, dim=0)
        return breslow
        
    def _internal_fwd(self, x: torch.Tensor):
        cox = self.cox_model(x)
        surv_func = torch.exp(-self.base_func()[None, :] * cox)
        surv_func = surv_func.masked_scatter(surv_func < 1e-7, 1e-7 + torch.zeros_like(surv_func))
        surv_steps = surv_func[:, :-1] - surv_func[:, 1:]
        first_step = 1 - surv_func[:, 0]
        surv_steps = torch.concat((first_step[:, None], surv_steps), dim=-1)
        sum = torch.sum(surv_steps, dim=-1, keepdim=True).broadcast_to(surv_steps.shape).clone()
        bad_idx = sum < 1e-7
        sum[bad_idx] = 1
        surv_steps = surv_steps / sum
        surv_steps[bad_idx] = 0
        return surv_func, surv_steps

    def _process_input(self, c: List[torch.Tensor]) -> torch.Tensor:
        if not self.f_proba:
            return torch.cat(c, dim=-1)
        probas_list = []
        for cur_c in c:
            sm = softmax(cur_c, dim=-1)
            probas_list.append(sm)
        return torch.cat(probas_list, dim=-1)
        
        
    def forward(self, c: List[torch.Tensor]) -> torch.Tensor:
        assert self.base_func is not None, "Use _force_init first"
        c_all = self._process_input(c)
        if self.cox_model is None:
            self._lazy_init(c_all)
        return self._internal_fwd(c_all)

    def _calc_cox_likelihood(self, c: List[torch.Tensor], t: torch.Tensor, d: torch.Tensor):
        dot_prod = lambda x: torch.sum(self.cox_model[0].weight * x, dim=-1)
        t_sorted, sort_args = torch.sort(t, descending=True)
        c_all = self._process_input(c)
        log_theta_cum_sum = torch.log(torch.cumsum(torch.exp(dot_prod(c_all[sort_args])), dim=0))
        uncens_idx = torch.argwhere(d == 1).ravel()
        t_unc = t[uncens_idx]
        sum_idx = torch.clamp(torch.searchsorted(-t_sorted, -t_unc, side='right') - 1, min=0)
        likelihood = torch.sum(dot_prod(c_all[uncens_idx]) - torch.take_along_dim(log_theta_cum_sum, sum_idx))
        return likelihood

class BaselineCox(Cox):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self.S_km is not None, "Use _init_km first"
        if self.cox_model is None:
            self._lazy_init(x)
        return self._internal_fwd(x)

