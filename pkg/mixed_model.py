# the intermidiate approach between SurvBottleNeck 
# and a survival model without concepts

from copy import deepcopy
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import List, Tuple, Optional, Callable, Dict, Literal
from sksurv.metrics import concordance_index_censored
from .beran import Beran
from .cox import BaselineCox
from time import time
from tqdm import tqdm
from numba import njit, bool_
from copy import deepcopy
import torch.optim as optim

@njit
def dataset_generator(X, C, T, D, background_size, target_size, batch_n):
    dim = X.shape[1:]
    N = background_size
    if target_size > C.shape[0] - N:
        target_size = C.shape[0] - N
    x_bg = np.empty((batch_n, N) + dim, dtype=np.float64)
    t_bg = np.empty((batch_n, N), dtype=np.float64)
    d_bg = np.empty((batch_n, N), dtype=np.int64)
    x_target = np.empty((batch_n, target_size) + X.shape[1:], dtype=np.float64)
    c_target = np.empty((batch_n, target_size, C.shape[1]), dtype=np.int64)
    d_target = np.empty((batch_n, target_size), dtype=np.int64)
    t_target = np.empty((batch_n, target_size), dtype=np.float64)
    idx = np.arange(C.shape[0])
    for i in range(batch_n):
        back_idx = np.random.choice(idx, N, False)
        target_mask = np.ones(C.shape[0]).astype(bool_)
        target_mask[back_idx] = 0
        target_idx = np.argwhere(target_mask)[:, 0]
        if target_size < C.shape[0] - N:
            target_idx = np.random.choice(target_idx, target_size, False)
        x_bg[i, ...] = X[back_idx]
        t_bg[i, :] = T[back_idx]
        d_bg[i, :] = D[back_idx]
        x_target[i, ...] = X[target_idx]
        c_target[i, ...] = C[target_idx]
        t_target[i, :] = T[target_idx]
        d_target[i, :] = D[target_idx]
    return x_bg, t_bg, d_bg, x_target, c_target, t_target, d_target


class ConceptEasyClf(torch.nn.Module):
    def __init__(self, c_cards: np.ndarray | List[int],
                 core_out_dim: int,
                 device: torch.device):
        super().__init__()
        if type(c_cards) is np.ndarray:
            assert c_cards.ndim == 1
        self.nn_list_ = []
        for i, card in enumerate(c_cards):
            sub_model = torch.nn.Sequential(
                torch.nn.Linear(core_out_dim, core_out_dim // 2).to(device),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(core_out_dim // 2, core_out_dim // 2).to(device),
                torch.nn.LeakyReLU(),
                torch.nn.Linear(core_out_dim // 2, card).to(device)
            )
            self.register_module(f'sub_nn_{i}', sub_model)
            self.nn_list_.append(sub_model)
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        result = [] # logits
        for nn in self.nn_list_:
            result.append(nn(x))
        return result


class SurvRCM(torch.nn.Module):
    def __init__(self, nn_model: torch.nn.Module,
                 nn_model_out_dim: int,
                 device: torch.device, 
                 surv_loss: Literal['likelihood', 'c_index'] = 'likelihood',
                 surv_model: Literal['beran', 'cox'] = 'beran',
                 cox_bias: bool = True,
                 alpha: float = 0.5,
                 sigma_temp: float = 1,
                 batch_num: int = 64, # number of tasks per epoch
                 target_size: int = 2048, # len of a task, gradient is cumulative
                 epochs: int = 100,
                 train_bg_part: float = 0.6,
                 patience: int = 10, 
                 optimizer: str | None = None,
                 optimizer_kw: Dict | None = None):
        super().__init__()
        self.nn_model = nn_model.to(device)
        self.alpha = alpha
        self.surv_model = surv_model
        if surv_model == 'beran':
            self.beran = Beran(device, 'x_l2')
        elif surv_model == 'cox':
            self.cox = BaselineCox(device, cox_bias)
        self.device = torch.device(device)
        self.batch_num = batch_num
        self.target_size = target_size
        self.epochs = epochs
        self.patience = patience
        self.train_bg_part = train_bg_part
        self.nn_model_out_dim = nn_model_out_dim
        self.optimizer = optimizer
        self.optimizer_kw = optimizer_kw
        self.best_val_loss = None
        self.sigma_temp = sigma_temp
        self.surv_loss = surv_loss
        
        # nvmlInit()
        
    def np2torch(self, arr: np.ndarray, 
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[str] = None) -> torch.Tensor:
        if dtype is None:
            dtype = torch.get_default_dtype()
        if device is None:
            device = self.device
        return torch.from_numpy(arr).type(dtype).to(device)
    
    def _get_optimizer(self) -> torch.optim.Optimizer:
        if self.optimizer is None:
            return torch.optim.AdamW(self.parameters(), lr=1e-3)
        else:
            optim_cls = getattr(optim, self.optimizer)
            return optim_cls(self.parameters(), **self.optimizer_kw)

    def _set_background(self, X: torch.Tensor, T: torch.Tensor, D: torch.Tensor):
        self.T_bg, sort_args = torch.sort(T)
        dataset = TensorDataset(X)
        data_loader = DataLoader(dataset, self.target_size, shuffle=False)
        z_list = []
        for cur_x, in data_loader:
            z_list.append(self.nn_model(cur_x.to(self.device)).detach())
        z = torch.cat(z_list, dim=0)
        self.Z_bg = z[sort_args]
        self.D_bg = D[sort_args]
        self.T_diff_int = self.T_bg[1:] - self.T_bg[:-1]
        if self.surv_model == 'cox':
            t, d = self.T_bg.detach().cpu().numpy(), self.D_bg.detach().cpu().numpy()
            self.cox._init_km(t, d)

    # val_set is (T, D, concept_labels)
    def fit(self, X: np.ndarray, y: np.recarray, c: np.ndarray, 
            val_set: Optional[Tuple[np.ndarray, np.recarray, np.ndarray]]=None,
            metrics_logger: Optional[Callable]=None) -> 'SurvRCM':
        sub_batch_len = 512
        t = y['time'].copy()
        d = y['cens'].copy()
        assert np.all(np.min(c, axis=0) == 0), "All labels must start with 0"
        c_cards = np.max(c, axis=0) + 1
        assert np.all(c_cards > 1), "All concepts must have at least 2 cats"
        self.concepts_model_ = ConceptEasyClf(c_cards, self.nn_model_out_dim,
                                              self.device)
        
        self.train()
        optimizer = self._get_optimizer()
        start_time = time()
        bg_size = int(self.train_bg_part * X.shape[0])
        cur_patience = 0
        self.best_val_loss = 0

        if val_set is not None:
            x_val = val_set[0].astype(np.float32)
        
        for e in range(1, self.epochs + 1):
            cum_surv_loss = 0
            cum_ce_loss = 0
            cum_loss = 0
            i = 0
            
            data = dataset_generator(X, c, t, d, bg_size, self.target_size, self.batch_num)

            X_back = self.np2torch(data[0], device='cpu')
            T_back = self.np2torch(data[1])
            D_back = self.np2torch(data[2], torch.int)
            X_target = self.np2torch(data[3], device='cpu')
            C_target = self.np2torch(data[4], torch.long)
            T_target = self.np2torch(data[5])
            D_target = self.np2torch(data[6], torch.int)
            dataset = TensorDataset(X_back, T_back, D_back, X_target, C_target, T_target, D_target)
            data_loader = DataLoader(dataset, 1, shuffle=False)

            prog_bar = tqdm(data_loader,
                            f'Epoch {e}', unit='task', ascii=True)

            for x_b, t_b, d_b, x_t, c_t, t_t, d_t in prog_bar:
                x_b.squeeze_(0)
                t_b.squeeze_(0)
                d_b.squeeze_(0)
                x_t.squeeze_(0)
                c_t.squeeze_(0)
                t_t.squeeze_(0)
                d_t.squeeze_(0)
                
                optimizer.zero_grad()
                self._set_background(x_b.to(self.device), t_b, d_b)
                
                target_ds = TensorDataset(x_t.to(self.device), c_t, t_t, d_t)
                target_loader = DataLoader(target_ds, sub_batch_len, False)
                
                for x_t_b, c_t_b, t_t_b, d_t_b in target_loader:
                    # optimizer.zero_grad()
                    z_t_b = self.nn_model(x_t_b)
                    c_pred = self.concepts_model_(z_t_b)
                    if self.surv_model == 'beran':
                        sf, pi = self.beran(self.Z_bg, self.D_bg, z_t_b)
                    elif self.surv_model == 'cox':
                        sf, pi = self.cox(z_t_b)
                    else:
                        raise NotImplementedError("Unknown survival model")
                    if self.surv_loss == 'likelihood':
                        surv_loss = self._calc_likelihood(sf, pi, t_t_b, d_t_b)
                    else:
                        surv_loss = self._calc_c_index(sf, t_t_b, d_t_b)
                    ce, ce_list = self._calc_cross_entropy(c_pred, c_t_b)
                    
                    loss = -self.alpha * surv_loss + (1 - self.alpha) * ce
                    loss.backward()
                    
                    # optimizer.step()

                    cum_surv_loss += surv_loss.item()
                    cum_ce_loss += np.asarray(ce_list)
                    cum_loss += loss.item()
                
                tl_len = len(target_loader)
                optimizer.step()
                
                i += 1
                epoch_metrics = {
                    'Loss': cum_loss / (i * tl_len),
                    'Surv': cum_surv_loss / (i * tl_len),
                    'CE': (cum_ce_loss / (i * tl_len)).tolist()
                }
                prog_bar.set_postfix(epoch_metrics)

            if metrics_logger is not None:
                log_metrics_kw = {}
                train_metrics = {
                    'loss/train': epoch_metrics['Loss'],
                    'surv/train': epoch_metrics['Surv']
                }
                for i, ce in enumerate(epoch_metrics['CE']):
                    train_metrics[f'cross_entropy_{i + 1}/train'] = ce
                log_metrics_kw['custom_dict'] = train_metrics
            if val_set is not None:
                cur_patience += 1
                # self._set_background(X_full_tens, T_full_tens, D_full_tens)
                E_T_v, c_v = self.predict(x_val)
                val_loss, *_ = concordance_index_censored(val_set[1]['cens'], val_set[1]['time'], -E_T_v)
                log_metrics_kw['custom_dict']['c_index/valid'] = val_loss
                if val_loss >= self.best_val_loss:
                    self.best_val_loss = val_loss
                    weights = deepcopy(self.state_dict())
                    cur_patience = 0
                print(f'Val C-index: {round(val_loss, 5)}, patience: {cur_patience}')
                if metrics_logger is not None:
                    log_metrics_kw['c_proba'] = c_v
                    log_metrics_kw['true_c_vals'] = val_set[2]
                if cur_patience >= self.patience:
                    print('Early stopping!')
                    self.load_state_dict(weights)
                    if metrics_logger is not None:
                        metrics_logger(**log_metrics_kw)
                    break    
            if metrics_logger is not None:
                metrics_logger(**log_metrics_kw)
        self._set_background(self.np2torch(X, device='cpu'), self.np2torch(t), self.np2torch(d, dtype=torch.int))
        time_elapsed = time() - start_time
        print('Training time:', round(time_elapsed, 1), 's.')
        self.eval()
        return self
                    
                    
    def _calc_cross_entropy(self, c_pred: List[torch.Tensor], labels: torch.Tensor) -> Tuple[torch.Tensor, List[float]]:
        loss = 0
        float_vals = []
        for i, pred in enumerate(c_pred):
            cur_loss = torch.nn.functional.cross_entropy(pred, labels[:, i])
            float_vals.append(round(cur_loss.item(), 1))
            loss += cur_loss
        return loss, float_vals
    
    def _calc_likelihood(self, sf: torch.Tensor, pi: torch.Tensor,
                         t: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        idx = torch.searchsorted(self.T_bg, t).clamp_max_(pi.shape[1] - 1)
        cens_mask = d == 0
        idx_cens = idx[cens_mask]
        idx_uncens = idx[~cens_mask]
        pi_vals = torch.take_along_dim(pi[~cens_mask], idx_uncens[:, None], dim=-1)
        sf_vals = torch.take_along_dim(sf[cens_mask], idx_cens[:, None], dim=-1)
        pi_vals[pi_vals < 1e-15] = 1
        likelihood = torch.sum(torch.log(pi_vals)) + torch.sum(torch.log(sf_vals))
        return likelihood

    def _calc_c_index(self, sf: torch.Tensor, t: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        T_calc = self._calc_exp_time(sf)
        T_diff = t[:, None] - t[None, :]
        T_mask = T_diff < 0
        sigma = torch.nn.functional.sigmoid(self.sigma_temp * (T_calc[None, :] - T_calc[:, None]))
        C = torch.sum(T_mask * sigma * d[:, None]) / torch.sum(T_mask * d[:, None])
        return C
            
    def _calc_exp_time(self, surv_func: torch.Tensor) -> torch.Tensor:
        integral = self.T_bg[None, 0] + \
            torch.sum(surv_func[:, :-1] * self.T_diff_int[None, :], dim=-1)
        return integral # (batch)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        z = self.nn_model(x)
        c = self.concepts_model_(z)
        if self.surv_model == 'beran':
            sf, pi = self.beran(self.Z_bg, self.D_bg, z)
        elif self.surv_model == 'cox':
            sf, pi = self.cox(z)
        E_T = self._calc_exp_time(sf)
        return E_T, c
        
    @torch.inference_mode()
    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        X = self.np2torch(x, device='cpu')
        dataset = TensorDataset(X)
        data_loader = DataLoader(dataset, 512, shuffle=False)
        all_val_preds = [[] for _ in range(len(self.concepts_model_.nn_list_))]
        all_e_t_preds = []
        for cur_x, in data_loader:
            E_T_t, c_t = self(cur_x.to(self.device))
            for k, c in enumerate(c_t):
                all_val_preds[k].append(c.cpu().numpy())
            all_e_t_preds.append(E_T_t.cpu().numpy())
        c = []
        for cur_tens in all_val_preds:
            c.append(np.concatenate(cur_tens, axis=0))
        E_T = np.concatenate(all_e_t_preds, axis=0)
        return E_T, c
    
    @torch.inference_mode()
    def score(self, x: np.ndarray, y: np.recarray) -> float:
        E_T, _ = self.predict(x)
        c_ind, *_ = concordance_index_censored(y['cens'], y['time'], -E_T)
        return c_ind
        