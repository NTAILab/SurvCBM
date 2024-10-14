import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import List, Tuple, Any, Optional
from .beran import Beran
from time import time
from tqdm import tqdm
from numba import njit, bool_
from copy import deepcopy

@njit
def dataset_generator(X, C, T, D, background_size, batch_n):
    N = background_size
    n = C.shape[0] - N
    c_bg = np.empty((batch_n, N, C.shape[1]), dtype=np.float64)
    t_bg = np.empty((batch_n, N), dtype=np.float64)
    d_bg = np.empty((batch_n, N), dtype=np.int64)
    x_target = np.empty((batch_n, n, X.shape[1]), dtype=np.float64)
    c_target = np.empty((batch_n, n, C.shape[1]), dtype=np.int64)
    d_target = np.empty((batch_n, n), dtype=np.int64)
    t_target = np.empty((batch_n, n), dtype=np.float64)
    idx = np.arange(C.shape[0])
    for i in range(batch_n):
        back_idx = np.random.choice(idx, N, False)
        target_mask = np.ones(C.shape[0]).astype(bool_)
        target_mask[back_idx] = 0
        target_idx = np.argwhere(target_mask)[:, 0]
        c_bg[i, ...] = C[back_idx]
        t_bg[i, :] = T[back_idx]
        d_bg[i, :] = D[back_idx]
        x_target[i, ...] = X[target_idx]
        c_target[i, ...] = C[target_idx]
        t_target[i, :] = T[target_idx]
        d_target[i, :] = D[target_idx]
    return c_bg, t_bg, d_bg, x_target, c_target, t_target, d_target


class ConceptsClf(torch.nn.Module):
    def __init__(self, c_cards: np.ndarray | List[int], 
                 nn_core: torch.nn.Module,
                 core_out_dim: int,
                 device: torch.device):
        super().__init__()
        if type(c_cards) is np.ndarray:
            assert c_cards.ndim == 1
        self.nn_list_ = []
        for i, card in enumerate(c_cards):
            sub_model = torch.nn.Sequential(
                deepcopy(nn_core),
                torch.nn.Linear(core_out_dim, card).to(device)
            )
            self.register_module(f'sub_nn_{i}', sub_model)
            self.nn_list_.append(
                sub_model
            )
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        result = [] # logits
        for nn in self.nn_list_:
            result.append(nn(x))
        return result


class SurvBN(torch.nn.Module):
    def __init__(self, nn_model: torch.nn.Module,
                 nn_model_out_dim: int,
                 device: torch.device, 
                 alpha: float = 0.5,
                 batch_num: int = 64,
                 epochs: int = 100,
                 lr_rate: float = 0.001,
                 train_bg_part: float = 0.6,
                 patience: int = 10):
        super().__init__()
        self.nn_model = nn_model.to(device)
        self.alpha = alpha
        self.beran = Beran(device)
        self.device = device
        self.batch_num = batch_num
        self.epochs = epochs
        self.lr_rate = lr_rate
        self.patience = patience
        self.train_bg_part = train_bg_part
        self.nn_model_out_dim = nn_model_out_dim
        
        
    def np2torch(self, arr: np.ndarray, 
                 dtype: Optional[torch.dtype] = None,
                 device: Optional[str] = None) -> torch.Tensor:
        if dtype is None:
            dtype = torch.get_default_dtype()
        if device is None:
            device = self.device
        return torch.from_numpy(arr).type(dtype).to(device)
    
    def _get_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(self.parameters(), lr=self.lr_rate)
    
    def _set_background(self, C: torch.Tensor, T: torch.Tensor, D: torch.Tensor):
        self.T_bg, sort_args = torch.sort(T)
        self.C_bg = C[sort_args]
        self.D_bg = D[sort_args]
        self.T_diff_int = self.T_bg[1:] - self.T_bg[:-1]
        
    def fit(self, X: np.ndarray, y: np.recarray, c: np.ndarray) -> 'SurvBN':
        sub_batch_len = 512
        
        t = y['time'].copy()
        d = y['cens'].copy()
        assert np.all(np.min(c, axis=0) == 0), "All labels must start with 0"
        c_cards = np.max(c, axis=0) + 1
        assert np.all(c_cards > 1), "All concepts must have at least 2 cats"
        self.concepts_model_ = ConceptsClf(c_cards, self.nn_model,
                                           self.nn_model_out_dim, self.device)
        
        self.train()
        optimizer = self._get_optimizer()
        start_time = time()
        bg_size = int(self.train_bg_part * X.shape[0])
        
        for e in range(1, self.epochs + 1):
            data = dataset_generator(X, c, t, d, bg_size, self.batch_num)
            
            C_back = self.np2torch(data[0], dtype=torch.long)
            T_back = self.np2torch(data[1])
            D_back = self.np2torch(data[2], torch.int)
            X_target = self.np2torch(data[3], device='cpu')
            C_target = self.np2torch(data[3], device='cpu', dtype=torch.long)
            T_target = self.np2torch(data[4], device='cpu')
            D_target = self.np2torch(data[5], torch.int, device='cpu')
            dataset = TensorDataset(C_back, T_back, D_back, X_target, C_target, T_target, D_target)
            data_loader = DataLoader(dataset, 1, shuffle=False)

            prog_bar = tqdm(data_loader,
                            f'Epoch {e}', unit='task', ascii=True)

            for c_b, t_b, d_b, x_t, c_t, t_t, d_t in prog_bar:
                c_b.squeeze_(0)
                t_b.squeeze_(0)
                d_b.squeeze_(0)
                x_t.squeeze_(0)
                c_t.squeeze_(0)
                t_t.squeeze_(0)
                d_t.squeeze_(0)
                
                optimizer.zero_grad()
                self._set_background(c_b, t_b, d_b)
                
                target_ds = TensorDataset(x_t, t_t, d_t)
                target_loader = DataLoader(target_ds, sub_batch_len, False)
                likelihood = 0
                
                for x_t_b, t_t_b, d_t_b in target_loader:
                    x_t_b, t_t_b, d_t_b = x_t_b.to(self.device), t_t_b.to(self.device), d_t_b.to(self.device)
                    hmm = self(self.np2torch(X))
            
    def _calc_exp_time(self, surv_func: torch.Tensor) -> torch.Tensor:
        integral = self.T_bg[None, 0] + \
            torch.sum(surv_func[:, :-1] * self.T_diff_int[None, :], dim=-1)
        return integral # (batch)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        c = self.concepts_model_(x)
        sf = self.beran(self.C_bg, self.D_bg, c)
        E_T = self._calc_exp_time(sf)
        return E_T, c
        
    @torch.inference_mode()
    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        E_T_t, c_t = self(x)
        E_T = E_T_t.cpu().to_numpy()
        c = []
        for cur_tens in c_t:
            c.append(cur_tens.cpu().numpy())
        return E_T, c
