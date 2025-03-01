from torchvision.datasets import CIFAR10, MNIST
from torchvision.transforms.v2 import PILToTensor
from torch.utils.data import DataLoader
import numpy as np
from typing import Tuple

def make_weibull_time(l: float, b: np.ndarray,
                      nu: float, x: np.ndarray, 
                      seed: int | None = None,
                      f_sin_cont: bool = False) -> np.ndarray:
    assert b.ndim == 1 and b.shape[0] == x.shape[1], (b.shape[0], x.shape[1])
    gen = np.random.default_rng(seed)
    u = gen.uniform(0, 1, x.shape[0])
    v = np.sum(b[None, :] * x, axis=-1)
    if not f_sin_cont:
        return np.power(-np.log(u) / (l * np.exp(v)), 1 / nu)
    else:
        return np.power(-np.log(u) / (l * (np.sin(v) + 1.001)), 1 / nu)

def get_proc_cifar_np() -> Tuple[np.ndarray, np.ndarray]:
    X_res_list = []
    y_res_list = []
    for train_flag in [True, False]:
        ds = DataLoader(CIFAR10('CIFAR10', download=False, 
                                transform=PILToTensor(), train=train_flag), 8192, False)
        X_list = []
        y_list = []
        for x, y in ds:
            X_list.append(x)
            y_list.append(y)
        X_np = np.concatenate(X_list, axis=0) # (:, 3, 32, 32)
        # std = np.std(X_np, 0, keepdims=True)
        # std[std < 1e-15] = 1.0
        # X_np = (X_np - np.mean(X_np, 0, keepdims=True)) / std
        X_np = X_np / 255.0
        y_np = np.concatenate(y_list, axis=0)
        X_res_list.append(X_np)
        y_res_list.append(y_np)
    return X_res_list[0], X_res_list[1], y_res_list[0], y_res_list[1] # train, test

# labels: ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# concept 0: number of animals
# concept 1: number of vehicles
# concept 2: number of flying objects
# concept 3: is there a cat on the picture
def get_4cifar_ds(X, y, samples_num: int, 
                  uncens_part: float,
                  l: float, nu: float,
                  b: np.ndarray,
                  seed: int,
                  f_use_one_hot: bool=False,
                  f_return_oh: bool=False):
    if f_use_one_hot:
        assert b.shape[0] == 17
    cls_to_id = dict(zip(['airplane', 'automobile', 'bird', 'cat',
                          'deer', 'dog', 'frog', 'horse', 'ship', 'truck'], range(10)))
    rng = np.random.default_rng(seed)
    result_X = np.empty((samples_num, 3, 64, 64))
    result_T = np.zeros(samples_num, np.float64)
    result_D = np.zeros(samples_num, np.intp)
    range_idx = np.arange(X.shape[0])
    I = rng.choice(range_idx, (samples_num, 4))
    result_X[..., :32, :32] = X[I[:, 0]]
    result_X[..., :32, 32:] = X[I[:, 1]]
    result_X[..., 32:, :32] = X[I[:, 2]]
    result_X[..., 32:, 32:] = X[I[:, 3]]
    Y = np.empty((samples_num, 4), dtype=np.intp)
    for i in range(4):
        Y[:, i] = y[I[:, i]]
    C = np.empty((samples_num, 4))
    C[:, 0] = np.sum((Y == cls_to_id['bird']) | (Y == cls_to_id['cat']) |
                     (Y == cls_to_id['deer']) | (Y == cls_to_id['dog']) | 
                     (Y == cls_to_id['frog']) | (Y == cls_to_id['horse']), axis=-1)
    C[:, 1] = np.sum((Y == cls_to_id['airplane']) | (Y == cls_to_id['automobile']) |
                     (Y == cls_to_id['ship']) | (Y == cls_to_id['truck']), axis=-1)
    C[:, 2] = np.sum((Y == cls_to_id['airplane']) | (Y == cls_to_id['bird']), axis=-1)
    C[:, 3] = np.sum(Y == cls_to_id['cat'], axis=-1) > 0
    C = C - C.min(axis=0, keepdims=True)
    if f_use_one_hot:
        E_5 = np.eye(5, dtype=np.intp)
        C_oh1 = np.take_along_axis(E_5, C[:, 0, None].astype(np.intp), axis=0)
        C_oh2 = np.take_along_axis(E_5, C[:, 1, None].astype(np.intp), axis=0)
        C_oh3 = np.take_along_axis(E_5, C[:, 2, None].astype(np.intp), axis=0)
        E_2 = np.eye(2, dtype=np.intp)
        C_oh4 = np.take_along_axis(E_2, C[:, 3, None].astype(np.intp), axis=0)
        C_oh = np.concatenate((C_oh1, C_oh2, C_oh3, C_oh4), axis=-1)
        print(C_oh[0])
        print(C[0])
    if not f_use_one_hot:
        result_T = make_weibull_time(l, b, nu, C.astype(np.double), seed)
    else:
        result_T = make_weibull_time(l, b, nu, C_oh.astype(np.double), seed)
    result_D = rng.binomial(2, uncens_part, samples_num)
    to_return = [result_X, result_T, result_D, C.astype(np.intp)]
    if f_return_oh:
        to_return.append(C_oh)
    return (*to_return,)

def get_proc_mnist_np() -> Tuple[np.ndarray, np.ndarray]:
    ds = DataLoader(MNIST('MNIST', download=False, transform=PILToTensor()), 8192, False)
    X_list = []
    y_list = []
    for x, y in ds:
        X_list.append(x)
        y_list.append(y)
    X_np = np.concatenate(X_list, axis=0) # (60000, 1, 28, 28)
    X_np = X_np / 255.0
    # std = np.std(X_np, 0, keepdims=True)
    # std[std < 1e-15] = 1.0
    # X_np = (X_np - np.mean(X_np, 0, keepdims=True)) / std
    y_np = np.concatenate(y_list, axis=0)
    return X_np, y_np

def get_4mnist_ds(X, y, samples_num: int, 
                  uncens_part: float,
                  l: float, nu: float,
                  b: np.ndarray,
                  seed: int,
                  f_use_one_hot: bool=False):
    rng = np.random.default_rng(seed)
    result_X = np.empty((samples_num, 1, 56, 56))
    if not f_use_one_hot:
        assert b.shape[0] == 4, "b must be length of 4"
        Y = np.empty((samples_num, 4), np.intp)
    else:
        assert b.shape[0] == 40, "b must be length of 40"
        Y_oh = np.zeros((samples_num, 4, 10), np.intp)
        Y = np.empty((samples_num, 4), np.intp)
    result_T = np.zeros(samples_num, np.float64)
    result_D = np.zeros(samples_num, np.intp)
    range_idx = np.arange(10)
    dig_idx_list = []
    for i in range(10):
        dig_idx_list.append(np.argwhere(y == i))
    slice_list = [(slice(None, 28), slice(None, 28)),
                  (slice(None, 28), slice(28, None)),
                  (slice(28, None), slice(None, 28)),
                  (slice(28, None), slice(28, None))]
    for i in range(samples_num):
        cur_digits = rng.choice(range_idx, 4, False, shuffle=True)
        Y[i, :] = cur_digits
        for j in range(4):
            if f_use_one_hot:
                Y_oh[i, j, cur_digits[j]] = 1
            dig_idx = dig_idx_list[cur_digits[j]][rng.integers(0, len(dig_idx_list[cur_digits[j]]))]
            result_X[i, :, slice_list[j][0], slice_list[j][1]] = X[dig_idx]
    if f_use_one_hot:
        Y_oh = Y_oh.reshape((samples_num, -1))
        result_T = make_weibull_time(l, b, nu, Y_oh, seed)
    else:
        result_T = make_weibull_time(l, b, nu, Y, seed)
    result_D = rng.binomial(2, uncens_part, samples_num)
    return result_X, result_T, result_D, Y