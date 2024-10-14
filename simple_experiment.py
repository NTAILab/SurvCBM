from pkg.model import SurvBN
from pkg.utility import get_str_array
import numpy as np
import torch

if __name__=="__main__":
    X = np.random.normal(0, 1, (100, 10))
    c = np.random.randint(0, 4, (100, 3))
    c = np.concatenate((np.random.randint(0, 2, (100, 1)), c), axis=-1)
    t = np.random.uniform(10, 100, 100)
    d = np.random.binomial(2, 0.5, 100)
    y = get_str_array(t, d)
    model = SurvBN(
        torch.nn.Sequential(
            torch.nn.Linear(10, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 4)
        ),
        4,
        'cpu',
        
    )
    model.fit(X, y, c)
