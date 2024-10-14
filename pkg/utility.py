import numpy as np


CENS_LABEL_TYPE = [('cens', '?'), ('time', 'f8')]

# separate arrays to the structured one
# first field - censoring flag (bool)
# second field - time to event
def get_str_array(T: np.ndarray, D: np.ndarray) -> np.recarray:
    assert T.shape[0] == D.shape[0]
    str_array = np.ndarray(shape=(T.shape[0]), dtype=CENS_LABEL_TYPE)
    str_array[CENS_LABEL_TYPE[0][0]] = D
    str_array[CENS_LABEL_TYPE[1][0]] = T
    return str_array
