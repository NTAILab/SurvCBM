import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from typing import Dict, Optional, List
try:
    import mlflow
except ImportError:
    print("can't import the mlflow library, mlflow logging is unavailable")

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

def proba_to_labels(proba: List[np.ndarray]) -> np.ndarray:
    labels = np.empty((proba[0].shape[0]), len(proba))
    for i, p in enumerate(proba):
        labels[:, i] = np.argmax(proba[i], axis=-1)
    return labels

class MlflowBinLogger:
    def __init__(self):
        self.calls = 0
        # self.start_time_stamp = time()
        
    def __call__(self, c_proba: Optional[List[np.ndarray]], true_c_vals: np.ndarray, custom_dict: Optional[Dict] = None) -> None:
        step_counter = self.calls + 1
        # time_stamp = time() - self.start_time_stamp
        metrics = {}
        # if E_T_pred is not None:
        #     metrics['c_index/valid'] = concordance_index_censored(y_true['cens'], y_true['time'], -E_T_pred)
        if c_proba is not None:
            all_roc_auc = []
            all_accuracy = []
            for i, c in enumerate(c_proba):
                s = np.argmax(c, axis=-1)
                all_accuracy.append(accuracy_score(true_c_vals[:, i], s))
                metrics[f'accuracy_{i + 1}/valid'] = all_accuracy[-1]
                if c.shape[-1] == 2:
                    c = c[:, 1]
                all_roc_auc.append(roc_auc_score(true_c_vals[:, i], c, average='macro', multi_class='ovr'))
                metrics[f'roc_auc_{i + 1}/valid'] = all_roc_auc[-1]
        if len(metrics) > 0: 
            mlflow.log_metrics(metrics, step_counter)
            mlflow.log_metric("roc_auc_mean/valid", np.mean(all_roc_auc), step_counter)
            mlflow.log_metric("accuracy_mean/valid", np.mean(all_accuracy), step_counter)
        if custom_dict is not None:
            mlflow.log_metrics(custom_dict, step_counter)
        self.calls += 1

class MlflowMultiLogger:
    def __init__(self):
        self.calls = 0
        
    def __call__(self, c_proba: Optional[List[np.ndarray]], true_c_vals: np.ndarray, custom_dict: Optional[Dict] = None) -> None:
        step_counter = self.calls + 1
        metrics = {}
        if c_proba is not None:
            all_f1 = []
            all_accuracy = []
            for i, c in enumerate(c_proba):
                s = np.argmax(c, axis=-1)
                all_accuracy.append(accuracy_score(true_c_vals[:, i], s))
                metrics[f'accuracy_{i + 1}/valid'] = all_accuracy[-1]
                all_f1.append(f1_score(true_c_vals[:, i], s, average='macro'))
                metrics[f'f1_{i + 1}/valid'] = all_f1[-1]
        if len(metrics) > 0: 
            mlflow.log_metrics(metrics, step_counter)
            mlflow.log_metric("f1_mean/valid", np.mean(all_f1), step_counter)
            mlflow.log_metric("accuracy_mean/valid", np.mean(all_accuracy), step_counter)
        if custom_dict is not None:
            mlflow.log_metrics(custom_dict, step_counter)
        self.calls += 1

# without concepts
class MlFlowRawLogger:
    def __init__(self):
        self.calls = 0
        
    def __call__(self, custom_dict: Optional[Dict] = None) -> None:
        step_counter = self.calls + 1
        if custom_dict is not None:
            mlflow.log_metrics(custom_dict, step_counter)
        self.calls += 1
