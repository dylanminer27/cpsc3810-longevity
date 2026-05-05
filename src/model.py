# Implementation of LinearRegression and LogisticRegression classes 

from typing import List, Optional
import numpy as np
from src.optimizer import Optimizer, _softmax


def _prepend_bias(x: np.ndarray) -> np.ndarray:
    return np.vstack([np.ones((1, x.shape[1]), dtype=x.dtype), x])


class LinearRegression(object):

    def __init__(self):
        self.__weights: Optional[np.ndarray] = None
        self.__loss_history: List[float] = []


    def __fit_normal_equation(self, x: np.ndarray, y: np.ndarray) -> None:
        A = x @ x.T
        b = x @ y.T
        self.__weights = np.linalg.solve(A, b)  

    # Uses the SVD-based pseudo-inverse
    def __fit_pseudoinverse(self, x: np.ndarray, y: np.ndarray) -> None:
        self.__weights = np.linalg.pinv(x.T) @ y.T  

    def __fit_gradient_descent(self,
                               x: np.ndarray,
                               y: np.ndarray,
                               T: int,
                               alpha: float,
                               eta_decay_factor: float,
                               beta: float,
                               batch_size: int,
                               optimizer_type: str,
                               n_step_per_log: int,
                               verbose: bool) -> None:
        d = x.shape[0]
        self.__weights = np.zeros((d, 1), dtype=np.float32)
        self.__loss_history = []

        optimizer = Optimizer(
            alpha=alpha,
            eta_decay_factor=eta_decay_factor,
            beta=beta,
            optimizer_type=optimizer_type)

        for t in range(1, T + 1):
            if (t % n_step_per_log) == 0 or t == 1:
                y_hat = self.__weights.T @ x
                loss = float(np.mean((y_hat - y) ** 2))
                self.__loss_history.append(loss)
                if verbose:
                    print('Step={}  Loss={:.4f}'.format(t, loss))

            self.__weights = optimizer.update(
                w=self.__weights,
                x=x,
                y=y,
                loss_func='mse',
                batch_size=batch_size,
                time_step=t)

    # Fits the model
    def fit(self,
            x: np.ndarray,
            y: np.ndarray,
            solver: str = 'normal_equation',
            T: int = 5000,
            alpha: float = 1e-2,
            eta_decay_factor: float = 0.5,
            beta: float = 0.0,
            batch_size: Optional[int] = None,
            optimizer_type: str = 'stochastic_gradient_descent',
            n_step_per_log: int = 50,
            verbose: bool = False) -> List[float]:
       
        x = _prepend_bias(np.asarray(x, dtype=np.float32))
        y = np.asarray(y, dtype=np.float32)

        if solver == 'normal_equation':
            self.__fit_normal_equation(x, y)
        elif solver == 'pseudoinverse':
            self.__fit_pseudoinverse(x, y)
        elif solver == 'gradient_descent':
            if batch_size is None:
                batch_size = x.shape[1]
            self.__fit_gradient_descent(
                x=x, y=y, T=T,
                alpha=alpha,
                eta_decay_factor=eta_decay_factor,
                beta=beta,
                batch_size=batch_size,
                optimizer_type=optimizer_type,
                n_step_per_log=n_step_per_log,
                verbose=verbose)
        else:
            raise ValueError('Encountered unsupported solver: {}'.format(solver))

        return list(self.__loss_history)

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = _prepend_bias(np.asarray(x, dtype=np.float32))
        return self.__weights.T @ x

    def score(self,
              x: np.ndarray,
              y: np.ndarray,
              scoring_func: str = 'r_squared') -> float:
    
        y = np.asarray(y, dtype=np.float32)
        y_hat = self.predict(x)

        if scoring_func == 'mean_squared_error':
            return float(np.mean((y_hat - y) ** 2))

        if scoring_func == 'r_squared':
            y_mean = np.mean(y)
            ss_tot = np.sum((y - y_mean) ** 2)
            ss_res = np.sum((y - y_hat) ** 2)
            return float(1.0 - ss_res / ss_tot)

        raise ValueError('Encountered unsupported scoring function: {}'.format(scoring_func))

    @property
    def weights(self) -> np.ndarray:
        '''Returns a copy of the trained weight vector for inspection.'''
        return self.__weights.copy() if self.__weights is not None else None

# Multinomial logistic regression with softmax cross-entropy
class LogisticRegression(object):

    def __init__(self):
        self.__weights: Optional[np.ndarray] = None
        self.__n_classes: Optional[int] = None
        self.__loss_history: List[float] = []

    def fit(self,
            x: np.ndarray,
            y_one_hot: np.ndarray,
            T: int = 5000,
            alpha: float = 1e-2,
            eta_decay_factor: float = 0.5,
            beta: float = 0.0,
            batch_size: Optional[int] = None,
            optimizer_type: str = 'stochastic_gradient_descent',
            n_step_per_log: int = 50,
            verbose: bool = False) -> List[float]:
    
        x = _prepend_bias(np.asarray(x, dtype=np.float32))
        y_one_hot = np.asarray(y_one_hot, dtype=np.float32)

        d = x.shape[0]
        C = y_one_hot.shape[0]
        self.__n_classes = C
        self.__weights = np.zeros((d, C), dtype=np.float32)
        self.__loss_history = []

        if batch_size is None:
            batch_size = x.shape[1]

        optimizer = Optimizer(
            alpha=alpha,
            eta_decay_factor=eta_decay_factor,
            beta=beta,
            optimizer_type=optimizer_type)

        eps = 1e-12
        for t in range(1, T + 1):
            if (t % n_step_per_log) == 0 or t == 1:
                p = _softmax(self.__weights.T @ x)
                p = np.clip(p, eps, 1.0 - eps)
                loss = float(-np.mean(np.sum(y_one_hot * np.log(p), axis=0)))
                self.__loss_history.append(loss)
                if verbose:
                    print('Step={}  CE={:.4f}'.format(t, loss))

            self.__weights = optimizer.update(
                w=self.__weights,
                x=x,
                y=y_one_hot,
                loss_func='softmax',
                batch_size=batch_size,
                time_step=t)

        return list(self.__loss_history)

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = _prepend_bias(np.asarray(x, dtype=np.float32))
        logits = self.__weights.T @ x
        return np.argmax(logits, axis=0, keepdims=True).astype(np.int64)

    # Class probabilities
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        x = _prepend_bias(np.asarray(x, dtype=np.float32))
        return _softmax(self.__weights.T @ x)

    def score(self, x: np.ndarray, y_int: np.ndarray) -> float:
        y_hat = self.predict(x)
        return float(np.mean(y_hat == y_int))

    @property
    def weights(self) -> np.ndarray:
        return self.__weights.copy() if self.__weights is not None else None
