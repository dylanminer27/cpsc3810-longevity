# Optimizer for gradient-based learning

from typing import Optional
import numpy as np

SUPPORTED_LOSSES = ('mse', 'logistic', 'softmax')
SUPPORTED_OPTIMIZERS = (
    'gradient_descent',
    'momentum_gradient_descent',
    'stochastic_gradient_descent',
    'momentum_stochastic_gradient_descent',
)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=0, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=0, keepdims=True)


class Optimizer(object):

    def __init__(self,
                 alpha: float,
                 eta_decay_factor: float,
                 beta: float,
                 optimizer_type: str):

        if optimizer_type not in SUPPORTED_OPTIMIZERS:
            raise ValueError('Unsupported optimizer type: {}'.format(optimizer_type))

        self.__alpha = alpha
        self.__eta_decay_factor = eta_decay_factor
        self.__beta = beta
        self.__optimizer_type = optimizer_type
        self.__momentum: Optional[np.ndarray] = None

    def __polynomial_decay(self, time_step: int) -> float:
        return (time_step + 1) ** (-self.__eta_decay_factor)

    # Computes the gradient of the given loss function wrt the weights
    def __compute_gradients(self,
                            w: np.ndarray,
                            x: np.ndarray,
                            y: np.ndarray,
                            loss_func: str) -> np.ndarray:

        N = x.shape[1]

        if loss_func == 'mse':
            y_hat = w.T @ x                       
            grad = (2.0 / N) * x @ (y_hat - y).T  
            return grad

        if loss_func == 'logistic':
            z = w.T @ x                           
            p = 1.0 / (1.0 + np.exp(-z))          
            grad = (1.0 / N) * x @ (p - y).T      
            return grad

        if loss_func == 'softmax':
            logits = w.T @ x                      
            p = _softmax(logits)                 
            grad = (1.0 / N) * x @ (p - y).T      
            return grad

        raise ValueError('Unsupported loss function: {}'.format(loss_func))

    # Performs one gradient step and returns the updated weights
    def update(self,
               w: np.ndarray,
               x: np.ndarray,
               y: np.ndarray,
               loss_func: str,
               batch_size: int,
               time_step: int) -> np.ndarray:
  
        eta = self.__alpha * self.__polynomial_decay(time_step)

        if self.__momentum is None:
            self.__momentum = np.zeros_like(w)

        if self.__optimizer_type in ('stochastic_gradient_descent',
                                     'momentum_stochastic_gradient_descent'):
            n_samples = x.shape[1]
            bs = min(batch_size, n_samples)
            idx = np.random.choice(n_samples, size=bs, replace=False)
            x_batch = x[:, idx]
            y_batch = y[:, idx]
        else:
            x_batch = x
            y_batch = y

        gradients = self.__compute_gradients(w, x_batch, y_batch, loss_func)

        if self.__optimizer_type in ('momentum_gradient_descent',
                                     'momentum_stochastic_gradient_descent'):
            self.__momentum = self.__beta * self.__momentum + gradients
            return w - eta * self.__momentum

        return w - eta * gradients
