# utils.py

import numpy as np
from numba import njit, prange

from engine import Activation, Variable


# numba ops
@njit(parallel=True)
def relu_gpu(a: np.ndarray) -> np.ndarray:
    return np.maximum(0, a)

@njit(parallel=True)
def relu_derivative_gpu(a: np.ndarray, grad_out: np.ndarray) -> np.ndarray:
    num_samples, num_features = a.shape
    grad = grad_out.copy()
    for i in prange(num_samples):
        for j in range(num_features):
            if a[i, j] <= 0:
                grad[i, j] = 0.
    return grad

@njit(parallel=True)
def softmax_gpu(logits: np.ndarray) -> np.ndarray:
    num_samples, num_classes = logits.shape
    softmax = np.empty_like(logits)
    for i in prange(num_samples):
        max_logit = logits[i, 0]
        for j in range(1, num_classes):
            if logits[i, j] > max_logit:
                max_logit = logits[i, j]
        for j in range(num_classes):
            softmax[i, j] = np.exp(logits[i, j] - max_logit)
        sum_exp = 0.
        for j in range(num_classes):
            sum_exp += softmax[i, j]
        for j in range(num_classes):
            softmax[i, j] /= sum_exp
    return softmax

@njit(parallel=True)
def cross_entropy_loss_gpu(softmax: np.ndarray, target: np.ndarray) -> float:
    epsilon = 1e-10
    loss = 0.
    num_samples, num_classes = softmax.shape
    for i in prange(num_samples):
        for j in range(num_classes):
            loss -= target[i, j] * np.log(softmax[i, j] + epsilon)
    mean_loss = loss / num_samples
    return mean_loss

@njit(parallel=True)
def cross_entropy_grad_gpu(softmax: np.ndarray, target: np.ndarray, grad_output: float) -> np.ndarray:
    num_samples, num_classes = softmax.shape
    grad = np.empty_like(softmax)
    for i in prange(num_samples):
        for j in range(num_classes):
            grad[i, j] = (softmax[i, j] - target[i, j]) / num_samples * grad_output
    return grad

class ReLU(Activation):
    @staticmethod
    def apply(a):
        a = a if isinstance(a, Variable) else Variable(np.array(a, dtype=np.float32), requires_grad=False)
        out = Variable(relu_gpu(a.data))
        out._prev = {a}
        
        def _backward():
            if a.requires_grad:
                grad_a = relu_derivative_gpu(a.data, out.grad)
                a.grad += grad_a
        out._backward = _backward
        return out
    
class CrossEntropyWithLogitsLoss:
    @staticmethod
    def apply(pred_logits, target):
        pred_logits = pred_logits if isinstance(pred_logits, Variable) else Variable(np.array(pred_logits, dtype=np.float32), requires_grad=False)
        target = target if isinstance(target, Variable) else Variable(np.array(targets, dtype=np.float32), requires_grad=False)
        softmax = softmax_gpu(pred_logits.data)
        loss_value = cross_entropy_loss_gpu(softmax, target.data)
        loss = Variable(np.array(loss_value, dtype=np.float32), requires_grad=True)
        loss._prev = {pred_logits, target}
        
        def _backward():
            if pred_logits.requires_grad:
                grad_pred_logits = cross_entropy_grad_gpu(softmax, target.data, loss.grad)
                pred_logits.grad += grad_pred_logits
                
        loss._backward = _backward
        return loss
    
# gpu sgd step
@njit(parallel=True)
def sgd_step_gpu(parameters: list, grads: list, lr: float):
    for i in prange(len(parameters)):
        if grads[i] is not None:
            parameters[i] -= lr * grads[i]
    
class SGD:
    def __init__(self, parameters, lr):
        self.parameters = parameters
        self.lr = lr
        
    def step(self):
        # get param grads and data for numba funcs
        params = [param.data for param in self.parameters if param.requires_grad]
        grads = [param.grad for param in self.parameters if param.requires_grad]
        # step!
        sgd_step_gpu(params, grads, self.lr)
        updated_params = [param for param in self.parameters if param.requires_grad]
        for i, param in enumerate(updated_params):
            if param.grad is not None:
                param.data = params[i]
    
    def zero_grad(self):
        for param in self.parameters:
            param.zero_grad()
    
    
class Adam:
    def __init__(self, parameters, lr, beta_1=0.9, beta_2=0.999):
        self.parameters = parameters
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = 1e-10
        self.t = 0
        
        self.m = [np.zeros_like(param.data) for param in self.parameters]
        self.v = [np.zeros_like(param.data) for param in self.parameters]
        
    def step(self):
        self.t += 1
        for i, param in enumerate(self.parameters):
            grad = param.grad
            self.m[i] = self.beta_1 * self.m[i] + (1 - self.beta_1) * grad
            self.v[i] = self.beta_2 * self.v[i] + (1 - self.beta_2) * (grad ** 2)
            
            m_hat = self.m[i] / (1 - self.beta_1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta_2 ** self.t)
            
            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
            
    def zero_grad(self):
        for param in self.parameters:
            param.zero_grad()
        
        
class ScheduleFreeAdamW:
    def __init__(self, parameters, lr, warmup=100, beta_1=0.9, beta_2=0.999):
        self.parameters = parameters
        self.lr = lr
        self.warmup = warmup
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = 1e-10
        self.t = 0
        
        self.z = [np.copy(param.data) for param in self.parameters]
        self.v = [np.zeros_like(param.data) for param in self.parameters]
        self.c = 0
        
    def step(self):
        self.t += 1
        for i, param in enumerate(self.parameters):
            # momentum interpolation
            y = (1 - self.beta_1) * self.z[i] + self.beta_1 * param.data
            
            # compute grad and update var
            g = param.grad
            self.v[i] = self.beta_2 * self.v[i] + (1 - self.beta_2) * (g**2)
            
            # lr with warmup and bias correction
            lr_t = self.lr * np.sqrt(1 - self.beta_2 ** self.t) / (1 - self.beta_1 ** self.t)
            lr_t *= min(1, self.t / self.warmup)
            
            # update z (parameters)
            self.z[i] -= lr_t * g / (np.sqrt(self.v[i]) + self.epsilon)
            
            # weights iterate mean
            gamma = self.lr ** 2
            c_next = gamma / (gamma + self.t * self.lr ** 2)
            param.data = (1 - c_next) * param.data + c_next * self.z[i]
            
    def zero_grad(self):
        for param in self.parameters:
            param.zero_grad()
            
            
            