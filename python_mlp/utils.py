# utils.py

import numpy as np

from engine import Activation, Operation, Variable

class ReLU(Activation):
    @staticmethod
    def apply(a):
        a = a if isinstance(a, Variable) else Variable(np.array(a), requires_grad=False)
        out = Variable(np.maximum(0, a.data))
        out._prev = {a}
        
        def _backward():
            if a.requires_grad:
                grad = out.grad * (a.data > 0).astype(a.data.dtype)
                a.grad = a.grad + grad if a.grad is not None else grad
        out._backward = _backward
        return out
    
class CrossEntropyWithLogitsLoss:
    @staticmethod
    def apply(pred_logits, target):
        epsilon = 1e-10
        shift = pred_logits.data - np.max(pred_logits.data, axis=1, keepdims=True)
        e_logits = np.exp(shift)
        softmax = e_logits / np.sum(e_logits, axis=1, keepdims=True)
        log_softmax = np.log(softmax + epsilon)
        loss_data = - np.sum(target.data * log_softmax, axis=1)
        mean_loss = np.mean(loss_data)
        loss = Variable(np.array(mean_loss), requires_grad=True)
        loss._prev = {pred_logits, target}
        
        def _backward():
            if pred_logits.requires_grad:
                grad_pred_logits = (softmax - target.data) / pred_logits.data.shape[0]
                if pred_logits.grad is not None:
                    pred_logits.grad += grad_pred_logits
                else:
                    pred_logits.grad = grad_pred_logits
                    
        loss._backward = _backward
        return loss
    
class SGD:
    def __init__(self, parameters, lr):
        self.parameters = parameters
        self.lr = lr
        
    def step(self):
        for param in self.parameters:
            if param.requires_grad and param.grad is not None:
                param.data -= self.lr * param.grad
    
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
        