# engine.py

import numpy as np
from typing import Dict, Optional, Tuple

class Operation:
    @staticmethod
    def apply(*args):
        raise NotImplementedError
    
class Activation:
    @staticmethod
    def apply(a):
        raise NotImplementedError

class Variable:
    def __init__(self, data, requires_grad=True):
        self.data = data
        self.grad = None
        self.requires_grad = requires_grad
        self._backward = lambda: None 
        self._prev = set()
        
    def __add__(self, other):
        return Add.apply(self, other)
    
    def __mul__(self, other):
        return Mul.apply(self, other)
    
    def __matmul__(self, other):
        return MatMul.apply(self, other)
    
    def backward(self):
        self.grad = np.ones_like(self.data)
        topology = []
        visited = set()
        def build_topology(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topology(child)
                topology.append(v)
        build_topology(self)
        # reverse the graph
        for v in reversed(topology):
            v._backward()
            
    def zero_grad(self):
        self.grad = None
        
        
class Add(Operation):
    @staticmethod
    def apply(a, b):
        a = a if isinstance(a, Variable) else Variable(np.array(a), requires_grad=False)
        b = b if isinstance(b, Variable) else Variable(np.array(b), requires_grad=False)
        out = Variable(a.data + b.data)
        out._prev = {a, b}
        
        def _backward():
            if a.requires_grad:
                if a.data.shape != out.data.shape:
                    # Bias term: sum gradients over the batch dimension
                    grad_a = out.grad.sum(axis=0)
                else:
                    grad_a = out.grad
                a.grad = a.grad + grad_a if a.grad is not None else grad_a
            if b.requires_grad:
                if b.data.shape != out.data.shape:
                    # Bias term: sum gradients over the batch dimension
                    grad_b = out.grad.sum(axis=0)
                else:
                    grad_b = out.grad
                b.grad = b.grad + grad_b if b.grad is not None else grad_b
        out._backward = _backward
        return out

        
class Mul(Operation):
    @staticmethod
    def apply(a, b):
        a = a if isinstance(a, Variable) else Variable(np.array(a), requires_grad=False)
        b = b if isinstance(b, Variable) else Variable(np.array(b), requires_grad=False)
        out = Variable(a.data * b.data)
        out._prev = {a, b}
        
        def _backward():
            if a.requires_grad:
                a.grad = a.grad + (b.data * out.grad) if a.grad is not None else b.data * out.grad
            if b.requires_grad:
                b.grad = b.grad + (a.data * out.grad) if b.grad is not None else a.data * out.grad
        out._backward = _backward
        return out
    
class MatMul(Operation):
    @staticmethod
    def apply(a, b):
        a = a if isinstance(a, Variable) else Variable(np.array(a), requires_grad=False)
        b = b if isinstance(b, Variable) else Variable(np.array(b), requires_grad=False)
        out = Variable(a.data @ b.data)
        out._prev = {a, b}
        
        def _backward():
            if a.requires_grad:
                a.grad = a.grad + out.grad @ b.data.T if a.grad is not None else out.grad @ b.data.T
            if b.requires_grad:
                b.grad = b.grad + a.data.T @ out.grad if b.grad is not None else a.data.T @ out.grad
        out._backward = _backward
        return out
        