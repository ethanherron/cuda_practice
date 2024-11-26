# engine.py

import numpy as np
from typing import Set
from numba import cuda
import math


# add some numba accelerated ops 
@cuda.jit
def add_gpu(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.shape == b.shape:
        return a + b
    elif b.shape[0] == 1 and a.shape[1] == b.shape[1]:
        out = np.empty_like(a)
        for i in prange(a.shape[0]):
            for j in range(a.shape[1]):
                out[i, j] = a[i, j] + b[0, j]
        return out
    else:
        raise AssertionError("Sizes of a, b do not match")

@njit
def mul_gpu(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a * b

@njit
def matmul_gpu(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b

@njit
def matmul_T_right(grad_output: np.ndarray, b: np.ndarray) -> np.ndarray:
    return grad_output @ b.T

@njit
def matmul_T_left(a: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
    return a.T @ grad_output

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
        self.data = data.astype(np.float32)
        self.grad = np.zeros_like(self.data) if requires_grad else None
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
        if self.requires_grad:
            self.grad = np.zeros_like(self.data)
        else:
            self.grad = None
        
class Add(Operation):
    @staticmethod
    def apply(a, b):
        a = a if isinstance(a, Variable) else Variable(np.array(a), requires_grad=False)
        b = b if isinstance(b, Variable) else Variable(np.array(b), requires_grad=False)
        out = Variable(add_gpu(a.data, b.data))
        out._prev = {a, b}
        
        def _backward():
            if a.requires_grad:
                if a.data.shape != out.data.shape:
                    grad_a = out.grad.sum(axis=0)
                else:
                    grad_a = out.grad
                a.grad += grad_a
            if b.requires_grad:
                if b.data.shape != out.data.shape:
                    grad_b = out.grad.sum(axis=0)
                else:
                    grad_b = out.grad
                b.grad += grad_b
        out._backward = _backward
        return out

class Mul(Operation):
    @staticmethod
    def apply(a, b):
        a = a if isinstance(a, Variable) else Variable(np.array(a), requires_grad=False)
        b = b if isinstance(b, Variable) else Variable(np.array(b), requires_grad=False)
        out = Variable(mul_gpu(a.data, b.data))
        out._prev = {a, b}
        
        def _backward():
            if a.requires_grad:
                grad_a = mul_gpu(out.grad, b.data)
                a.grad += grad_a
            if b.requires_grad:
                grad_b = mul_gpu(out.grad, a.data)
                b.grad += grad_b
        out._backward = _backward
        return out
    
class MatMul(Operation):
    @staticmethod
    def apply(a, b):
        a = a if isinstance(a, Variable) else Variable(np.array(a), requires_grad=False)
        b = b if isinstance(b, Variable) else Variable(np.array(b), requires_grad=False)
        out = Variable(matmul_gpu(a.data, b.data))
        out._prev = {a, b}
        
        def _backward():
            if a.requires_grad:
                grad_a = matmul_T_right(out.grad, b.data)
                a.grad += grad_a
            if b.requires_grad:
                grad_b = matmul_T_left(a.data, out.grad)
                b.grad += grad_b
        out._backward = _backward
        return out
        