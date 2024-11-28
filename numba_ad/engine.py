# engine.py

import numpy as np
from numba import cuda, float32
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
        if isinstance(data, np.ndarray):
            self.data = cuda.to_device(data.astype(np.float32))
        else:
            self.data = cuda.to_device(np.array(data, dtype=np.float32))
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
        self.grad = np.ones_like(self.data.copy_to_host())
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
        out = Variable(np.empty_like(a.data.copy_to_host()))
        out._prev = {a, b}
        
        # kernel
        @cuda.jit
        def add_kernel(x, y, out):
            idx = cuda.grid(1)
            if idx < out.size:
                out[idx] = x[idx] +y[idx]
        
        threads_per_block = 256
        blocks_per_grid = (out.data.size + (threads_per_block - 1)) // threads_per_block
        add_kernel[blocks_per_grid, threads_per_block](a.data, b.data, out.data)
        
        def _backward():
            if a.requires_grad:
                a_grad = out.grad
                if isinstance(a.grad, np.ndarray):
                    a.grad = a.grad + a_grad.copy_to_host() if a.grad is not None else a_grad.copy_to_host()
                else:
                    a.grad = a.grad + a_grad
            if b.requires_grad:
                b_grad = out.grad
                if isinstance(b.grad, np.ndarray):
                    b.grad = b.grad + b_grad.copy_to_host() if b.grad is not None else b_grad.copy_to_host()
                else:
                    b.grad = b.grad + b_grad
        out._backward = _backward
        return out

class Mul(Operation):
    @staticmethod
    def apply(a, b):
        a = a if isinstance(a, Variable) else Variable(np.array(a), requires_grad=False)
        b = b if isinstance(b, Variable) else Variable(np.array(b), requires_grad=False)
        out = Variable(np.empty_like(a.data.copy_to_host()))
        out._prev = {a, b}
        
        # kernel
        @cuda.jit
        def mul_kernel(x, y, out):
            idx = cuda.grid(1)
            if idx < out.size:
                out[idx] = x[idx] * y[idx]
        
        threads_per_block = 256
        blocks_per_grid = (out.data.size + (threads_per_block - 1)) // threads_per_block
        mul_kernel[blocks_per_grid, threads_per_block](a.data, b.data, out.data)
        
        def _backward():
            if a.requires_grad:
                @cuda.jit
                def mul_grad_a(y, grad_out, grad_a):
                    idx = cuda.grid(1)
                    if idx < grad_a.size:
                        grad_a[idx] = y[idx] * grad_out[idx]
                
                grad_a_device = cuda.device_array_like(a.data)
                mul_grad_a[blocks_per_grid, threads_per_block](b.data, out.grad, grad_a_device)
                if a.grad is not None:
                    a.grad = a.grad + grad_a_device
                else:
                    a.grad = grad_a_device
            
            if b.requires_grad:
                @cuda.jit
                def mul_grad_b(x, grad_out, grad_b):
                    idx = cuda.grid(1)
                    if idx < grad_b.size:
                        grad_b[idx] = x[idx] * grad_out[idx]
                
                grad_b_device = cuda.device_array_like(b.data)
                mul_grad_b[blocks_per_grid, threads_per_block](a.data, out.grad, grad_b_device)
                if b.grad is not None:
                    b.grad = b.grad + grad_b_device
                else:
                    b.grad = grad_b_device
                    
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
        