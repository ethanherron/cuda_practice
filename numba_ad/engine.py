# engine.py

import numpy as np
from numba import cuda, float32
from numba.cuda.cudadrv.devicearray import DeviceNDArray

from typing import Dict, Optional, Tuple, Set

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
        if isinstance(data, DeviceNDArray):
            self.data = data
            self.shape = data.shape
        elif isinstance(data, np.ndarray):
            host_data = np.ascontiguousarray(data, dtype=np.float32)
            self.data = cuda.to_device(host_data)
            self.shape = host_data.shape
        else:
            host_data = np.ascontiguousarray(np.array(data, dtype=np.float32))
            self.data = cuda.to_device(host_data)
            self.shape = host_data.shape
        
        self.requires_grad = requires_grad
        self._backward = lambda: None 
        self._prev = set()
        
        if self.requires_grad:
            self.grad = cuda.device_array_like(self.data)
            self.zero_grad()
        else:
            self.grad = None
        
    def __add__(self, other):
        return Add.apply(self, other)
    
    def __mul__(self, other):
        return Mul.apply(self, other)
    
    def __matmul__(self, other):
        return MatMul.apply(self, other)
    
    def backward(self):
        if not self.requires_grad:
            raise RuntimeError("can't backprop variable that doesn't require grad")
        
        @cuda.jit
        def fill_grad(grad, value):
            idx = cuda.grid(1)
            if idx < grad.size:
                grad[idx] = value
        
        threads_per_block = 256
        blocks_per_grid = (self.grad.size + (threads_per_block - 1)) // threads_per_block
        try:
            fill_grad[blocks_per_grid, threads_per_block](self.grad, 1.0)
            cuda.synchronize()
        except cuda.cudadrv.driver.CudaAPIError as e:
            print(f'cuad error: {e}')
        
        # build graph
        topology = []
        visited = set()
        def build_topology(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topology(child)
                topology.append(v)
        build_topology(self)
        
        for v in reversed(topology):
            v._backward()
            
    def zero_grad(self):
        if self.grad is not None:
            @cuda.jit
            def reset_grad(grad):
                idx = cuda.grid(1)
                if idx < grad.size:
                    grad[idx] = 0.0
            
            threads_per_block = 256
            blocks_per_grid = (self.grad.size + (threads_per_block - 1)) // threads_per_block
            try:
                reset_grad[blocks_per_grid, threads_per_block](self.grad)
                cuda.synchronize()
            except cuda.cudadrv.driver.CudaAPIError as e:
                print(f'cuad error: {e}')
        
class Add(Operation):
    @staticmethod
    def apply(a, b):
        if a.shape != b.shape:
            raise ValueError(f"tensors in add op dont line up - {a.shape} and {b.shape}")
        a = a if isinstance(a, Variable) else Variable(np.array(a), requires_grad=False)
        b = b if isinstance(b, Variable) else Variable(np.array(b), requires_grad=False)
   
        out = Variable(cuda.device_array_like(a.data), requires_grad=a.requires_grad or b.requires_grad)
        out._prev = {a, b}
        
        # kernel
        @cuda.jit
        def add_kernel(x, y, out):
            idx = cuda.grid(1)
            if idx < out.size:
                out[idx] = x[idx] +y[idx]
        
        threads_per_block = 256
        blocks_per_grid = (out.data.size + (threads_per_block - 1)) // threads_per_block
        try:
            add_kernel[blocks_per_grid, threads_per_block](a.data, b.data, out.data)
            cuda.synchronize()
        except cuda.cudadrv.driver.CudaAPIError as e:
            print(f'cuad error: {e}')
        
        def _backward():
            if a.requires_grad:
                @cuda.jit
                def add_grad_a(grad_out, grad_a):
                    idx = cuda.grid(1)
                    if idx < grad_a.size:
                        grad_a[idx] += grad_out[idx]
                try:
                    add_grad_a[blocks_per_grid, threads_per_block](out.grad, a.grad)
                    cuda.synchronize()
                except cuda.cudadrv.driver.CudaAPIError as e:
                    print(f'cuad error: {e}')
            if b.requires_grad:
                @cuda.jit
                def add_grad_b(grad_out, grad_b):
                    idx = cuda.grid(1)
                    if idx < grad_b.size:
                        grad_b[idx] += grad_out[idx]
                try:
                    add_grad_b[blocks_per_grid, threads_per_block](out.grad, b.grad)
                    cuda.synchronize()
                except cuda.cudadrv.driver.CudaAPIError as e:
                    print(f'cuad error: {e}')
        out._backward = _backward
        return out

class Mul(Operation):
    @staticmethod
    def apply(a, b):
        a = a if isinstance(a, Variable) else Variable(np.array(a), requires_grad=False)
        b = b if isinstance(b, Variable) else Variable(np.array(b), requires_grad=False)
        
        out = Variable(cuda.device_array_like(a.data), requires_grad=a.requires_grad or b.requires_grad)
        out._prev = {a, b}
        
        # kernel
        @cuda.jit
        def mul_kernel(x, y, out):
            idx = cuda.grid(1)
            if idx < out.size:
                out[idx] = x[idx] * y[idx]
        
        threads_per_block = 256
        blocks_per_grid = (out.data.size + (threads_per_block - 1)) // threads_per_block
        try:
            mul_kernel[blocks_per_grid, threads_per_block](a.data, b.data, out.data)
            cuda.synchronize()
        except cuda.cudadrv.driver.CudaAPIError as e:
            print(f'cuad error: {e}')
        
        def _backward():
            if a.requires_grad:
                @cuda.jit
                def mul_grad_a(b, grad_out, grad_a):
                    idx = cuda.grid(1)
                    if idx < grad_a.size:
                        grad_a[idx] = b[idx] * grad_out[idx]
                
                try:
                    mul_grad_a[blocks_per_grid, threads_per_block](b.data, out.grad, a.grad)
                    cuda.synchronize()
                except cuda.cudadrv.driver.CudaAPIError as e:
                    print(f'cuad error: {e}')
            
            if b.requires_grad:
                @cuda.jit
                def mul_grad_b(a, grad_out, grad_b):
                    idx = cuda.grid(1)
                    if idx < grad_b.size:
                        grad_b[idx] = a[idx] * grad_out[idx]
                
                try:
                    mul_grad_b[blocks_per_grid, threads_per_block](a.data, out.grad, b.grad)
                    cuda.synchronize()
                except cuda.cudadrv.driver.CudaAPIError as e:
                    print(f'cuad error: {e}')
        out._backward = _backward
        return out
    
class MatMul(Operation):
    @staticmethod
    def apply(a, b):
        a = a if isinstance(a, Variable) else Variable(np.array(a), requires_grad=False)
        b = b if isinstance(b, Variable) else Variable(np.array(b), requires_grad=False)
        
        if len(a.shape) == 2 and len(b.shape) == 2:
            batch = 1
            a_batch, a_M, a_K = 1, a.shape[0], a.shape[1]
            b_batch, b_K, b_N = 1, b.shape[0], b.shape[1]
        elif len(a.shape) == 3 and len(b.shape) == 3:
            if a.shape[0] != b.shape[0]:
                raise ValueError(f"Batch sizes do not match: {a.shape[0]} vs {b.shape[0]}")
            batch = a.shape[0]
            a_M, a_K = a.shape[1], a.shape[2]
            b_K, b_N = b.shape[1], b.shape[2]
        else:
            raise ValueError("MatMul only supports 2D or 3D tensors for now.")
        
        # Validate inner dimensions
        if a_K != b_K:
            raise ValueError(f"Incompatible inner dimensions for MatMul: {a_K} vs {b_K}")
        
        out_shape = (batch, a_M, b_N) if batch > 1 else (a_M, b_N)
        out = Variable(cuda.device_array(out_shape, dtype=np.float32), requires_grad=a.requires_grad or b.requires_grad)
        out._prev = {a, b}
        
        @cuda.jit
        def batched_matmul_kernel(a, b, c, batch, M, K, N):
            batch_idx, row, col = cuda.grid(3)
            if batch_idx < batch and row < M and col < N:
                t = 0.0
                for k in range(K):
                    t += a[batch_idx, row, k] * b[batch_idx, k, col]
                c[batch_idx, row, col] = t
                
        threads_per_block = (8, 8, 8)
        blocks_per_grid = (
            (batch + threads_per_block[0] - 1) // threads_per_block[0],
            (a_M + threads_per_block[1] - 1) // threads_per_block[1],
            (b_N + threads_per_block[2] - 1) // threads_per_block[2]
        )
        try:
            matmul_kernel[blocks_per_grid, threads_per_block](a.data, b.data, out.data, batch, a_M, a_K, b_N)
            cuda.synchronize()
        except cuda.cudadrv.driver.CudaAPIError as e:
            print(f'cuad error: {e}')
        
        def _backward():
            if a.requires_grad:
                # grad_a += grad out @ b.T
                @cuda.jit
                def batched_matmul_grad_a(grad_out, b, grad_a, batch, M, K, N):
                    batch_idx, row, col = cuda.grid(3)
                    if batch_idx < batch and row < M and col < K:
                        t = 0.0
                        for n in range(N):
                            t += grad_out[batch_idx, row, n] * b[batch_idx, col, n]
                        cuda.atomic.add(grad_a, (batch_idx, row, col), t)
                try:
                    batched_matmul_grad_a[blocks_per_grid, threads_per_block](out.grad, b.data, a.grad, batch, a_M, a_K, b_N)
                    cuda.synchronize()
                except cuda.cudadrv.driver.CudaAPIError as e:
                    print(f'cuad error: {e}')
                
            if b.requires_grad:
                # grad b = a.T @ grad out
                @cuda.jit
                def batched_matmul_grad_b(a, grad_out, grad_b, batch, M, K, N):
                    batch_idx, row, col = cuda.grid(3)
                    if batch_idx < batch and row < K and col < N:
                        t = 0.0
                        for m in range(M):
                            t += a[batch_idx, m, row] * grad_out[batch_idx, m, col]
                        cuda.atomic.add(grad_b, (batch_idx, row, col), t)
                try:
                    batched_matmul_grad_b[blocks_per_grid, threads_per_block](a.data, out.grad, b.grad, batch, a_M, a_K, b_N)
                    cuda.synchronize()
                except cuda.cudadrv.driver.CudaAPIError as e:
                    print(f'cuad error: {e}')
            
            out._backward = _backward
            return out
    
    
if __name__ == "__main__":
    import numpy as np
    
    # Define batch size and matrix dimensions
    batch_size = 32  # Example batch size
    M, K, N = 128, 256, 64  # Example dimensions
    
    # Initialize random batched matrices A and B
    a_np = np.random.randn(batch_size, M, K).astype(np.float32)
    b_np = np.random.randn(batch_size, K, N).astype(np.float32)
    
    # Wrap them in Variables
    try:
        print("Creating Variable a...")
        a = Variable(a_np, requires_grad=True)
        print("Variable a created successfully.")
    except cuda.cudadrv.driver.CudaAPIError as e:
        print(f'cuda error during Variable a initialization: {e}')
    
    try:
        print("Creating Variable b...")
        b = Variable(b_np, requires_grad=True)
        print("Variable b created successfully.")
    except cuda.cudadrv.driver.CudaAPIError as e:
        print(f'cuda error during Variable b initialization: {e}')
    
    # Perform batched matrix multiplication
    try:
        print("Performing MatMul operation...")
        c = MatMul.apply(a, b)
        print("MatMul operation completed.")
    except cuda.cudadrv.driver.CudaAPIError as e:
        print(f'cuda error during MatMul.apply: {e}')
    
    # Assume some loss function; for simplicity, sum all elements
    # Equivalent to L = sum(C)
    # So grad_out is a batched matrix of ones
    @cuda.jit
    def fill_grad_out(grad_out, value):
        idx = cuda.grid(1)
        if idx < grad_out.size:
            grad_out[idx] = value
    
    threads_per_block = 256
    blocks_per_grid = (c.grad.size + (threads_per_block - 1)) // threads_per_block
    try:
        print("Launching fill_grad_out kernel...")
        fill_grad_out[blocks_per_grid, threads_per_block](c.grad, 1.0)
        cuda.synchronize()
        print("fill_grad_out kernel executed successfully.")
    except cuda.cudadrv.driver.CudaAPIError as e:
        print(f'cuda error during fill_grad_out: {e}')
    
    # Perform backward pass
    try:
        print("Starting backward pass...")
        c.backward()
        print("Backward pass completed.")
    except cuda.cudadrv.driver.CudaAPIError as e:
        print(f'cuda error during backward pass: {e}')
    
    # Fetch gradients from GPU to host for verification
    try:
        print("Copying gradients to host...")
        a_grad = a.grad.copy_to_host()
        b_grad = b.grad.copy_to_host()
        print("Gradients copied successfully.")
    except cuda.cudadrv.driver.CudaAPIError as e:
        print(f'cuda error during copy_to_host: {e}')
        exit(1)
    
    # Compute expected gradients using NumPy for verification
    try:
        print("Computing expected gradients using NumPy...")
        expected_a_grad = np.matmul(np.ones((batch_size, M, N), dtype=np.float32), b_np.transpose(0, 2, 1))
        expected_b_grad = np.matmul(a_np.transpose(0, 2, 1), np.ones((batch_size, M, N), dtype=np.float32))
        print("Expected gradients computed.")
    except Exception as e:
        print(f'Error during expected gradients computation: {e}')
        exit(1)
    
    # Verify gradients
    try:
        print("Verifying gradients...")
        assert np.allclose(a_grad, expected_a_grad, atol=1e-5), "Gradients w.r.t A do not match!"
        assert np.allclose(b_grad, expected_b_grad, atol=1e-5), "Gradients w.r.t B do not match!"
        print("Batched MatMul Forward and Backward Pass Successful!")
    except AssertionError as e:
        print(e)
        exit(1)
