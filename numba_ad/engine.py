import numpy as np
from typing import Callable, List, Dict, Tuple, Optional

# AD

Tensor = Dict[str, Optional[np.ndarray]]

computation_graph: List[Dict] = []

def create_tensor(data: np.ndarray, requires_grad: bool=False) -> Tensor:
    tensor = {'data': data, 'grad': None, 'requires_grad': requires_grad}
    return tensor

def backward(output: Tensor) -> None:
    output['grad'] = np.ones_like(output['data'])
    grads = {id(output): output['grad']}

    for node in reversed(computation_graph):
        output_tensor = node['output']
        output_id = id(output_tensor)
        grad_out = grads.get(output_id)
        if grad_out is None:
            raise KeyError(f'missing grad for tensor: {output_id}')
        grad_inputs = node['backward'](grad_out)

        for inp, grad in zip(node['inputs'], grad_inputs):
            if inp['requires_grad']:
                if inp['grad'] is None:
                    inp['grad'] = grad
                else:
                    inp['grad'] += grad
                grads[id(inp)] = inp['grad']

def clear_graph():
    computation_graph.clear()
                    
def add_fn(a: Tensor, b: Tensor) -> Tensor:
    out_data = a['data'] + b['data']
    requires_grad = a['requires_grad'] or b['requires_grad']
    
    def backward(grad_out: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        grad_a = grad_out if a['requires_grad'] else None
        grad_b = grad_out if b['requires_grad'] else None
        return grad_a, grad_b
    
    out = create_tensor(out_data, requires_grad)
    computation_graph.append({'inputs': [a, b], 'output': out, 'backward': backward})
    return out

def mul_fn(a: Tensor, b: Tensor) -> Tensor:
    out_data = a['data'] * b['data']
    requires_grad = a['requires_grad'] or b['requires_grad']
    
    def backward(grad_out: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        grad_a = grad_out * b['data'] if a['requires_grad'] else None
        grad_b = grad_out * a['data'] if b['requires_grad'] else None
        return grad_a, grad_b
    
    out = create_tensor(out_data, requires_grad)
    computation_graph.append({'inputs': [a, b], 'output': out, 'backward': backward})
    return out

def matmul_fn(a: Tensor, b: Tensor) -> Tensor:
    out_data = np.dot(a['data'], b['data'])
    requires_grad = a['requires_grad'] or b['requires_grad']
    
    def backward(grad_out: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        grad_a = np.dot(grad_out, b['data'].T) if a['requires_grad'] else None
        grad_b = np.dot(a['data'].T, grad_out) if b['requires_grad'] else None
        return grad_a, grad_b
    
    out = create_tensor(out_data, requires_grad)
    computation_graph.append({'inputs': [a, b], 'output': out, 'backward': backward})
    return out
