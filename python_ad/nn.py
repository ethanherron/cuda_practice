import numpy as np
from typing import Callable, List, Tuple, Optional

from engine import Tensor, computation_graph, create_tensor, add_fn, mul_fn, matmul_fn

def relu_fn(a: Tensor) -> Tensor:
    out_data = np.maximum(0, a['data'])
    requires_grad = a['requires_grad']
    
    def backward(grad_out: np.ndarray) -> Tuple[np.ndarray]:
        grad_a = grad_out * (a['data'] > 0) if a['requires_grad'] else None
        return (grad_a,)
    
    out = create_tensor(out_data, requires_grad)
    computation_graph.append({'inputs': [a], 'output': out, 'backward': backward})
    return out

def cross_entropy_with_logits_loss_fn(logits: Tensor, targets: Tensor) -> Tensor:
    shift_logits = logits['data'] - np.max(logits['data'], axis=1, keepdims=True)
    exp_logits = np.exp(shift_logits)
    softmax = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    log_softmax = np.log(softmax + 1e-10)
    loss_data = -np.sum(targets['data'] * log_softmax, axis=1)
    mean_loss = np.mean(loss_data)
    
    requires_grad = logits['requires_grad']
    
    def backward(grad_out: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        grad_logits = (softmax - targets['data']) / targets['data'].shape[0]
        grad_targets = None
        return grad_logits, grad_targets
    
    out = create_tensor(np.array(mean_loss, dtype=np.float32), requires_grad=True)
    computation_graph.append({'inputs': [logits, targets], 'output': out, 'backward': backward})
    return out

def mse_loss_fn(preds: Tensor, targets: Tensor) -> Tensor:
    sq_diff = (preds['data'] - targets['data']) ** 2
    mean_loss = np.mean(sq_diff)
    requires_grad = preds['requires_grad']
    
    def backward(grad_out: np.ndarray) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        grad_preds = (2 * (preds['data'] - targets['data'])) / targets['data'].shape[0]
        grad_targets = None
        return grad_preds, grad_targets
    
    out = create_tensor(np.array(mean_loss, dtype=np.float32), requires_grad=True)
    computation_graph.append({'inputs': [preds, targets], 'output': out, 'backward': backward})
    return out

def sgd(parameters: List[Tensor], lr: float) -> None:
    for param in parameters:
        if param['requires_grad'] and param['grad'] is not None:
            grad = param['grad']
            if param['data'].shape != grad.shape:
                grad = np.sum(grad, axis=0, keepdims=True)
            param['data'] -= lr * grad

def zero_grad(parameters: List[Tensor]) -> None:
    for param in parameters:
        if param['requires_grad']:
            param['grad'] = None

def init_linear_layer(input_dim: int, output_dim: int) -> Tuple[Tensor, Tensor]:
    w = create_tensor(
        np.random.randn(input_dim, output_dim) * np.sqrt(2. / input_dim), requires_grad=True
    )
    b = create_tensor(
        np.zeros((1, output_dim)), requires_grad=True
    )
    return w, b

def linear_layer(input: Tensor, weights: Tensor, bias: Tensor, activation: Optional[Callable] = None) -> Tensor:
    linear_output = add_fn(matmul_fn(input, weights), bias)
    if activation:
        activated_output = activation(linear_output)
        return activated_output
    return linear_output

def init_mlp(layers: List[Tuple[int, int, Optional[Callable]]]) -> List[Tuple[Tensor, Tensor, Optional[Callable]]]:
    mlp = []
    for input_dim, output_dim, activation in layers:
        w, b = init_linear_layer(input_dim, output_dim)
        mlp.append((w, b, activation))
    return mlp

def mlp(input: Tensor, arch: List[Tuple[int, int, Optional[Callable]]]) -> Tensor:
    output = input
    for w, b, act in arch:
        output = linear_layer(output, w, b, act)
    return output
