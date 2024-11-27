# train.py

import numpy as np
from numba import cuda
from timeit import timeit

from engine import Variable
from network import MLP
from utils import CrossEntropyWithLogitsLoss, SGD

# numba eval funcs
@njit(parallel=True)
def eval_sofmax_gpu(logits: np.ndarray) -> np.ndarray:
    i = cuda.grad(1)
    if i < logtis.shape[0]:
        num_classes = logits.shape[1]
        max_val = logits[i, 0]
        for j in range(1, num_classes):
            if logits[i, j] > max_val:
                max_val = logits[i, j]
        sum_exp = 0.
        for j in range(num_classes):
            softmax[i, j] = math.exp(logits[i, j] - max_val)
            sum_exp += sofmax[i, j]
        for j in range(num_classes):
            softmax[i, j] /= sum_exp
    return softmax

@njit(parallel=True)
def compute_preds(softmax: np.ndarray) -> np.ndarray:
    num_samples, num_classes = softmax.shape
    preds = np.empty(num_samples, dtype=np.int32)
    for i in prange(num_samples):
        max_idx = 0
        max_val = softmax[i, 0]
        for j in range(1, num_classes):
            if softmax[i, j] > max_val:
                max_val = softmax[i, j]
                max_idx = j
        preds[i] = max_idx  
    return preds


@njit(parallel=True)
def compute_acc(preds: np.ndarray, true: np.ndarray) -> float:
    num_samples = preds.shape[0]
    correct = 0
    for i in prange(num_samples):
        if preds[i] == true[i]:
            correct += 1
    return correct / num_samples

def gen_data(num_classes=3, features=3, samples_per_class=50, seed=None):
    if seed is not None:
        np.random.seed(seed)
    X = []
    y = []

    centers = [-5, 0, 5]

    for class_idx in range(num_classes):
        center = np.zeros(features)
        center[0] = centers[class_idx]
        cov = np.eye(features)
        class_samples = np.random.multivariate_normal(center, cov, samples_per_class)
        X.append(class_samples)

        labels = np.zeros(num_classes)
        labels[class_idx] = 1
        y.append(np.tile(labels, (samples_per_class, 1)))

    X = np.vstack(X)  # (num_classes * samples_per_class, features)
    y = np.vstack(y)  # (num_classes * samples_per_class, num_classes)

    X = Variable(X, requires_grad=False)
    y = Variable(y, requires_grad=False)

    return X, y

input_size = 3    
hidden_size = 512   
output_size = 3   
num_layers = 4    

mlp = MLP(input_size, output_size, hidden_size, num_layers)
print('Total parameters:', sum(param.data.size for param in mlp.parameters()))


loss_fn = CrossEntropyWithLogitsLoss()

learning_rate = 0.0001 
optimizer = SGD(mlp.parameters(), learning_rate)

test_X_var, test_y_var = gen_data(seed=999)  


epochs = 500
start = timeit.time()
for epoch in range(1, epochs + 1):
    X_var, y_var = gen_data()
    
    optimizer.zero_grad()
    
    logits = mlp(X_var) 
    
    loss = loss_fn.apply(logits, y_var)
    
    loss.backward()
    
    optimizer.step()
    
    if epoch % 100 == 0 or epoch == 1:
        print(f'Epoch {epoch}, Loss: {loss.data:.4f}')

print(f'total time elapsed {timeit.time() - start}')
print("\nTesting after training:")

test_logits = mlp(test_X_var)

# Convert logits and targets to NumPy arrays for Numba functions
logits_data = test_logits.data
targets_data = test_y_var.data

# Compute softmax probabilities using Numba-accelerated function
softmax = eval_sofmax_gpu(logits_data)

# Compute predicted classes using Numba-accelerated function
predicted_classes = compute_preds(softmax)

# Compute true class indices
true_classes = np.argmax(targets_data, axis=1).astype(np.int32)

# Compute accuracy using Numba-accelerated function
accuracy = compute_acc(predicted_classes, true_classes)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Optional: Display Sample Predictions
for i in range(5):
    print(f'Sample {i+1}: True Class: {true_classes[i]}, '
          f'Predicted Class: {predicted_classes[i]}, '
          f'Probabilities: {softmax[i]}')
