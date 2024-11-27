import numpy as np
from time import time

from engine import create_tensor, backward, clear_graph, computation_graph
from nn import init_mlp, mlp, cross_entropy_with_logits_loss_fn, sgd, zero_grad, relu_fn

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

    X = create_tensor(X.astype(np.float32), requires_grad=False)
    y = create_tensor(y.astype(np.float32), requires_grad=False)

    return X, y

# Hyperparameters
input_size = 3
hidden_size = 64
output_size = 3
learning_rate = 0.0001
epochs = 1000

# Initialize MLP network
arch = [
    (input_size, hidden_size, relu_fn),  # Input to first hidden layer
    (hidden_size, hidden_size, relu_fn),  # Hidden to hidden layer
    (hidden_size, output_size, None),      # Hidden to output layer
]
mlp_params = init_mlp(arch)

# Total parameters
total_params = sum(w['data'].size + b['data'].size for w, b, _ in mlp_params)
print(f'Total parameters: {total_params}')

# Generate test data
test_X, test_y = gen_data(seed=999)

# Training
start = time()
for epoch in range(1, epochs + 1):
    # 1. Clear the computation graph at the start of each epoch
    clear_graph()
    
    # 2. Generate data for this epoch
    X, y = gen_data()
    
    # 3. Forward pass
    preds = mlp(X, mlp_params)
    
    # 4. Loss calculation
    loss = cross_entropy_with_logits_loss_fn(preds, y)
    
    # 5. Backward pass
    try:
        backward(loss)
    except KeyError as e:
        print(f"Backward Pass Error: {e}")
        break  
    
    # 6. Optimization step
    all_params = [param for layer in mlp_params for param in layer[:2]]
    sgd(all_params, lr=learning_rate)
    
    # 7. Zero gradients
    zero_grad(all_params)
    
    # 8. Logging
    if epoch % 100 == 0 or epoch == 1:
        print(f'Epoch {epoch}, Loss: {loss["data"]:.4f}')

print(f'Total time elapsed: {time() - start:.2f} seconds')

# Testing
test_preds = mlp(test_X, mlp_params)
shifted_logits = test_preds['data'] - np.max(test_preds['data'], axis=1, keepdims=True)
exp_logits = np.exp(shifted_logits)
softmax = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

predicted_classes = np.argmax(softmax, axis=1)
true_classes = np.argmax(test_y['data'], axis=1)

accuracy = np.mean(predicted_classes == true_classes)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Display sample predictions
for i in range(5):
    print(f'Sample {i+1}: True Class: {true_classes[i]}, '
          f'Predicted Class: {predicted_classes[i]}, '
          f'Probabilities: {softmax[i]}')
