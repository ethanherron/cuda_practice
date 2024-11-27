# train.py

import numpy as np
from time import time

from engine import Variable
from network import MLP
from utils import CrossEntropyWithLogitsLoss, SGD, Adam, ScheduleFreeAdamW

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
hidden_size = 16  
output_size = 3   
num_layers = 1    

mlp = MLP(input_size, output_size, hidden_size, num_layers)
print('Total parameters:', sum(param.data.size for param in mlp.parameters()))


loss_fn = CrossEntropyWithLogitsLoss()

learning_rate = 0.0001 
optimizer = SGD(mlp.parameters(), learning_rate)
# optimizer = Adam(mlp.parameters(), learning_rate)
# optimizer = ScheduleFreeAdamW(mlp.parameters(), learning_rate)

test_X_var, test_y_var = gen_data(seed=999)  


epochs = 500 
start = time()
for epoch in range(1, epochs + 1):
    X_var, y_var = gen_data()
    
    optimizer.zero_grad()
    
    logits = mlp(X_var) 
    
    loss = loss_fn.apply(logits, y_var)
    
    loss.backward()
    
    optimizer.step()
    
    if epoch % 100 == 0 or epoch == 1:
        print(f'Epoch {epoch}, Loss: {loss.data:.4f}')

print(f'Total time elapsed: {time() - start:.2f} seconds')
print("\nTesting after training:")

test_logits = mlp(test_X_var)

shifted_logits = test_logits.data - np.max(test_logits.data, axis=1, keepdims=True)  
exp_logits = np.exp(shifted_logits)
softmax = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

predicted_classes = np.argmax(softmax, axis=1)
true_classes = np.argmax(test_y_var.data, axis=1)

accuracy = np.mean(predicted_classes == true_classes)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Optional: Display Sample Predictions
for i in range(5):
    print(f'Sample {i+1}: True Class: {true_classes[i]}, '
          f'Predicted Class: {predicted_classes[i]}, '
          f'Probabilities: {softmax[i]}')
