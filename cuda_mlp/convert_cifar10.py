# convert_cifar10.py

import os
import pickle
import numpy as np

def load_cifar10_batch(batch_file):
    with open(batch_file, 'rb') as f:
        dict = pickle.load(f, encoding='bytes')
        # Decode bytes to strings
        data = dict[b'data']  # Shape: (10000, 3072)
        labels = dict[b'labels']  # List of length 10000
        return data, labels

def convert_to_binary(data_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    all_data = []
    all_labels = []
    
    # Load all training batches
    for i in range(1, 6):
        batch_file = os.path.join(data_dir, f'data_batch_{i}')
        data, labels = load_cifar10_batch(batch_file)
        all_data.append(data)
        all_labels.extend(labels)
    
    # Convert to numpy arrays
    all_data = np.vstack(all_data)  # Shape: (50000, 3072)
    all_labels = np.array(all_labels, dtype=np.int32)  # Shape: (50000,)
    
    # Normalize pixel values to [0,1]
    all_data = all_data.astype(np.float32) / 255.0
    
    # Save images
    images_file = os.path.join(output_dir, 'cifar10_images.bin')
    all_data.tofile(images_file)
    
    # Save labels
    labels_file = os.path.join(output_dir, 'cifar10_labels.bin')
    all_labels.tofile(labels_file)
    
    print(f"Saved {all_data.shape[0]} images to {images_file}")
    print(f"Saved {all_labels.shape[0]} labels to {labels_file}")
    
    # Similarly, process test batch
    test_batch_file = os.path.join(data_dir, 'test_batch')
    test_data, test_labels = load_cifar10_batch(test_batch_file)
    test_data = test_data.astype(np.float32) / 255.0
    test_labels = np.array(test_labels, dtype=np.int32)
    
    # Save test images
    test_images_file = os.path.join(output_dir, 'cifar10_test_images.bin')
    test_data.tofile(test_images_file)
    
    # Save test labels
    test_labels_file = os.path.join(output_dir, 'cifar10_test_labels.bin')
    test_labels.tofile(test_labels_file)
    
    print(f"Saved {test_data.shape[0]} test images to {test_images_file}")
    print(f"Saved {test_labels.shape[0]} test labels to {test_labels_file}")

if __name__ == "__main__":
    # Adjust these paths as necessary
    data_dir = '/data/cifar10/cifar-10-batches-py'  # Directory containing CIFAR-10 batches
    output_dir = '/data/cifar10/cifar10_binary'
    convert_to_binary(data_dir, output_dir)
