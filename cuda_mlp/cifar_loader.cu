// cifar_loader.cu

#include "cifar_loader.h"
#include <fstream>
#include <iostream>
#include <cstdint>
#include <cstring>

// Function to load cifar 10 images from binary file
bool load_cifar10_images(const std::string &file_path, CIFAR10Data &data, int num_samples) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "unable to open cifar-10 image file: " <<file_path << std::endl;
        return false;
    }

    // Each image is 3072 floats (32, 32, 3)
    size_t image_size = 32 * 32 * 3;
    size_t total_floats = static_cast<size_t>(num_samples) * image_size;
    data.images.resize(total_floats);

    // Read the entire file content
    file.read(reinterpret_cast<char*>(data.images.data()), total_floats * sizeof(float));
    if (!file) {
        std::cerr << "Error reading cifar-10 images file: " << file_path << std::endl;
        return false;
    }

    file.close();
    return true;
}

// Function to load cifar-10 labels from binary file
bool load_cifar10_labels(const std::string &file_path, CIFAR10Data &data, int num_samples){
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "unable to open cifar10 labels file: " <<file_path << std::endl;
        return false;
    }

    // Each label is a int32
    size_t label_size = sizeof(int32_t);
    data.labels.resize(static_cast<size_t>(num_samples));

    // Read the entire file content
    file.read(reinterpret_cast<char*>(data.labels.data()), num_samples * label_size);
    if (!file) {
        std::cerr << "Error reading cifar10 labels file: " << file_path << std::endl;
        return false;
    }

    file.close();
    return true;
}

