
// Header guards
// Header guards prevent multiple inclusions. This ensures that the contents of the header files are not included mulitple times during the compilation process. This also prevents redefinition errors. 
// Mechanism: The #infndef, #define, and #endif directives work together to include the enclosed code only if CIFAR_LOADER_H has not been defined previously. 

#ifndef CIFAR_LOADER_H
#define CIFAR_LOADER_H


// Import libraries

#include <vector>
#include <string>

// Structure to hold CIFAR-10 data
struct CIFAR10Data {
    std::vector<float> images; // Flattened and normalized images
    std::vector<int> labels;   // Corresponding labels
};

// Function to load CIFAR-10 images from binary file
bool load_cifar10_images(const std::string &file_path, CIFAR10Data &data, int num_samples);

bool load_cifar10_labels(const std::string &file_path, CIFAR10Data &data, int num_samples);


#endif // CIFAR_LOADER_H