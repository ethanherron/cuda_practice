// main.cu

#include "cifar_loader.h"  // Include the cifar10 data loader
#include "mlp.h"           // Include the mlp
#include "cuda_utils.h"    // include loss function 
#include <iostream>        // for console I/O
#include <cuda_runtime.h>  // for cuda mem allocation

// progress bar
void printProgressBar(int current, int total, int barWidth = 70) {
    float progress = static_cast<float>(current) / total;
    int pos = static_cast<int>(barWidth * progress);

    std::cout << "[";
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos)
            std::cout << "=";
        else if (i == pos)
            std::cout << ">";
        else
            std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << " %\r";
    std::cout.flush();
}

int main() {
    // 1. init cifar10 data structure
    CIFAR10Data train_data;

    // 2. Define paths to the binary data files
    std::string train_images_path = "/data/cifar10/cifar10_binary/cifar10_images.bin";
    std::string train_labels_path = "/data/cifar10/cifar10_binary/cifar10_labels.bin";

    // 3. Specify the number of training samples
    int num_train_samples = 50000; // cifar 10 train dataset size

    // 4. load training images
    std::cout << "loading cifar10 training images...\n";
    bool images_loaded = load_cifar10_images(train_images_path, train_data, num_train_samples);
    if (!images_loaded) {
        std::cerr << "failed to load cifar10 training images.\n";
        return EXIT_FAILURE;
    }
    std::cout << "successfully loaded training images" << train_data.images.size() / (32 * 32 * 3) << "training images.\n\n";

    // 5. Load cifar10 labels
    std::cout << "loading cifar10 training labels... \n";
    bool labels_loaded = load_cifar10_labels(train_labels_path, train_data, num_train_samples);
    if (!labels_loaded) {
        std::cerr << "failed to load cifar10 training labels.\n";
        return EXIT_FAILURE;
    } 
    std::cout << "successfully loaded training labels" << train_data.labels.size() << " training labels. \n\n";

    // 6. Verify loaded data
    std::cout << "verifying loaded data...\n";

    // 6.1 print first 5 labels
    std::cout << "first 5 sample labels: ";
    for(int i = 0; i < 5 && i < train_data.labels.size(); ++i) {
        std::cout << train_data.labels[i] << " ";
    }
    std::cout << "\n";

    // 6.2 print the first 10 pixel values of the first image
    std::cout << "first 10 pixel values of the first image: ";
    for(int i = 0; i < 10 && i < train_data.images.size(); ++i) {
        std::cout << train_data.images[i] << " ";
    }
    std::cout << "\n\n";

    // print total number of pixels loaded
    std::cout << "total number of pixels loaded: " <<train_data.images.size() << "\n\n";

    // completed
    std::cout << "Data loading and verification completed successfully.\n";

    // 7. init the mlp with cifar10 input size, hidden layer size and output size
    int input_size = 3072;   // cifar 10 image flattened 32*32*3
    int hidden_size = 128;   // hidden layer size in mlp
    int output_size = 10;    // cifar 10 classes
    
    MLP mlp(input_size, hidden_size, output_size);
    mlp.initializer();

    // 8. allocate device memory for input and output
    float* d_input;
    float* d_output;
    float* d_loss;
    cudaMalloc(&d_input, input_size * sizeof(float));
    cudaMalloc(&d_output, output_size * sizeof(float));
    cudaMalloc(&d_loss, sizeof(float));  // just for a single loss value

    // 9. Copy the first image from train data to d_input as the test input
    cudaMemcpy(d_input, train_data.images.data(), input_size * sizeof(float), cudaMemcpyHostToDevice);

    // 10. run the forward pass
    mlp.forward(d_input, d_output);

    // 11. compute loss over logits from forward pass
    int label = train_data.labels[0]; // use the first images label for testing
    crossEntropyWithLogitsCUDA(d_output, label, d_loss, output_size);

    // 12. retrieve and print output of forward pass and loss function
    std::vector<float> output(output_size);
    cudaMemcpy(output.data(), d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "forward pass output: \n";
    for (int i = 0; i < output_size; ++i) {
        std::cout << "class " << i << ": " << output[i] << "\n";
    }

    float h_loss;
    cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Cross-entropy loss: " << h_loss << std::endl;


    // training params
    float learning_rate = 0.01f;
    int num_epochs = 1;

    // training loop
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        float epoch_loss = 0.0f;

        for (int i = 0; i < num_train_samples; ++i) {
            // copy input image to device
            CUDA_CHECK(cudaMemcpy(d_input, &train_data.images[i * input_size], input_size * sizeof(float), cudaMemcpyHostToDevice));
            int label = train_data.labels[i];

            // forward pass
            mlp.forward(d_input, d_output);

            // compute loss
            crossEntropyWithLogitsCUDA(d_output, label, d_loss, output_size);

            // get loss
            float h_loss;
            CUDA_CHECK(cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
            epoch_loss += h_loss;

            // backward pass
            mlp.backward(d_input, d_output, label);

            // sgd step
            mlp.SGDUpdate(learning_rate);

            // update progress bar
            printProgressBar(i + 1, num_train_samples);
            // if ((i + 1) % 10 == 0 || (i + 1) == num_train_samples) {
            //     printProgressBar(i + 1, num_train_samples);
            // }
        }
        // print new line after each progress bar
        std::cout << std::endl;
        // print out epoch loss
        std::cout << "Epoch " << (epoch + 1) << "/" << num_epochs << " - Loss: " << (epoch_loss / num_train_samples) << "\n";
    }


    //  free up allocated memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_loss);

    std::cout << "training completed successfully.\n";

    return 0;

}