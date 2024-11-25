// main.cu

#include "cifar_loader.h"  // Include the cifar10 data loader
#include "mlp.h"           // Include the mlp
#include "cuda_utils.h"    // include loss function 
#include <iostream>        // for console I/O
#include <cuda_runtime.h>  // for cuda mem allocation
#include <iomanip>         // for std::setprecision
#include <sstream>         // for std::ostringstream

// Function to print progress bar with optional EMA
void printProgressBarWithEMA(int current, int total, float ema_loss, bool update_ema = false, int barWidth = 70) {
    static bool first_call = true;            // Flag to track the first call
    static float last_ema_loss = 0.0f;        // To store the last EMA loss

    // Calculate progress
    float progress = static_cast<float>(current) / total;
    int pos = static_cast<int>(barWidth * progress);
    if (pos >= barWidth) pos = barWidth - 1; // Prevent 'pos' from exceeding 'barWidth'

    // Round the percentage to the nearest integer
    int percentage = static_cast<int>(progress * 100.0 + 0.5);

    // Construct progress bar string
    std::string bar = "[";
    for (int j = 0; j < barWidth; ++j) { // Use 'j' instead of 'i' to avoid shadowing
        if (j < pos)
            bar += "=";
        else if (j == pos)
            bar += ">";
        else
            bar += " ";
    }
    bar += "] " + std::to_string(percentage) + " %";

    // Update EMA loss if required
    if (update_ema) {
        last_ema_loss = ema_loss;
    }

    // Construct EMA string with fixed decimal places
    std::ostringstream ema_ss;
    ema_ss << "EMA Loss: " << std::fixed << std::setprecision(4) << last_ema_loss;
    std::string ema_str = ema_ss.str();

    if (!first_call) {
        // Move cursor up two lines to overwrite previous progress bar and EMA
        std::cout << "\033[2A";
    }

    // Print progress bar
    std::cout << bar << "\n";

    // Print EMA loss
    std::cout << ema_str << "\n";

    // Flush the output to ensure it appears immediately
    std::cout.flush();

    if (first_call) {
        first_call = false; // Update the flag after the first call
    }
}







int main() {
    // 1. init cifar10 data structure
    CIFAR10Data train_data;

    // 2. Define paths to the binary data files
    std::string train_images_path = "/home/edherron/data/cifar10_binary/cifar10_images.bin";
    std::string train_labels_path = "/home/edherron/data/cifar10_binary/cifar10_labels.bin";

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

    // Print out num params in mlp
    size_t total_params = mlp.getTotalParameters();
    std::cout << "Total number of parameters in the model: " << total_params << "\n";

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
    float learning_rate = 0.0003f;
    int num_epochs = 1;

    // init ema variables
    float ema_loss = 0.0f;
    float alpha = 0.1f;

    int ema_batch_size = 25;
    float batch_loss_accumulator = 0.0f;
    int batch_count = 0;
    bool ema_initialized = false;

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

            // accumulate loss for ema
            batch_loss_accumulator += h_loss;
            batch_count++;

            // backward pass
            mlp.backward(d_input, d_output, label);

            // sgd step
            mlp.SGDUpdate(learning_rate);

            // update progress bar
            if (batch_count == ema_batch_size) {
                float average_batch_loss = batch_loss_accumulator / ema_batch_size;

                if (!ema_initialized) {
                    ema_loss = average_batch_loss;
                    ema_initialized = true;
                } else {
                    ema_loss = alpha * average_batch_loss + (1.0f - alpha) * ema_loss;
                }

                printProgressBarWithEMA(i + 1, num_train_samples, ema_loss, true);

                batch_loss_accumulator = 0.0f;
                batch_count = 0;
            } else {
                printProgressBarWithEMA(i + 1, num_train_samples, ema_loss, false);
            }
        }
        // handle any leftovers 
        if (batch_count > 0) {
            float average_batch_loss = batch_loss_accumulator / batch_count;

            // update ema
            if (!ema_initialized) {
                ema_loss = average_batch_loss;
                ema_initialized = true;
            } else {
                ema_loss = alpha * average_batch_loss + (1.0f - alpha) * ema_loss;
            }
            printProgressBarWithEMA(num_train_samples, num_train_samples, ema_loss, true);
        }
        // print new line after each progress bar
        std::cout << std::endl;
        std::cout << "Epoch " << (epoch + 1) << "/" << num_epochs 
                  << " - Average Loss: " << (epoch_loss / num_train_samples) << "\n";
    }


    //  free up allocated memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_loss);

    std::cout << "training completed successfully.\n";

    return 0;

}