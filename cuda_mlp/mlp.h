// mlp.h

#ifndef MLP_H
#define MLP_H
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

// Activation function
enum ActivationType {
    RELU,
    SIGMOID
};

// Structure to hold the mlp params
struct MLPParameters {
    // weight matrices
    float* d_weights_input_hidden;  // weights from input to hidden layer
    float* d_weights_hidden_output; // weight matrix from hidden to output layer

    // bias vector
    float* d_bias_hidden;           // bias for hidden layer
    float* d_bias_output;           // bias for output layer

    // gradients
    float* d_grad_weights_input_hidden;  // grads for input to hidden layer
    float* d_grad_weights_hidden_output; // grads for hidden to output layer
    float* d_grad_bias_hidden;           // grads for hidden bias
    float* d_grad_bias_output;           // grads for output bias

    // dims
    int input_size;
    int hidden_size;
    int output_size;
};

// MLP class
class MLP {
public:
    // constructor
    MLP(int input_size, int hidden_size, int output_size, ActivationType activation=RELU);

    // destructor
    ~MLP();

    // init weights and biases
    void initializer();

    // forward pass
    // input: device pointer to input data
    // output: device pointer to store output data
    void forward(float* d_input, float* d_output);

    // backward pass
    // input: device pointer to input data
    // output: device pointer to output data?? im not sure about these args
    void backward(float* d_input, float* d_output, int label);

    // sgd update function to apply grads
    void SGDUpdate(float learning_rate);

    // calc total num params in mlp
    size_t getTotalParameters() const;

private:
    MLPParameters params;
    ActivationType activation;

    // intermediate storage for activations and grads
    float* d_hidden;        // hidden layer activations
    float* d_output_grad;   // grad of the loss wrt output layer
    float* d_hidden_grad;   // grad of the loss wrt hidden layer

    // CUDA kernels
    void launch_matrix_multiply(float* d_A, float* d_B, float* d_C, int M, int N, int k);
    void launch_add_bias(float* d_vector, float* d_bias, int size);
    void launch_activation(float* d_vector, int size);
};

#endif // MLP_H