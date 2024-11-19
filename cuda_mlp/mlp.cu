// mlp.cu

#include "mlp.h"
#include "cuda_utils.h"
#include <cstdlib>
#include <ctime>
#include <iostream>

// cuda error checking macro
#define CUDA_CHECK(call)                                                 \
    do {                                                                 \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess) {                                        \
            std::cerr << "cuda error in " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl;  \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    } while (0)            


// constructor
MLP::MLP(int input_size, int hidden_size, int output_size, ActivationType activation_type)
    : activation(activation_type)
{
    params.input_size = input_size;
    params.hidden_size = hidden_size;
    params.output_size = output_size;

    // init pointer to nullptr for weights
    params.d_weights_input_hidden = nullptr;
    params.d_weights_hidden_output = nullptr;
    params.d_bias_hidden = nullptr;
    params.d_bias_output = nullptr;

    // init pointer to nullptr for grads
    params.d_grad_weights_input_hidden = nullptr;
    params.d_grad_weights_hidden_output = nullptr;
    params.d_grad_bias_hidden = nullptr;
    params.d_grad_bias_output = nullptr;
}

// destructor
MLP::~MLP() {
    // free up gpu memory
    if (params.d_weights_input_hidden) cudaFree(params.d_weights_input_hidden);
    if (params.d_weights_hidden_output) cudaFree(params.d_weights_hidden_output);
    if (params.d_bias_hidden) cudaFree(params.d_bias_hidden);
    if (params.d_bias_output) cudaFree(params.d_bias_output);

    if (params.d_grad_weights_input_hidden) cudaFree(params.d_grad_weights_input_hidden);
    if (params.d_grad_weights_hidden_output) cudaFree(params.d_grad_weights_hidden_output);
    if (params.d_grad_bias_hidden) cudaFree(params.d_grad_bias_hidden);
    if (params.d_grad_bias_output) cudaFree(params.d_grad_bias_output);

    if (d_hidden) cudaFree(d_hidden);
    if (d_output_grad) cudaFree(d_output_grad);
    if (d_hidden_grad) cudaFree(d_hidden_grad);
}

// init weights and biases
void MLP::initializer() {
    // seed for random number generation
    srand(static_cast<unsigned int>(time(0)));

    // allocate gpu memory for weights and biases
    size_t size_input_hidden = params.input_size * params.hidden_size * sizeof(float);
    size_t size_hidden_output = params.hidden_size * params.output_size * sizeof(float);
    size_t size_bias_hidden = params.hidden_size * sizeof(float);
    size_t size_bias_output = params.output_size * sizeof(float);

    CUDA_CHECK(cudaMalloc(&params.d_weights_input_hidden, size_input_hidden));
    CUDA_CHECK(cudaMalloc(&params.d_weights_hidden_output, size_hidden_output));
    CUDA_CHECK(cudaMalloc(&params.d_bias_hidden, size_bias_hidden));
    CUDA_CHECK(cudaMalloc(&params.d_bias_output, size_bias_output));

    // allocate gpu memory for hidden activations and grads
    size_t size_grad_input_hidden = params.input_size * params.hidden_size * sizeof(float);
    size_t size_grad_hidden_output = params.hidden_size * params.output_size * sizeof(float);
    size_t size_grad_bias_hidden = params.hidden_size * sizeof(float);
    size_t size_grad_bias_output = params.output_size * sizeof(float);

    CUDA_CHECK(cudaMalloc(&params.d_grad_weights_input_hidden, size_grad_input_hidden));
    CUDA_CHECK(cudaMalloc(&params.d_grad_weights_hidden_output, size_grad_hidden_output));
    CUDA_CHECK(cudaMalloc(&params.d_grad_bias_hidden, size_grad_bias_hidden));
    CUDA_CHECK(cudaMalloc(&params.d_grad_bias_output, size_grad_bias_output));

    // init gradient buffers to zero
    CUDA_CHECK(cudaMemset(params.d_grad_weights_input_hidden, 0, size_grad_input_hidden));
    CUDA_CHECK(cudaMemset(params.d_grad_weights_hidden_output, 0, size_grad_hidden_output));
    CUDA_CHECK(cudaMemset(params.d_grad_bias_hidden, 0, size_grad_bias_hidden));
    CUDA_CHECK(cudaMemset(params.d_grad_bias_output, 0, size_grad_bias_output));

    // allocate gpu mem for intermediate activations and grads
    size_t size_hidden = params.hidden_size * sizeof(float);
    size_t size_output_grad = params.output_size * sizeof(float);
    size_t size_hidden_grad = params.hidden_size * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_hidden, size_hidden));
    CUDA_CHECK(cudaMalloc(&d_output_grad, size_output_grad));
    CUDA_CHECK(cudaMalloc(&d_hidden_grad, size_hidden_grad));

    // init weights with small random values
    std::vector<float> h_weights_input_hidden(params.input_size * params.hidden_size);
    std::vector<float> h_weights_hidden_output(params.hidden_size * params.output_size);
    std::vector<float> h_bias_hidden(params.hidden_size, 0.0f);
    std::vector<float> h_bias_output(params.output_size, 0.0f);

    // random weights init with xavier init
    float limit_input_hidden = sqrtf(6.0f / (params.input_size + params.hidden_size));
    float limit_hidden_output = sqrtf(6.0f / (params.hidden_size + params.output_size));

    for(auto &w : h_weights_input_hidden) {
        w = ((float)rand() / RAND_MAX) * 2 * limit_input_hidden - limit_input_hidden;
    }
    for(auto &w : h_weights_hidden_output) {
        w = ((float)rand() / RAND_MAX) * 2 * limit_hidden_output - limit_hidden_output;
    }

    // copy ws and bs to gpu
    CUDA_CHECK(cudaMemcpy(params.d_weights_input_hidden, h_weights_input_hidden.data(), size_input_hidden, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(params.d_weights_hidden_output, h_weights_hidden_output.data(), size_hidden_output, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(params.d_bias_hidden, h_bias_hidden.data(), size_bias_hidden, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(params.d_bias_output, h_bias_output.data(), size_bias_output, cudaMemcpyHostToDevice));
}

// forward pass
void MLP::forward(float* d_input, float* d_output) {
    // 1. input to hidden layer
    // hidden = relu(w_in * input + b_in)
    // matmul: w_in (hidden_size, input_size) * input (input_size, 1) = hidden (hidden_size, 1)
    launch_matrix_multiply(params.d_weights_input_hidden, d_input, d_hidden, params.hidden_size, 1, params.input_size);

    // add bias to hidden 
    launch_add_bias(d_hidden, params.d_bias_hidden, params.hidden_size);

    // activation
    launch_activation(d_hidden, params.hidden_size);

    // 2. hidden to output layer
    // output = softmax(w_out * hidden_act + b_out)
    // ignore softmax here, well put that in loss function, but thats what the final layer will actually be
    // matmul: w_out (outpu_size, hidden_size) * hidden (hidden_size, output_size) = output (prelogits) (logits/output_size, 1)
    launch_matrix_multiply(params.d_weights_hidden_output, d_hidden, d_output, params.output_size, 1, params.hidden_size);

    // add bias to output layer
    launch_add_bias(d_output, params.d_bias_output, params.output_size);
}

// mat mul kernel with cuda utils
void MLP::launch_matrix_multiply(float* d_A, float* d_B, float* d_C, int M, int N, int K) {
    // A: M, K
    // B: K, N
    // C: M, N
    // use a simple cuda mat mul from cuda_utils.cu

    matrixMultiplyCUDA(d_A, d_B, d_C, M, N, K);
}

// add bias kernel
void MLP::launch_add_bias(float* d_vector, float* d_bias, int size) {
    // launch a kernel to add bias to each element
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    addBias<<<blocks, threads>>>(d_vector, d_bias, size);
    CUDA_CHECK(cudaDeviceSynchronize());
}

// activation function kernel
void MLP::launch_activation(float* d_vector, int size) {
    // launch kernel to apply relu on hidden layer
    int threads = 256;
    int blocks = (size + threads - 1) / threads;

    if (activation == RELU) {
        relu<<<blocks, threads>>>(d_vector, size);
    }
    else if (activation == SIGMOID) {
        sigmoid<<<blocks, threads>>>(d_vector, size);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

// mlp backward
void MLP::backward(float* d_input, float* d_output, int label) {
    // 1. Compute the gradient of the loss with respect to the output layer logits
    cross_entropy_with_logits_backward<<<1, 1>>>(d_output, label, d_output_grad, params.output_size);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 2. Compute gradients for weights and biases of the output layer
    // Gradient for weights: d_weights_hidden_output = d_output_grad * d_hidden^T
    computeWeightGradients<<<dim3((params.output_size + 15) / 16, (params.hidden_size + 15) / 16), dim3(16, 16)>>>(
        d_output_grad, d_hidden, params.d_grad_weights_hidden_output, params.output_size, 1, params.hidden_size);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Gradient for biases: d_bias_output = d_output_grad
    computeBiasGradients<<<(params.output_size + 255) / 256, 256>>>(d_output_grad, params.d_grad_bias_output, params.output_size);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 3. Backpropagate gradients to the hidden layer
    // d_hidden_grad = d_output_grad * w_hidden_output (transpose)
    computeInputGradients<<<dim3((params.hidden_size + 15) / 16, (params.input_size + 15) / 16), dim3(16, 16)>>>(
        d_output_grad, params.d_weights_hidden_output, d_hidden_grad, params.hidden_size, params.input_size, params.output_size);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 4. Compute gradients for weights and biases of the hidden layer
    // Gradient for weights: d_weights_input_hidden = d_hidden_grad * d_input^T
    computeWeightGradients<<<dim3((params.hidden_size + 15) / 16, (params.input_size + 15) / 16), dim3(16, 16)>>>(
        d_hidden_grad, d_input, params.d_grad_weights_input_hidden, params.hidden_size, 1, params.input_size);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Gradient for biases: d_bias_hidden = d_hidden_grad
    computeBiasGradients<<<(params.hidden_size + 255) / 256, 256>>>(d_hidden_grad, params.d_grad_bias_hidden, params.hidden_size);
    CUDA_CHECK(cudaDeviceSynchronize());
}


// SGD update for mlp
void MLP::SGDUpdate(float learning_rate) {
    // total size for each layer
    int input_hidden_size = params.input_size * params.hidden_size;
    int hidden_output_size = params.hidden_size * params.output_size;

    // update weights and biases from hidden layer to output
    sgdUpdateCUDA(params.d_weights_hidden_output, params.d_grad_weights_hidden_output, learning_rate, hidden_output_size);
    sgdUpdateCUDA(params.d_bias_output, params.d_grad_bias_output, learning_rate, params.output_size);

    // update weights and biases from input to hidden_layer
    sgdUpdateCUDA(params.d_weights_input_hidden, params.d_grad_weights_input_hidden, learning_rate, input_hidden_size);
    sgdUpdateCUDA(params.d_bias_hidden, params.d_grad_bias_hidden, learning_rate, params.hidden_size);

    // reset grad buffers to zero for next iter
    size_t size_grad_input_hidden = params.input_size * params.hidden_size * sizeof(float);
    size_t size_grad_hidden_output = params.hidden_size * params.output_size * sizeof(float);
    size_t size_grad_bias_hidden = params.hidden_size * sizeof(float);
    size_t size_grad_bias_output = params.output_size * sizeof(float);

    CUDA_CHECK(cudaMemset(params.d_grad_weights_input_hidden, 0, size_grad_input_hidden));
    CUDA_CHECK(cudaMemset(params.d_grad_weights_hidden_output, 0, size_grad_hidden_output));
    CUDA_CHECK(cudaMemset(params.d_grad_bias_hidden, 0, size_grad_bias_hidden));
    CUDA_CHECK(cudaMemset(params.d_grad_bias_output, 0, size_grad_bias_output));

}