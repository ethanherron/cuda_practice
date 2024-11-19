// cuda_utils.h

#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda_runtime.h>
#include <iostream> // for error output

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

// activation function kernel declarations
__global__ void relu(float* data, int size);
__global__ void sigmoid(float* data, int size);

// bias kernel
__global__ void addBias(float* vector, float* bias, int size);

// mat mul function declaration
void matrixMultiplyCUDA(float* d_A, float* d_B, float* d_C, int M, int N, int K);

// ce w/ logits declaration
__global__ void cross_entropy_with_logits(float* logits, int label, float* loss, int num_classes);
void crossEntropyWithLogitsCUDA(float* logits, int label, float* loss, int num_classes);

// grad comp kernels
__global__ void computeWeightGradients(float* d_input, float* d_grad_out, float* d_grad_weights, int M, int K, int N);
__global__ void computeBiasGradients(float* d_grad_out, float* d_grad_bias, int N);
__global__ void computeInputGradients(float* d_grad_out, float* d_weights, float* d_grad_input, int M, int K, int N);
__global__ void cross_entropy_with_logits_backward(float* logits, int label, float* grad, int num_classes);

// sgd function declaration
void sgdUpdateCUDA(float* d_params, float* d_grads, float learning_rate, int size);

#endif 