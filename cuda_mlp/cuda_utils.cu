// cuda_utils.cu

#include "cuda_utils.h"
#include <cmath> // for expf function in sigmoid
#include <cstdio>



// relu activation kernel
__global__ void relu(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}

// sigmoid activation kernel
__global__ void sigmoid(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = 1.0f / (1.0f + expf(-data[idx]));
    }
}

// bias kernel
__global__ void addBias(float* vector, float* bias, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        vector[idx] += bias[idx];
    }
}

// mat mul kernel
__global__ void matrixMultiplyKernel(float* A, float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

// host function to launch the mat mul kernel
void matrixMultiplyCUDA(float* d_A, float* d_B, float* d_C, int M, int N, int K) {
    // configure a 16x16 block size for mat mul
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);
    
    // launch kernel
    matrixMultiplyKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);

    // synchronize to check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error in matrixMultiplyCUDA: %s\n", cudaGetErrorString(err));
    }
}

// kernel for cross-entropy with logits
__global__ void cross_entropy_with_logits(float* logits, int label, float* loss, int num_classes) {
    // find the max logit for numerical stability
    float max_logit = logits[0];
    for (int j = 1; j < num_classes; ++j) {
        max_logit = fmaxf(max_logit, logits[j]);
    }

    // compute the sum of exp logits (softmax denominator)
    float sum_exp = 0.0f;
    for (int j = 0; j < num_classes; ++j) {
        logits[j] = expf(logits[j] - max_logit); // subtract max logit for stability
        sum_exp += logits[j];
    }

    // calc cross-entropy loss for the true class
    float log_prob = logf(logits[label] / sum_exp);
    *loss = -log_prob;
}

// host function to launch the cross entropy with logits kernel
void crossEntropyWithLogitsCUDA(float* logits, int label, float* loss, int num_classes) {
    // launch cewl kernel with a single block and thread for this single input
    cross_entropy_with_logits<<<1, 1>>>(logits, label, loss, num_classes);
    CUDA_CHECK(cudaDeviceSynchronize());
}

// kernel for computing weight grads
__global__ void computeWeightGradients(float* d_input, float* d_grad_out, float* d_grad_weights, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < K; ++i) {
            sum += d_input[row * K + i] * d_grad_out[i * N + col];
        }
        d_grad_weights[row * N + col] = sum;
    }
}

// kernel to compute bias grads
__global__ void computeBiasGradients(float* d_grad_out, float* d_grad_bias, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        atomicAdd(&d_grad_bias[idx], d_grad_out[idx]);
    }
}

// kernel to compute grads for the input of the previous layer
__global__ void computeInputGradients(float* d_grad_out, float* d_weights, float* d_grad_input, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < K) {
        float sum = 0.0f;
        for (int i = 0; i < N; ++i) {
            sum += d_grad_out[row * N + i] * d_weights[i * K + col];
        }
        d_grad_input[row * K + col] = sum;
    }
}

// kernel for grads of loss function - cross-entropy with logits
__global__ void cross_entropy_with_logits_backward(float* logits, int label, float* grad, int num_classes) {
    // 1. find max logit for numerical stability
    float max_logit = logits[0];
    for (int j = 1; j < num_classes; ++j) {
        max_logit = fmaxf(max_logit, logits[j]);
    }

    // 2. compute softmax and grad for each logit
    float sum_exp = 0.0f;
    for (int j = 0; j < num_classes; ++j) {
        logits[j] = expf(logits[j] - max_logit);
        sum_exp += logits[j];
    }

    // 3. comp grads for each class
    for (int j = 0; j < num_classes; ++j) {
        float softmax = logits[j] / sum_exp;
        grad[j] = softmax - (j == label ? 1.0f : 0.0f); // subtract 1 for the true class
    }
}

// kernel for sgd update
__global__ void sgd_update(float* params, float* grads, float learning_rate, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        params[idx] -= learning_rate * grads[idx];
    }
}

// host function to call the sgd update
void sgdUpdateCUDA(float* d_params, float* d_grads, float learning_rate, int size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    sgd_update<<<blocks, threads>>>(d_params, d_grads, learning_rate, size);
    CUDA_CHECK(cudaDeviceSynchronize());
}