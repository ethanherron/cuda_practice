# Programming Massively Parallel Processors: A Hands-on Approach

## Contents

- 1. Introduction
#### Part 1: Fundamental Concepts
- 2. Heterogeneous Data Parallel Computing
- 3. Multidimensional Grids and Data
- 4. GPU Architecture and Scheduling
- 5. Memory Architecture and Data Locality
- 6. Performance Considerations
#### Part 2: Parallel Patterns
- 7. Convolution
- 8. Stencil
- 9. Parallel Histogram
- 10. Reduction and Minimizing Divergence
- 11. Prefix Sum (Scan)
#### Part 3: Advanced Patterns and Applications
- 12. Merge
- 13. Sorting
- 14. Sparse Matrix Computation
- 15. Graph Traversal
- 16. Deep Learning
- 17. Iterative MRI Reconstruction
- 18. Electrostatic Potential Map
- 19. Parallel Programming and Computational Thinking
#### Part 4: Advanced Practices
- 20. Programming a Heterogeneous Computing Cluster
- 21. Dynamic Parallelism
- 22. Advanced Practices and Future Evolution
- 23. Conclusion and Outlook







## 1. Introduction

### 1.1 Heterogeneous Parallel Computing
The introduction starts off by discussing the difference between the CPU and the GPU.

One interesting idea they point out is that improving CPUs (latency-oriented designs) is more expensive than increasing throughput (throughput-oriented designs, i.e. GPUs). This is because if you double the arithmetic throughput by doubling the number of arithmetic units, at the cost of doubling the chip area and power consumption. But, if you try to reduce the arithmetic latency by half would require doubling the current at the cost of more than doubling the chip area used and quadrupling the power consumption. 

Tldr; increasing throughput is cheaper and easier than reducing latency. 


## 2. Heterogeneous Data Parallel Computing

This chapter introduces the concept of data parallelism and CUDA C features. It introduces CUDA device memory management and data transfer applications programming interface functions. It covers the basic structure of a CUDA C kernel function, built-in variables, function declaration keywords, and kernel launch syntax. 


#### Task parallelism vs. Data parallelism

Task parallelism is when you have a set of independent tasks that can be executed in parallel.

Data parallelism is when you have a set of data that can be executed in parallel.

### 2.2 CUDA C program structure

#### Threads

A thread is a simplified view of how a processor executes a sequential program in modern computers. A thread consiste of the code of the program, the point in the code that is being executed, and the data that is being processed. The execution of a thread is sequential as far as a user is concerned. 

### 2.3 A vector addition kernel

#### Vector addition in C
// compute vector sum C_h = A_h + B_h
void vecAdd(float* A_h, float* B_h, float* C_h, int N) {
    for (int i = 0; i < N; i++) {
        C_h[i] = A_h[i] + B_h[i];
    }
}
int main() {
    // Memory allocation for arrays A, B, and C
    // I/O to read A and B, N elements each
    ...
    vecAdd(A, B, C, N);
}

#### Pointers in the C language

The function arguments A, B, and C are pointers. In the C language, a pointer can be used to access variables and data structures. While a floating-point variable V can be declared with:
float V;

a pointer variable P can be declared with:
float *P;

By assigning the address of V to P with the statement P=&V, we make P "point to" V. The & operator is used to get the address of a variable. *P becomes a synonym for V. For example, U=*P assigns the value of V to U. For another example, *P=3 changes the value of V to 3.

An array in a C program can be accessed through a pointer that points to its 0th element. For example, the statement P=&(A[0]) makes P point to the 0th element of A. P[i] becomes a synonym for A[i]. In fact, the array name A is in itself a pointer to its 0th element.

#### Vector addition in CUDA C
// compute vector sum C_d = A_d + B_d
void vecAdd(float* A, float* B, float* C, int N) {
    int size = n* sizeof(float);
    float *A_d, *B_d, *C_d;
    
    // Part 1: Allocate device memory for A, B, and C
    // Copy A and B to device memory
    ...

    // Part 2: Launch the kernel - to launch a grid of threads
    // to perform the actual vector addition
    ...

    // Part 3: Copy C from device memory to host memory
    Free device memory
    ...
}

Note that the kernel function vecAdd is essentially an outsourcing agent that ships input data to a device, activates a calculation on the device, and collects the results from the device. 
The agent does so in a way that the main program does not need to even be aware that the vector addtition is now actually done on a device.

### 2.4 Device global memory and data transfer

cudaMalloc()
- Allocates object in the device global memory
- Two parameters
    - Address of a pointer to the allocated object
    - Size of allocated object in term of bytes

The first parameter to thte cudaMalloc function is the address of a pointer variable that will be set to point to the allocated object. The address of the pointer variable should be cast to (void**) because the function expects a generic pointer; the memory allocation funciton is a generic function that is not restricted to any parrticular type of object. This parameter allows the cudaMalloc function to write the address of the allocated memory into the provided pointer variable regardless of its type. The host code that calls kernels passes this pointer value to the kernels that need to access the allocated memory object. 

cudaFree()
- Frees object from device global memory
- One parameter
    - Address of the pointer to the allocated object

cudaMemcpy()
- memory data transfer
- Four parameters
    - Destination pointer
    - Source pointer
    - Number of bytes to transfer
    - cudaMemcpyKind (type/direction of transfer)

void vecAdd(float* A_h, float* B_h, float* C_h, int N) {
    int size = n * sizeof(float);
    float *A_d, *B_d, *C_d;

    cudaMalloc((void**)&A_d, size);
    cudaMalloc((void**)&B_d, size);
    cudaMalloc((void**)&C_d, size);

    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

    // kernel invocation code
    ...

    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

#### Error checking and handling in CUDA

In practice, we should surround the call with code that test for error condition and print out error messages so that the user can be aware of the fact that an error has occurred. Example:

cudaError_t err = cudaMalloc((void**)&A_d, size);
if (err != cudaSuccess) {
    printf("Error allocating device memory: %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
}

### 2.5 Kernel functions and threading

In CUDA C, a kernel function specifies the code that will be executed by all threads during a parallel phase. Since all these threads execute the same code, CUDA C programming is an instance of the well-known single-program multiple-data (SPMD) programming model.

#### Built-in variables

- threadIdx (thread identifier)
- blockIdx (block identifier)
- blockDim (block dimension, number of threads per block)
- gridDim (grid dimension, number of blocks per grid)

The blockDim variable is a structure with three unsigned integer fields (x, y, and z) that help the programmer to organize the threads into a one-, two-, or three-dimensional array. For a one-dimensional organization, only the x field is used. For a two-dimensional organization, the x and y fields are used. For a three-dimensional organization, the x, y, and z fields are used. The choice of dimensionality for organizing threads usually reflext the dimensionality of the data.

CUDA kernels have access to two more built-in variables:
- threadIdx (thread identifier)
- blockIdx (block identifier)
that allow threads to distinguish themselves from each other and to determine the area of data each thread is to work on. The threadIdx variables gives each thread a unique coordinate within a block. 

#### Hierarchical organizations

To define a global index i for a thread, we can use the following formula:

i = blockIdx.x * blockDim.x + threadIdx.x

// compute vector sum C_d = A_d + B_d
__global__ void vecAddKernel(float* A, float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

Keyword    Callable from    Executes on   Executed by
__host__        Host            Host          CPU
__global__      Host           Device         GPU
__device__     Device          Device         GPU

The "__device__" keyword indicates that the function being declared is a CUDA device function. A device function executes on a CUDA device and can be called only from a kernel function or another device function. The deevice function is executed by the device thread that calls it and does not result in any new device threads being launched.

In vecAddKernel there is no for-loop, like there is in the cpu version. In a CUDA kernel the loop is replaced with the grid of threads. The entire grid forms the equivalent of the loop. Each thread in the grid corresponds to a loop iteration. This is sometimes referred to as "loop parallelism".

Note the "if (i < N)" in addVecKernel is because not all vector lengths can be expressed as multiples of the block size. For example, let's assume the vector size is 100. The smallest efficient thread block dimension is 32. Assume that we pick 32 as block size. One would need to launch 4 blocks to cover all 100 elements. However, the 4 blocks would have 128 threads, which is 28 threads more than needed. This is why we need to check if the thread's global index is less than the vector size so we can disable threads that would access out of bounds memory.

### 2.6 Calling kernel functions

Denote a kernel launch configuration with the following parameters:
- Grid dimension (number of blocks per grid)
- Block dimension (number of threads per block)
Which are written as:
<<<gridDim, blockDim>>>

To ensure that we have enough threads in the grid to cover all the vector elements, we need to set the number of blocks in teh grid to the ceiling division (rounding up the quotient to the immediate higher integer value) of the desired number of threads.

For example, if we want 1000 threads, we would launch ceil(1000/256.0) = 4 blocks. As a result, the statement will launch 4 blocks of 256 threads each (256*4=1024). With the if (i < N) check, we ensure that only the threads that are within the valid range of the vector are active.

int vectAdd(float* A, float* B, float* C, int N) {
    // A_d, B_d, C_d allocations and copies omitted
    ...
    // Launch ceil(n/256) blocks of 256 threads each
    vecAddKernel<<<ceil(n/256.0), 256>>>(A_d, B_d, C_d, n);
    ...
}

void vecAdd(float* A, float* B, float* C, int N) {
    float *A_d, *B_d, *C_d;
    int size = n * sizeof(float);

    cudaMalloc((void**)&A_d, size);
    cudaMalloc((void**)&B_d, size);
    cudaMalloc((void**)&C_d, size);

    cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);

    vecAddKernel<<<ceil(n/256.0), 256>>>(A_d, B_d, C_d, n);

    cudaMemcpy(C, C_d, size, cudaMemcpyDeviceToHost);

    cudaFree(A_d);  
    cudaFree(B_d);
    cudaFree(C_d);
}

### Exercises

1. If we want to use each thread in a grid to calculate one output element of a vector addition, what would be the expression for ammping the thread/block indices to the data index (i)?

i = blockIdx.x * blockDim.x + threadIdx.x

2. Assume that we want to use each thread to calculate two adjacent elements of a vector addition. What would be the expression for mapping the thread/block indices to the data index (i) of the first element to be processed by a thread?

i = (blockIdx.x * blockDim.x + threadIdx.x) * 2

3. We want to use each thread to calculate two elements of a vector addition. Each thread block processes 2*blockDim.x consecutive elements that form two sections. ALl threads in each block will process a section first, each processing one element. They will then all move to the next section, each processing one element. Assume that variables i should be the index for hte first element to be processed by a thread. What would be the expression for mapping the thread/block indices to the data index of the first element?

i = blockIdx.x * blockDim.x * 2 + threadIdx.x

4. For a vector addition, assume that the vector length is 8000, each thread calculates one output element, and the thread block size is 1024 threads. The programmer configures the kernel call to have a minimum number of thread blocks to cover all output elements. How many threads will be in the grid?

8192

5. If we want to allocate an array of v integer elements in the CUDA device global memory, what would be an appropriate expression for the second argument of the cudaMalloc call?

v * sizeof(int)

6. If we want to allocate an array of n floating-point elements and have a floating-point pointer variable A_d to point to the allocated memory, what would be an appropriate expression for the first argument of the cudaMalloc call?

(void**)&A_d

7. If we want to copy 3000 bytes of data from host array A_h (A_h is a pointer to element 0 of the source array) to device array A_d (A_d is a pointer to element 0 of the destination array), what would be an appropriate API call for this data copy in CUDA?

cudaMemcpy(A_d, A_h, 3000, cudaMemcpyHostToDevice)

8. How would one declare a variable err that can appropriately receive teh returned value of a CUDA API call?

cudaError_t err;

9. Consider the following CUDA kernel and the corresponding host function that calls it:

__global__ void foo_kernel(float* a, float* b, unsigned int N) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N) {
        b[i] = 2.7f*a[i] + 4.3f;
    }
}

void foo(float* a_d, float* b_d) {
    unsigned int N = 200000;
    foo_kernel<<< (N + 128-1)/128, 128>>>(a_d, b_d, N);
}

a. What is the number of threads per block?
128
b. What is the number of threads in the grid?
(N + 128-1) * 128
c. What is the number of blocks in the grid?
(N + 128-1)/128
d. What is the number of threads that execute the code on line 2?
(N + 128-1) * 128 (total number of threads launched)
e. What is the number of threads that execute the code on line 4?
N (number of threads used for op - (num threads launched - excess threads))

10. A new summer intern was frustrated with CUDA. He has been complaining that CUDA is very tedious. He had to declare many functions that he plans to execute on both the host and the device twice, once as a host function and once as a device function. What is your response?

He's adding unnecessary redundancy to his code. We can use the "__host__" and "__device__" keywords to declare functions that will be executed on the host and the device respectively.


## 3. Multidimensional Grids and Data

### 3.1 Multidimensional grid organization

The total size of a block in current CUDA systems is limited to 1024 threads. These threads can be distributed across the three dimensions in any way as long as the total number of threads does not exceed 1024. For example, blockDim values of (512, 1, 1), (8, 16, 4) and (32, 16, 2) are all valid, but (32, 32, 2) is not allowed because the total number of threads would exceed 1024.

A grid and its blocks do not need to have the same dimensionality. A grid can have higher dimensionality than its blocks and vice versa. For example, a gridDim(2, 2, 1) and a blockDim(4, 2, 2) can be created with:

dim3 gridDim(2, 2, 1);
dim3 blockDim(4, 2, 2);
KernelFunction<<<gridDim, blockDim>>>(...);

This example creates 4 blocks organized into a 2x2 array. 
And each block has 4x2x2=16 threads.

### 3.2 Mapping threads to multidimensional data

Multidimensional arrays in C are linearized because modern computers use "flat" memory systems.

There are two ways in which a 2D array can be linearized:

1. Row-major order ()
2. Column-major order

In row-major order, all elements of the same row are placed consecutively in memory.
In column-major order, all elements of the same column are placed consecutively in memory.

x = [[1, 2, 3], [4, 5, 6]]

Row-major order: 1, 2, 3, 4, 5, 6
Column-major order: 1, 4, 2, 5, 3, 6

Example kernel function:
equation to convert image to grayscale:

// The input image is encoded as unsigned chars [0, 255]
// Each pixel is 3 consecutive chars for the 3 channels
__global__
void colorToGrayscaleConvertion(unsigned char * Pout,
                                unsigned char * Pin,
                                int width,
                                int height  ) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        // Get 1D offset for the grayscale image
        int grayOffset = row*width + col;
        // One can think of the RGB image having CHANNEL
        // times more columns than the grayscale image
        int rgbOffset = grayOffset * CHANNELS;
        unsigned char r = Pin[rgbOffset];
        unsigned char g = Pin[rgbOffset + 1];
        unsigned char b = Pin[rgbOffset + 2];
        // Perform the rescaling and store it
        // We mulitply by floating point constants
        Pout[grayOffset] = 0.21f*r + 0.71f*g + 0.07f*b;
    }
}

### 3.3 Image blur: a more complex kernel

// set blur to be a 3x3 conv kernel essentially
// BLUR_SIZE is the radius of the conv kernel (r=1, d=r*2, +1 for the center)
BLUR_SIZE = 1 
__global__
void blurKernel(unsigned char *in, unsigned char *out, int w, int h) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < w && row < h) {
        int pixVal = 0;
        int pixels = 0;
        // Get average of the surrounding BLUR_SIZE x BLUR_SIZE box
        for (int blurRow=-BLUR_SIZE; blurRow < BLUR_SIZE+1; ++blurRow ) {
            for (int blurCol=-BLUR_SIZE; blurCol < BLUR_SIZE+1; ++blurCol) {
                int curRow = row + blurRow;
                int curCol = col + blurCol;
                // verify we have a valid image pixel
                if(curRow >=0 && curRow < h && curCol >=0 && curCol < w) {
                    pixVal += in[curRow*w + curCol];
                    // keep track of number of pixels in the avg
                    pixels++;
                }
            }
        }
        // write our new pixel value out
        out[row*w + col] = pixVal / pixels;
    }
}


### 3.4 Matrix multiplication
Example matrix multiplication:
M @ N = P
(i,j) @ (j,k) = (i,k)
To implement matrix multiplication using CUDA, we can map the threads in the grid to the elements of the output matrix P with the same approach that we used for colorToGrayScaleConversion. That is, each thread is responsible for one element of the output matrix P. The row and column indices for the P element to be calculated by each thread are the same as before:
row = blockIdx.y * blockDim.y + threadIdx.y
col = blockIdx.x * blockDim.x + threadIdx.x

__global__ void MatrixMulKernel(float* M, float* N, float* P, int width) {
    // init all threads for the output matrix P
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    // entire mat mul for loop
    // "trim" threads that are outside the output matrix P
    if (row < width && col < width) {
        float Pvalue = 0;
        // row x col dot product for loop
        for (int k = 0; k < Width; ++k) {
            // index M row-wise, N column-wise
            // compute the dot product of P[i,k] by running forloop
            // over all threads in correct row-column position
            Pvalue += M[row*width + k] * N[k*width + col];
        }
        // write the computed value to the output matrix P
        P[row*width + col] = Pvalue;
    }
}

### Summary

CUDA grids and blocks are multidimensional with up tot three dimensions. The multidimensionality of grids and blocks is useful for organizing threads to be mapped to multidimensional data. The kernel execution configuration parameters define the dimensions of a grid and its blocks. Unique coordinates in blockIdx and threadIdx allow threads of a grid to identify themselves and their domains of data. It is a programmer's responsibility to use these variables in kernel functions so that the threads can properly identify the portion of the data to process. When accessing multidimensional data, programmers will often have to linearize multidimensional indices into a 1D offset. The reason is that dynamically allocated multidimensional arrays in C are typically stored as 1D arrays in row-major order. 

### Exercises

1. In this chapter we implemented a matrix multiplication kernel that has each thread produce one output matrix element. In this question, you will implement different matrix-matrix multiplication kernels and compare them. 
a. Write a kernel that has each thread produce one output matrix row. Fill in the execution configuration parameters for the design. 
// So we want to compute P row by row
// We need to compute each row of P by "holding" a row of M and iterating over the columns of N
// Need to think about this in row-major order so we actually dont need to initialize the col index
__global__ void MatrixMulRowKernel(float* M, float* N, float* P, int width) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < width) {
        for (int col = 0; col < width; ++col) {
            float Pvalue = 0;
            for (int k = 0; k < width; ++k) {
                Pvalue += M[row*width + k] * N[k*width + col];
            }
            P[row*width + col] = Pvalue;
        }
    }
}

b. Write a kernel that has each thread produce one output matrix column. Fill in the execution configuration parameters for the design. 
// same as above but iterate over the rows of M instead of the columns
__global__ void MatrixMulColKernel(float* M, float* N, float* P, int width) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < width) {
        for (int row = 0; row < width; ++row) {
            float Pvalue = 0;
            for (int k = 0; k < width; ++k) {
                Pvalue += M[row*width + k] * N[k*width + col];
            }
            P[row*width + col] = Pvalue;
        }
    }
}

c. Analyze the pros and cons of each of the two kernel designs. 
The row-wise kernel is more efficient because it has less memory access and less computation. The column-wise kernel is less efficient because it has more memory access and more computation. So, from what I understand which kernel to choose depends on the type of data you're operating on. 

2. A matrix-vector mulitplciation takes an input matrix B and a vector C and produces one output vector A. Each element of the output vector A is the dot product of one row of the input matrix B and C, that is, A[i] = \sum_{j=0}^{n-1} B[i,j] * C[j]. For simplicity, we will handle only square mamtrices whose elements are single-precision floating-point numbers. Write a matrix-vector multiplication kernel and the host stub function that can be called with 4 parameters: pointer to the output matrix, pointer to the input matrix, pointer to the input vector, and the number of elements in each dimension. Use one thread to calculate one element of the output vector.

__global__ void MatrixVecMulKernel(float* B, float* C, float* A, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float Avalue = 0;
        for (int j = 0; j < n; ++j) {
            Avalue += B[i*n + j] * C[j];
        }
        A[i] = Avalue;
    }
}

3. Consider the following CUDA kernel and the corresponding host function that calls it:

__global__ void foo_kernel(float* a, float* b, unsigned int M, unsigned int N) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        b[row*N + col] = a[row*N + col]/2.1f + 4.8f;
    }
}
void foo(float* a_d, float* b_d) {
    unsigned int M = 150;
    unsigned int N = 300;
    dim3 bd(16, 32);
    dim3 gd((N - 1)/16 + 1, (M - 1)/32 + 1);
    foo_kernel<<<gd, bd>>>(a_d, b_d, M, N);
}

a. What is the number of threads per block?
16 * 32 = 512
b. What is the number of threads in the grid?
((N - 1)/16 + 1 * (M - 1)/32 + 1) * 512
c. What is the number of blocks in the grid?
(N - 1)/16 + 1 * (M - 1)/32 + 1
d. What is the number of threads that execute the code on line 05?
150 * 300 = 45000

4. Consider a 2D matrix with a width of 400 and a height of 500. The matrix is stored as a one-dimensional array. Specify the array index of hte matrix element at row 20 and column 10:
a. Row-major order
row * width + col -> 20 * 400 + 10 = 8010

b. Column-major order
col * width + row -> 10 * 400 + 20 = 4020

5. Consider a 3D tensor with a width of 400, a height of 500, and a depth of 300. The tensor is stored as a one-dimensional array in row-major order. Specify the array index of the tensor element at x=10, y=20, z=5.
plane = x * y (or h*w)
plane*z + row*width + col = ((10 * 20) * 5) + (20 * 400) + 10 = 10000 + 8000 + 10 = 18010


## 4. Compute Architecture and Scheduling

