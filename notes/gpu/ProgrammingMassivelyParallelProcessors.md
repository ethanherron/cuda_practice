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

> 💡 **Core Concept**: This chapter introduces how to program GPUs using CUDA C by managing memory across devices and creating parallel kernels that execute the same code across thousands of threads.

### 2.1 Parallelism Paradigms

#### Task parallelism vs. Data parallelism

- **Task parallelism**: Different independent operations executed simultaneously
  - Example: Web server handling multiple client requests concurrently
  - Often uses fewer, more complex threads
  - Better for diverse workloads with different execution paths

- **Data parallelism**: Same operation applied to multiple data elements simultaneously
  - Example: Adding two vectors element by element
  - Uses many simple threads doing identical work on different data
  - Ideal for GPUs with their thousands of cores
  - CUDA primarily focuses on this model

### 2.2 CUDA C Program Structure

A typical CUDA application consists of these phases:
1. **Initialize data** on the host (CPU)
2. **Transfer data** from host to device (GPU)
3. **Execute kernel** on the device
4. **Transfer results** back from device to host

#### Threads

A thread represents a sequential execution path through a program. In CUDA:
- Each thread has its own program counter and registers
- Threads execute the same code but operate on different data
- The execution of a thread is sequential as far as a user is concerned
- CUDA applications can launch thousands or millions of threads simultaneously

```
Host Code                    Device Code (Kernel)
┌───────────────┐            ┌───────────────┐
│ Allocate mem  │            │ Thread 0      │
│ Copy to device├───────────►│ Thread 1      │
│ Launch kernel │            │ Thread 2      │
│ Copy to host  │◄───────────┤ ...           │
│ Free memory   │            │ Thread n-1    │
└───────────────┘            └───────────────┘
```

### 2.3 A Vector Addition Example

#### Vector addition in C (CPU)
```c
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
```

> 📝 **Note**: In CPU code, a single thread processes all elements sequentially in a for-loop.

#### Pointer Review (C Language)

- **Variables vs. Pointers**:
  - `float V;` - Declares a variable that stores a value
  - `float *P;` - Declares a pointer that stores a memory address
  
- **Key Operations**:
  - `P = &V;` - P now points to V's memory address
  - `*P` - Accesses the value at P's address (value of V)
  - `*P = 3;` - Changes the value at P's address (changes V to 3)

- **Arrays and Pointers**:
  - Array names are actually pointers to the first element
  - `A[i]` is equivalent to `*(A+i)`
  - `P = &(A[0])` makes P point to the first element of A

#### Vector addition in CUDA C (GPU approach)

CUDA vector addition involves three main steps:

```c
void vecAdd(float* A_h, float* B_h, float* C_h, int N) {
    int size = N * sizeof(float);
    float *A_d, *B_d, *C_d;
    
    // Part 1: Allocate device memory for A, B, and C
    // Copy A and B to device memory
    cudaMalloc((void**)&A_d, size);
    cudaMalloc((void**)&B_d, size);
    cudaMalloc((void**)&C_d, size);
    
    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

    // Part 2: Launch the kernel
    // Grid of threads performs the actual vector addition
    vecAddKernel<<<ceil(N/256.0), 256>>>(A_d, B_d, C_d, N);

    // Part 3: Copy C from device memory to host memory
    // Free device memory 
    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);
    
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}
```

> 🔑 **Key insight**: The host function orchestrates the entire process but the actual computation happens on the GPU, where thousands of threads execute the same kernel function on different pieces of data.

### 2.4 Device Global Memory and Data Transfer

#### Memory Allocation with cudaMalloc()

```c
cudaError_t cudaMalloc(void** devPtr, size_t size);
```

- **Purpose**: Allocates object in the device global memory
- **Parameters**:
  - `devPtr`: Address of a pointer to store the allocated memory address
  - `size`: Size of allocated object in bytes
- **Returns**: cudaSuccess or an error code

> ⚠️ **Important**: `cudaMalloc` takes a pointer-to-pointer (void**) because it needs to modify the pointer value to point to the newly allocated memory. This is one of the most common sources of errors for beginners.

Example with error checking:
```c
float *A_d;
cudaError_t err = cudaMalloc((void**)&A_d, size);
if (err != cudaSuccess) {
    printf("Error allocating device memory: %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
}
```

#### Memory Deallocation with cudaFree()

```c
cudaError_t cudaFree(void* devPtr);
```

- **Purpose**: Frees object from device global memory
- **Parameters**:
  - `devPtr`: Pointer to previously allocated memory
- **Returns**: cudaSuccess or an error code

> 🚫 **Common mistake**: Failing to free device memory can lead to memory leaks, especially in applications that repeatedly allocate memory.

#### Data Transfer with cudaMemcpy()

```c
cudaError_t cudaMemcpy(void* dst, const void* src, 
                      size_t count, enum cudaMemcpyKind kind);
```

- **Purpose**: Transfers data between host and device memory
- **Parameters**:
  - `dst`: Destination pointer
  - `src`: Source pointer
  - `count`: Number of bytes to transfer
  - `kind`: Type/direction of transfer
- **Transfer Types**:
  - `cudaMemcpyHostToDevice`: CPU → GPU
  - `cudaMemcpyDeviceToHost`: GPU → CPU
  - `cudaMemcpyDeviceToDevice`: GPU → GPU
  - `cudaMemcpyHostToHost`: CPU → CPU (rarely used)

> 📝 **Mental model**: Think of the GPU as having its own separate memory space. Data must be explicitly moved between CPU and GPU memory spaces using cudaMemcpy.

#### Complete Vector Addition with Memory Management

```c
void vecAdd(float* A_h, float* B_h, float* C_h, int N) {
    int size = N * sizeof(float);
    float *A_d, *B_d, *C_d;

    // Allocate device memory
    cudaMalloc((void**)&A_d, size);
    cudaMalloc((void**)&B_d, size);
    cudaMalloc((void**)&C_d, size);

    // Copy from host to device
    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

    // Launch kernel
    vecAddKernel<<<ceil(N/256.0), 256>>>(A_d, B_d, C_d, N);

    // Copy result back to host
    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}
```

#### Error Checking Pattern

Always check for errors in CUDA calls with this pattern:

```c
cudaError_t err = cudaFunction();
if (err != cudaSuccess) {
    printf("CUDA Error: %s\n", cudaGetErrorString(err));
    // Handle error (cleanup, exit, etc.)
}
```

> 🔍 **Pro tip**: Create a macro for error checking to avoid repetitive code:
> ```c
> #define CHECK_CUDA_ERROR(call) { \
>     cudaError_t err = call; \
>     if (err != cudaSuccess) { \
>         printf("CUDA Error: %s at %s:%d\n", \
>                cudaGetErrorString(err), __FILE__, __LINE__); \
>         exit(EXIT_FAILURE); \
>     } \
> }
> ```
> Then use: `CHECK_CUDA_ERROR(cudaMalloc((void**)&A_d, size));`

### 2.5 Kernel Functions and Threading

Kernel functions are the core of CUDA programming - they define what each thread executes.

#### CUDA Function Type Qualifiers

| Qualifier   | Callable from | Executes on | Executed by |
|-------------|---------------|------------|------------|
| `__host__`   | Host          | Host       | CPU        |
| `__global__` | Host          | Device     | GPU        |
| `__device__` | Device        | Device     | GPU        |

- **`__global__`**: Kernel functions that launch a grid of threads
- **`__device__`**: Helper functions called by kernels, executed by a single thread
- **`__host__`**: Regular CPU functions (default if no qualifier is specified)
- **`__host__ __device__`**: Functions that can be called and executed on both CPU and GPU

#### Thread Identification with Built-in Variables

Every CUDA thread has access to these built-in variables:

- **`threadIdx`**: Thread index within a block (unique within block)
- **`blockIdx`**: Block index within the grid (unique within grid)
- **`blockDim`**: Number of threads per block
- **`gridDim`**: Number of blocks in the grid

Each variable has `.x`, `.y`, and `.z` components for 3D organization.

#### Thread Indexing in 1D

To get a unique global index for each thread in a 1D grid:

```c
int i = blockIdx.x * blockDim.x + threadIdx.x;
```

Visualization of this formula:
```
Thread indices within blocks:   [0,1,2...31][0,1,2...31][0,1,2...31]
Block indices:                   Block 0     Block 1     Block 2
Global thread indices:         [0-31]      [32-63]     [64-95]...
```

#### Vector Addition Kernel Implementation

```c
__global__ void vecAddKernel(float* A, float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {  // Check boundary
        C[i] = A[i] + B[i];
    }
}
```

> ⚠️ **Critical**: The boundary check `if (i < N)` prevents out-of-bounds memory access when the number of threads exceeds the array size, which happens frequently because we round up the number of blocks.

#### From Loop to Parallelism

| CPU (Sequential)            | GPU (Parallel)                  |
|-----------------------------|---------------------------------|
| `for (int i = 0; i < N; i++)`  | Many threads, each with unique `i` |
| Explicit iteration          | Implicit iteration via threads   |
| Single thread does all work | Each thread does small amount   |

### 2.6 Calling Kernel Functions

#### Kernel Launch Syntax

```c
kernel<<<gridDim, blockDim, sharedMem, stream>>>(args...);
```

- **`gridDim`**: Number of blocks in the grid (can be int or dim3)
- **`blockDim`**: Number of threads per block (can be int or dim3)
- **`sharedMem`**: (Optional) Dynamic shared memory size in bytes
- **`stream`**: (Optional) CUDA stream for asynchronous execution

> 📝 **Note**: The `<<<...>>>` syntax is unique to CUDA C/C++ and is processed by the NVCC compiler.

#### Calculating Grid Dimensions

For a 1D vector of length N and block size of 256:

```c
int blockSize = 256;
int numBlocks = (N + blockSize - 1) / blockSize;  // Ceiling division
kernel<<<numBlocks, blockSize>>>(args...);
```

This is equivalent to:
```c
kernel<<<ceil(N/256.0), 256>>>(args...);
```

> 💭 **Intuition**: If N is 1000 and blockSize is 256, we need 4 blocks because:
> (1000 + 256 - 1) / 256 = 1255 / 256 = 4.9 → 4 blocks
> This gives us 4 × 256 = 1024 threads, which is more than the 1000 we need.
> The boundary check ensures only the first 1000 threads perform computation.

#### Complete Vector Addition Example

```c
void vecAdd(float* A_h, float* B_h, float* C_h, int N) {
    // Allocate device memory
    float *A_d, *B_d, *C_d;
    int size = N * sizeof(float);
    cudaMalloc((void**)&A_d, size);
    cudaMalloc((void**)&B_d, size);
    cudaMalloc((void**)&C_d, size);

    // Copy input vectors from host to device
    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

    // Launch kernel with proper grid dimensions
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    vecAddKernel<<<numBlocks, blockSize>>>(A_d, B_d, C_d, N);

    // Copy result back to host
    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}
```

### 2.7 Common Pitfalls and Best Practices

- **Memory Management**:
  - Always free GPU memory with `cudaFree()` to prevent leaks
  - Check return values of CUDA API calls
  - Minimize data transfers between host and device

- **Kernel Execution**:
  - Always include boundary checks in kernels
  - Choose block sizes that are multiples of 32 (warp size)
  - Typical block sizes: 128, 256, or 512 threads

- **Debugging**:
  - Use `cudaGetLastError()` after kernel launches to check for errors
  - For complex kernels, print intermediate values with `printf()` (available in CUDA)
  - Consider using NVIDIA's NSight or CUDA-GDB for debugging

### 2.8 Key Takeaways

- CUDA follows the **SPMD** (Single Program, Multiple Data) programming model
- A typical CUDA program: allocate GPU memory → copy data to GPU → execute kernel → copy results back → free memory
- Data must be explicitly moved between CPU and GPU memory spaces
- CUDA kernels replace loops with parallelism across thousands of threads
- Each thread needs a way to identify which data element(s) to process using built-in variables

***

**Exercise Idea**: Try implementing a vector SAXPY operation (`Y = a*X + Y`) in CUDA, where `a` is a scalar and `X` and `Y` are vectors.


## 3. Multidimensional Grids and Data

> 💡 **Core Concept**: This chapter explains how to organize CUDA threads into 2D/3D structures to efficiently process multidimensional data like matrices and images.

### 3.1 Multidimensional Thread Organization

#### Block and Grid Dimensionality

CUDA supports organizing threads into one, two, or three dimensions using the `dim3` type, which allows for:
- More intuitive mapping to multidimensional data
- Better spatial locality for certain algorithms
- More natural expression of problems with 2D/3D structure

#### Thread Block Limitations

```c
// Thread block with different configurations
dim3 block1D(256, 1, 1);        // 256 threads in x-dimension
dim3 block2D(16, 16, 1);        // 256 threads in x,y-dimensions
dim3 block3D(8, 8, 4);          // 256 threads in x,y,z-dimensions
```

> ⚠️ **Important**: The total number of threads in a block must not exceed 1024 in current CUDA systems.

Valid configurations include:
- `(512, 1, 1)` = 512 threads
- `(8, 16, 4)` = 512 threads
- `(32, 16, 2)` = 1024 threads

Invalid configuration:
- `(32, 32, 2)` = 2048 threads (exceeds 1024 limit)

#### Creating Grids with Different Dimensionality

Grid and block dimensions don't need to match. You can have:
- 2D grid of 1D blocks
- 1D grid of 3D blocks
- Any other combination

```c
// Creating a 2D grid of 3D blocks
dim3 gridDim(2, 2, 1);      // 2×2×1 = 4 blocks in grid
dim3 blockDim(4, 2, 2);     // 4×2×2 = 16 threads per block

// Launch kernel with these dimensions
KernelFunction<<<gridDim, blockDim>>>(...);
```

Visualization of this grid:
```
                 Block (0,0)         Block (1,0)
                ┌───────────┐       ┌───────────┐
                │ T T T T   │       │ T T T T   │
                │ T T T T   │       │ T T T T   │
                │ T T T T   │       │ T T T T   │
                │ T T T T   │       │ T T T T   │
                └───────────┘       └───────────┘

                 Block (0,1)         Block (1,1)
                ┌───────────┐       ┌───────────┐
                │ T T T T   │       │ T T T T   │
                │ T T T T   │       │ T T T T   │
                │ T T T T   │       │ T T T T   │
                │ T T T T   │       │ T T T T   │
                └───────────┘       └───────────┘
```

#### Thread Indexing in Multidimensions

With multidimensional organization, we access thread coordinates using:

```c
// Thread coordinates within a block
int threadX = threadIdx.x;
int threadY = threadIdx.y; 
int threadZ = threadIdx.z;

// Block coordinates within the grid
int blockX = blockIdx.x;
int blockY = blockIdx.y;
int blockZ = blockIdx.z;
```

For global 2D coordinates:
```c
int col = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;
```

### 3.2 Mapping Threads to Multidimensional Data

#### Linearizing Multidimensional Arrays

Modern computers use "flat" memory, so multidimensional arrays must be linearized.

##### Row-Major vs. Column-Major Order

**Row-Major Order** (C/C++ default):
- Elements in the same row are stored consecutively
- Moving horizontally in the array accesses adjacent memory locations
- Used in C, C++, Python, and most other languages

**Column-Major Order** (Fortran, MATLAB):
- Elements in the same column are stored consecutively
- Moving vertically in the array accesses adjacent memory locations
- Used in Fortran, R, MATLAB

Example 2×3 matrix:
```
[ 1  2  3 ]
[ 4  5  6 ]
```

Row-major representation: `[1, 2, 3, 4, 5, 6]`
Column-major representation: `[1, 4, 2, 5, 3, 6]`

#### Row-Major Linear Indexing Formulas

For a 2D array with dimensions `width × height`:
```c
// Access element at (row, col)
int index = row * width + col;
```

For a 3D array with dimensions `width × height × depth`:
```c
// Access element at (x, y, z)
int index = z * (height * width) + y * width + x;
```

> 🔍 **Tip**: Always use variables for the dimensions rather than hardcoding them, making your code easier to maintain and adapt.

#### Image Processing Example: RGB to Grayscale

The following kernel converts an RGB image to grayscale:

```c
#define CHANNELS 3  // RGB channels

__global__ void colorToGrayscaleConversion(
                unsigned char* Pout,  // Output grayscale image
                unsigned char* Pin,   // Input RGB image
                int width,            // Image width
                int height) {         // Image height
    
    // Calculate 2D position in the image
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Check if within image boundaries
    if (col < width && row < height) {
        // Calculate 1D offset for grayscale image
        int grayOffset = row * width + col;
        
        // Calculate 1D offset for RGB image (3 channels per pixel)
        int rgbOffset = grayOffset * CHANNELS;
        
        // Extract RGB components
        unsigned char r = Pin[rgbOffset];
        unsigned char g = Pin[rgbOffset + 1];
        unsigned char b = Pin[rgbOffset + 2];
        
        // Standard luminance conversion formula
        Pout[grayOffset] = 0.21f*r + 0.71f*g + 0.07f*b;
    }
}
```

To launch this kernel for a 1024×768 image:

```c
// Choose a 16×16 block size (common for 2D processing)
dim3 blockSize(16, 16);

// Calculate grid dimensions to cover the entire image
dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
              (height + blockSize.y - 1) / blockSize.y);

// Launch kernel
colorToGrayscaleConversion<<<gridSize, blockSize>>>(d_grayImage, d_rgbImage, width, height);
```

> 🔑 **Key insight**: The boundary check `if (col < width && row < height)` is crucial since we're launching more threads than pixels to ensure coverage of the entire image.

### 3.3 Image Blur: A More Complex Kernel

Image blur is a common operation that requires each thread to access surrounding pixels, demonstrating a more complex access pattern.

#### Blur Kernel Implementation

```c
#define BLUR_SIZE 1  // Radius of blur kernel (1=3×3 filter)

__global__ void blurKernel(unsigned char* in,    // Input image
                         unsigned char* out,    // Output image
                         int width, 
                         int height) {
    
    // Calculate pixel position
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col < width && row < height) {
        int pixVal = 0;     // Sum of pixel values
        int pixels = 0;     // Count of pixels in average
        
        // Blur kernel is (2*BLUR_SIZE+1) × (2*BLUR_SIZE+1)
        // For BLUR_SIZE=1, we have a 3×3 kernel
        for (int blurRow = -BLUR_SIZE; blurRow <= BLUR_SIZE; ++blurRow) {
            for (int blurCol = -BLUR_SIZE; blurCol <= BLUR_SIZE; ++blurCol) {
                
                // Calculate position of neighboring pixel
                int curRow = row + blurRow;
                int curCol = col + blurCol;
                
                // Check if neighbor is within image boundaries
                if (curRow >= 0 && curRow < height && 
                    curCol >= 0 && curCol < width) {
                    // Add pixel value to sum
                    pixVal += in[curRow * width + curCol];
                    pixels++; // Increment count
                }
            }
        }
        
        // Write average to output image
        out[row * width + col] = (unsigned char)(pixVal / pixels);
    }
}
```

> 💡 **Understanding**: This implements a box blur by averaging all pixels in a 3×3 neighborhood around each pixel. The boundary check ensures proper handling of image edges.

#### Thread Access Pattern Visualization

For a 3×3 blur filter, each thread's access pattern looks like:

```
┌───┬───┬───┐
│ X │ X │ X │ 
├───┼───┼───┤
│ X │ C │ X │
├───┼───┼───┤
│ X │ X │ X │
└───┴───┴───┘
```

Where `C` is the center pixel and `X` marks neighboring pixels that are accessed to compute the average.

> ⚠️ **Performance note**: This implementation has suboptimal memory access patterns. In Chapter 5, we'll learn how to use shared memory to optimize operations like this where multiple threads need access to the same data.

### 3.4 Matrix Multiplication

Matrix multiplication is a fundamental operation in scientific computing and demonstrates the power of 2D thread organization.

#### Matrix Multiplication Algorithm

For matrices:
- `M` (dimensions `m × n`)
- `N` (dimensions `n × k`)
- `P` (dimensions `m × k`) as the result of `M × N`

The formula for each element of P is:
```
P[i,j] = ∑(k=0 to n-1) M[i,k] × N[k,j]
```

#### Basic Matrix Multiplication Kernel

```c
__global__ void matrixMulKernel(float* M, float* N, float* P, int width) {
    // Calculate row and column indices of P
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check if within matrix dimensions
    if (row < width && col < width) {
        float Pvalue = 0;
        
        // Multiply row of M by column of N
        for (int k = 0; k < width; ++k) {
            Pvalue += M[row * width + k] * N[k * width + col];
        }
        
        // Store result in P
        P[row * width + col] = Pvalue;
    }
}
```

> 📝 **Note**: This simplified example assumes square matrices of the same width for clarity. A more general implementation would handle matrices of different dimensions.

#### Launch Configuration for Matrix Multiplication

```c
// For a 1024×1024 matrix
int width = 1024;
int blockSize = 16; // 16×16 = 256 threads per block

// Calculate grid dimensions
dim3 dimBlock(blockSize, blockSize);
dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x,
             (width + dimBlock.y - 1) / dimBlock.y);

// Launch kernel
matrixMulKernel<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, width);
```

#### Memory Access Pattern Analysis

This kernel has different memory access patterns for matrices M and N:

- For matrix M: Row-wise access (coalesced memory access)
- For matrix N: Column-wise access (non-coalesced, can be inefficient)

Visualization for one thread computing P[i,j]:

```
Matrix M:           Matrix N:           Matrix P:
┌───┬───┬───┬───┐   ┌───┬───┬───┬───┐   ┌───┬───┬───┬───┐
│   │   │   │   │   │   │ ↓ │   │   │   │   │   │   │   │
├───┼───┼───┼───┤   ├───┼───┼───┼───┤   ├───┼───┼───┼───┤
│ → │ → │ → │ → │   │   │ ↓ │   │   │   │   │   │ X │   │
├───┼───┼───┼───┤   ├───┼───┼───┼───┤   ├───┼───┼───┼───┤
│   │   │   │   │   │   │ ↓ │   │   │   │   │   │   │   │
├───┼───┼───┼───┤   ├───┼───┼───┼───┤   ├───┼───┼───┼───┤
│   │   │   │   │   │   │ ↓ │   │   │   │   │   │   │   │
└───┴───┴───┴───┘   └───┴───┴───┴───┘   └───┴───┴───┴───┘
```

Where `→` shows row access in M, `↓` shows column access in N, and `X` marks the output element being computed.

> 🔍 **Performance insight**: This implementation is inefficient because:
> 1. It performs redundant global memory accesses
> 2. Column-wise access in matrix N causes non-coalesced memory access
> 
> In Chapter 5, we'll learn to optimize this using shared memory.

### 3.5 Practical Tips for Multidimensional Thread Organization

#### Dimensionality Choice

- Choose dimensionality that naturally matches your data
- For images and matrices, 2D organization is typically best
- For 1D data, stick with 1D organization for simplicity

#### Block Size Selection

For 2D blocks, common choices include:
- 16×16 (256 threads)
- 32×8 (256 threads)
- 8×32 (256 threads)

Factors to consider:
1. Total threads should be a multiple of 32 (warp size)
2. Avoid very small dimensions (e.g., prefer 32×8 over 128×2)
3. Consider memory access patterns (e.g., prefer wider blocks for row-major access)

#### 3D Organization Considerations

- 3D thread organization is useful for volumetric data (e.g., medical imaging)
- Keep the total thread count under 1024
- Common 3D block configurations: 8×8×8, 16×8×4, 32×4×4

### 3.6 Key Takeaways

- CUDA supports 1D, 2D, and 3D thread organization through blockIdx and threadIdx
- Each thread needs to calculate its position in the data using these indices
- Boundary checks are essential when mapping threads to data
- Row-major vs. column-major storage affects memory access efficiency
- Thread organization should match data organization when possible
- Matrix multiplication and image processing benefit significantly from 2D thread organization

***

**Exercise Ideas**:
1. Modify the grayscale conversion to use a different weight formula
2. Implement a Gaussian blur instead of a box blur
3. Implement matrix-vector multiplication using 2D thread organization


## 4. GPU Architecture and Scheduling

> 💡 **Core Concept**: This chapter explores how GPUs physically execute CUDA programs through Streaming Multiprocessors (SMs), warp-based execution, and scheduling mechanisms that enable thousands of threads to run efficiently despite hardware limitations.

### 4.1 Architecture of a Modern GPU

#### Streaming Multiprocessors (SMs)

GPUs are organized into an array of highly threaded **Streaming Multiprocessors (SMs)**:

- Each SM contains multiple processing units called **CUDA cores**
- Cores within an SM share control logic and memory resources
- Modern GPUs have many SMs (e.g., NVIDIA A100 has 108 SMs with 64 cores each, totaling 6,912 cores)
- Global memory is off-chip device memory accessible by all SMs

```
┌─────────────────────── GPU ───────────────────────┐
│                                                    │
│  ┌────────┐ ┌────────┐ ┌────────┐     ┌────────┐  │
│  │   SM   │ │   SM   │ │   SM   │     │   SM   │  │
│  │ ┌────┐ │ │ ┌────┐ │ │ ┌────┐ │     │ ┌────┐ │  │
│  │ │Core│ │ │ │Core│ │ │ │Core│ │     │ │Core│ │  │
│  │ │Core│ │ │ │Core│ │ │ │Core│ │     │ │Core│ │  │
│  │ │Core│ │ │ │Core│ │ │ │Core│ │ ... │ │Core│ │  │
│  │ │Core│ │ │ │Core│ │ │ │Core│ │     │ │Core│ │  │
│  │ │Core│ │ │ │Core│ │ │ │Core│ │     │ │Core│ │  │
│  │ └────┘ │ │ └────┘ │ │ └────┘ │     │ └────┘ │  │
│  └────────┘ └────────┘ └────────┘     └────────┘  │
│                                                    │
│ ┌──────────────── Global Memory ─────────────────┐ │
│ └───────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────┘
```

> 🔑 **Key insight**: GPUs achieve high throughput by having many simpler cores rather than a few complex ones like CPUs. This design is optimized for data-parallel workloads where the same operation is performed on many data elements.

### 4.2 Block Scheduling

When a kernel is launched, the CUDA runtime system:

1. Creates a grid of threads that will execute the kernel
2. Assigns blocks to SMs on a block-by-block basis
3. All threads in a block are assigned to the same SM (never split across SMs)
4. Multiple blocks are likely assigned to the same SM (limited by SM resources)

#### Block Assignment Process

```
Kernel Launch
    │
    ▼
┌────────────────────┐
│ CUDA Runtime       │
│ ┌────────────────┐ │
│ │ Block Queue    │ │
│ │ ┌───┐┌───┐┌───┐│ │
│ │ │B12││B13││B14││ │ ─────┐
│ │ └───┘└───┘└───┘│ │      │     ┌──────┐     ┌──────┐
│ └────────────────┘ │      │     │  SM  │     │  SM  │
└────────────────────┘      │     │┌────┐│     │┌────┐│
                            └────►││B1  ││     ││B5  ││
                                  │└────┘│     │└────┘│
                                  │┌────┐│     │┌────┐│
                                  ││B2  ││     ││B6  ││
                                  │└────┘│     │└────┘│
                                  │┌────┐│     │┌────┐│
                                  ││B3  ││     ││B7  ││
                                  │└────┘│     │└────┘│
                                  │┌────┐│     │┌────┐│
                                  ││B4  ││     ││B8  ││
                                  │└────┘│     │└────┘│
                                  └──────┘     └──────┘
```

> 🔍 **Insight**: The CUDA runtime maintains a queue of blocks waiting to be executed. As SMs complete execution of blocks, new ones are assigned from this queue. This dynamic scheduling enables transparent scalability across different GPU models.

#### Block Scheduling Implications

- Each block must be able to execute independently of other blocks
- Blocks cannot reliably communicate with each other during execution
- The order of block execution is not guaranteed and may vary between runs
- The number of blocks that can run concurrently depends on the GPU resources

> ⚠️ **Important**: Never assume any particular execution order of blocks. This is why global synchronization across blocks is not directly supported in CUDA.

### 4.3 Synchronization and Transparent Scalability

#### Thread Synchronization with __syncthreads()

```c
__syncthreads();
```

- Acts as a barrier synchronization for all threads in a block
- All threads must reach the barrier before any can proceed
- Ensures threads in a block execute in lockstep at synchronization points
- Only works within a block (no global synchronization across blocks)

#### Example with Synchronization:

```c
__global__ void syncExample(float* data) {
    __shared__ float shared_data[256];
    
    // Load data into shared memory
    shared_data[threadIdx.x] = data[threadIdx.x];
    
    // Wait for all threads to complete their loads
    __syncthreads();
    
    // Now all threads can safely read shared_data
    // loaded by other threads
    float value = shared_data[255 - threadIdx.x];
    
    // ... rest of kernel
}
```

> ⚠️ **Warning**: If `__syncthreads()` is inside a conditional statement and some threads don't execute it, this will cause a deadlock. All threads in a block must execute the same `__syncthreads()` calls.

#### Transparent Scalability

CUDA achieves transparent scalability because:
- Blocks execute independently
- The runtime can distribute blocks across available SMs
- The same code works on different GPUs with varying numbers of SMs
- More powerful GPUs just process more blocks concurrently

> 🔑 **Key insight**: A CUDA program written for an entry-level GPU with few SMs will automatically utilize all SMs in a high-end GPU with no code changes. This is why block-level independence is a fundamental principle of CUDA programming.

### 4.4 Warps and SIMD Hardware

#### Warp Organization

Once a block is assigned to an SM, it is further divided into units of 32 threads called **warps**:

- A warp is the basic scheduling unit within an SM
- All threads in a warp execute the same instruction at the same time (SIMD)
- Warp size is 32 threads in current NVIDIA GPUs
- Warps are formed by consecutive thread IDs within a block

Examples of warp partitioning:
- Threads 0-31 form the first warp
- Threads 32-63 form the second warp
- And so on...

> 📝 **Note**: If block size is not a multiple of 32, the last warp will be partially filled with inactive threads (padding).

#### Warp Formation in Multidimensional Blocks

For multidimensional thread blocks, threads are first linearized in row-major order:

```
For a 8×4 block (2D):
┌───┬───┬───┬───┬───┬───┬───┬───┐
│ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │ ← Warp 0 (threads 0-31)
├───┼───┼───┼───┼───┼───┼───┼───┤
│ 8 │ 9 │10 │11 │12 │13 │14 │15 │
├───┼───┼───┼───┼───┼───┼───┼───┤
│16 │17 │18 │19 │20 │21 │22 │23 │
├───┼───┼───┼───┼───┼───┼───┼───┤
│24 │25 │26 │27 │28 │29 │30 │31 │
└───┴───┴───┴───┴───┴───┴───┴───┘
```

> 🔍 **Insight**: Understanding warp formation is crucial for optimizing thread organization and memory access patterns. For example, using block dimensions that are multiples of 32 in the x-dimension can lead to better performance.

### 4.5 Control Divergence

#### SIMD Execution and Divergence

SIMD (Single Instruction, Multiple Data) hardware executes the same instruction across all threads in a warp:

- Efficient when all threads take the same execution path
- Inefficient when threads take different paths (e.g., due to conditionals)

```c
__global__ void divergentCode(int* data, int* result) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // This conditional causes control divergence within warps
    if (i % 2 == 0) {
        result[i] = data[i] * 2;  // Even threads
    } else {
        result[i] = data[i] + 10; // Odd threads
    }
}
```

#### How Divergence Affects Execution

When threads in a warp diverge:

1. The hardware executes each path serially (multiple passes)
2. Threads not on the current path are inactive (masked)
3. Performance decreases proportionally to the number of different paths

```
Warp execution with divergence:
┌───────────────────────┐
│ if (i % 2 == 0) {     │ All threads evaluate condition
├───────────────────────┤
│   result[i] = data[i] │ Only even threads active (odd masked)
│   * 2;                │
├───────────────────────┤
│ } else {              │
│   result[i] = data[i] │ Only odd threads active (even masked)
│   + 10;               │
├───────────────────────┤
│ }                     │
└───────────────────────┘
```

> ⚠️ **Performance warning**: Control divergence can reduce performance by up to 32× in worst-case scenarios (when each thread in a warp takes a different path).

#### Identifying and Minimizing Divergence

Common causes of control divergence:

1. Conditionals based on thread ID: `if (threadIdx.x < 16)`
2. Data-dependent conditionals: `if (data[i] > threshold)`
3. Loops with variable iteration counts: `for (int j = 0; j < data[i]; j++)`

Strategies to minimize divergence:

1. Align conditionals with warp boundaries when possible
2. Restructure algorithms to avoid thread-dependent conditionals
3. Consider sorting data to group similar execution paths together

> 🔍 **Practical tip**: Some divergence is unavoidable, especially for boundary checks. Focus optimization efforts on the most performance-critical sections of your kernels.

### 4.6 Warp Scheduling and Latency Tolerance

#### Oversubscription of Threads

SMs are assigned more warps than they can execute simultaneously:

- Only a subset of assigned warps can execute at any given time
- This deliberate oversubscription enables latency hiding

Example: A100 SM has 64 cores but can have up to 2,048 threads (64 warps) assigned

```
                  ┌──────────────────────┐
                  │ Streaming Multiprocessor │
                  │                      │
 ┌─────────┐      │  ┌─────────────────┐ │
 │ Active  │      │  │ Execution Units │ │
 │ Warps   │──────┼─►│   (64 Cores)    │ │
 └─────────┘      │  └─────────────────┘ │
      │           │                      │
      │           │  ┌─────────────────┐ │
      └───────────┼─►│    Scheduler    │ │
                  │  └─────────────────┘ │
 ┌─────────┐      │          ▲           │
 │ Stalled │      │          │           │
 │ Warps   │──────┘          │           │
 └─────────┘                 │           │
                             │           │
 waiting for memory, etc.    └───────────┘
```

#### Latency Hiding Mechanism

When a warp encounters a long-latency operation (e.g., global memory access):

1. The warp stalls, waiting for the operation to complete
2. The SM's warp scheduler selects another ready warp for execution
3. The stalled warp resumes once its operation completes and it's selected again

> 🔑 **Key insight**: This is why GPUs don't need large caches like CPUs – they hide memory latency through massive thread parallelism rather than caching.

#### Zero-Overhead Thread Scheduling

- Warp context switches occur in hardware with no overhead
- All warp states are kept resident on the SM
- No register saving/restoring as in traditional context switches
- Warps are selected for execution based on a priority mechanism

> 💡 **Understanding**: This latency hiding is why GPUs can dedicate more chip area to arithmetic units instead of caches and branch prediction – parallelism is used to hide latency rather than trying to eliminate it.

### 4.7 Resource Partitioning and Occupancy

#### SM Resource Limitations

Each SM has limited resources that must be shared among resident blocks:

1. **Registers**: Fast on-chip memory for thread-private variables
2. **Shared Memory**: On-chip memory shared within a block
3. **Thread slots**: Maximum number of threads per SM
4. **Block slots**: Maximum number of blocks per SM

Example A100 constraints:
- 65,536 registers per SM
- 164 KB shared memory per SM
- 2,048 max threads per SM
- 32 max blocks per SM

#### Occupancy Definition

**Occupancy** is the ratio of active warps to the maximum possible warps on an SM:

```
Occupancy = Active Warps / Maximum Warps per SM
```

Higher occupancy typically provides better latency hiding, but is not always correlated with peak performance.

> 📝 **Note**: 100% occupancy is not always necessary for optimal performance. Many kernels achieve peak performance at 50-75% occupancy.

#### Resource-Limited Occupancy Examples

1. **Register-limited example**:
   - SM supports 2,048 threads (64 warps)
   - Kernel uses 32 registers per thread
   - Total registers needed: 2,048 threads × 32 registers = 65,536 registers
   - If SM has only 65,536 registers, occupancy is 100%
   - If kernel used 40 registers, only 1,638 threads could be active (80% occupancy)

2. **Block-limited example**:
   - SM supports 16 blocks maximum
   - Kernel uses 64 threads per block
   - Maximum warps = 16 blocks × 64 threads ÷ 32 threads/warp = 32 warps
   - If SM supports 64 warps, occupancy would be 50%

> ⚠️ **Performance cliff warning**: When a resource limit is reached, adding just one more register per thread or a bit more shared memory can dramatically reduce occupancy, causing a "performance cliff."

#### Balancing Resource Usage

Strategies for optimizing occupancy:
1. Use fewer registers (compiler flags like `--maxrregcount`)
2. Use smaller thread blocks to increase the number of blocks per SM
3. Reduce shared memory usage
4. Consider kernel splitting to reduce per-thread resource needs

> 🔍 **Practical tip**: Use the CUDA Occupancy Calculator or `cudaOccupancyMaxActiveBlocksPerMultiprocessor()` to determine the limiting factor for your kernel.

### 4.8 Querying Device Properties

CUDA provides API functions to query device capabilities at runtime:

```c
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);  // Get properties for device 0

// Display key properties
printf("Device name: %s\n", prop.name);
printf("Compute capability: %d.%d\n", prop.major, prop.minor);
printf("SMs: %d\n", prop.multiProcessorCount);
printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
printf("Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
printf("Max blocks per SM: %d\n", prop.maxBlocksPerMultiProcessor);
printf("Shared memory per SM: %lu KB\n", prop.sharedMemPerMultiprocessor / 1024);
printf("Registers per SM: %d\n", prop.regsPerMultiprocessor);
```

> 💡 **Practical application**: Use device properties to dynamically adjust kernel launch parameters based on the specific GPU your application is running on.

### 4.9 Key Takeaways

- **GPU Architecture**: GPUs consist of multiple SMs, each containing many CUDA cores that share control logic and memory resources.

- **Block Scheduling**: Blocks are assigned to SMs for independent execution, enabling transparent scalability across different GPU models.

- **Thread Synchronization**: `__syncthreads()` provides barrier synchronization within a block, but not across blocks.

- **Warp Execution**: Threads are executed in warps of 32 threads following the SIMD model, with all threads in a warp executing the same instruction simultaneously.

- **Control Divergence**: When threads in a warp take different paths, execution serializes, reducing performance. Minimize divergence when possible.

- **Latency Hiding**: GPUs tolerate long-latency operations by maintaining many more threads than can execute simultaneously, switching between them with zero overhead.

- **Occupancy**: The ratio of active warps to maximum possible warps affects performance. It's limited by register, shared memory, thread, and block constraints.

- **Resource Balance**: Optimizing for peak performance requires balancing register usage, shared memory, and thread organization to achieve sufficient (but not necessarily maximum) occupancy.

***

**Exercise Ideas**:
1. Calculate the occupancy for a kernel with different resource requirements on your specific GPU
2. Analyze a kernel for potential control divergence and suggest optimizations
3. Experiment with different block sizes to find the optimal configuration for a specific algorithm


## 5. Memory architecture and data Locality

> 💡 **Core Concept**: This chapter explores GPU memory hierarchy and optimization techniques that maximize performance by efficiently using the various memory types and minimizing global memory traffic.

This chapter focuses on the on-chip memory architecture of the GPU and how to organize and position data for efficient access by a massive number of threads.

### 5.1 Importance of memory access efficiency

> 🔑 **Key insight**: Many GPU applications are limited by memory bandwidth rather than computational power, making memory access optimization critical for performance.

#### Memory vs. Compute Bound Applications

The **compute-to-memory access ratio** (also called **arithmetic intensity**) measures how many floating-point operations (FLOPs) are performed per byte accessed from global memory:

```
Compute-to-Memory Ratio = FLOPs performed / Bytes accessed from global memory
```

This ratio determines whether an application is:
- **Memory-bound**: Performance limited by memory bandwidth (low arithmetic intensity)
- **Compute-bound**: Performance limited by computational throughput (high arithmetic intensity)

Modern GPUs can perform thousands of floating-point operations in the time it takes to access global memory once, creating a significant performance gap:

```
┌─────────────────────────────────────────────────────────┐
│                  Performance Comparison                  │
│                                                          │
│ A100 GPU:                                                │
│ ┌───────────────────┐                                    │
│ │ Compute: 19.5 TFLOPS                                   │
│ └───────────────────┘                                    │
│                                                          │
│ ┌───────────────────┐                                    │
│ │ Memory: 1.55 TB/s                                      │
│ └───────────────────┘                                    │
│                                                          │
│ Ratio: ~12.5 FLOPs possible per byte accessed           │
└─────────────────────────────────────────────────────────┘
```

#### Identifying Memory Bottlenecks

Memory bottlenecks can be identified by:
1. Profiling tools showing high memory utilization
2. Performance not scaling with increased compute resources
3. Calculating theoretical arithmetic intensity and comparing with hardware capabilities

#### Strategies for Improving Memory Efficiency

1. **Data reuse**: Maximize operations per memory access
2. **Coalesced access**: Ensure threads in a warp access contiguous memory
3. **Memory hierarchy utilization**: Use faster memory types (shared, constant) when possible
4. **Minimizing data transfers**: Reduce host-device communication
5. **Tiling**: Partition data to fit in faster memory levels

> ⚠️ **Common pitfall**: Many developers focus on optimizing computational aspects when memory access patterns are actually the limiting factor.

### 5.2 CUDA memory types

CUDA provides several memory types, each with different scope, lifetime, and performance characteristics:

```
┌───────────────────── CUDA Memory Hierarchy ─────────────────────┐
│                                                                  │
│  Fastest   ┌───────────┐                                         │
│    ▲       │ Registers │ Thread-private, on-chip                 │
│    │       └───────────┘                                         │
│    │       ┌───────────┐                                         │
│    │       │   Shared  │ Block-accessible, on-chip               │
│    │       └───────────┘                                         │
│    │       ┌───────────┐                                         │
│    │       │ Constant  │ Read-only, cached, all threads          │
│    │       └───────────┘                                         │
│    │       ┌───────────┐                                         │
│    │       │   Local   │ Thread-private, in global memory        │
│    │       └───────────┘                                         │
│    │       ┌───────────┐                                         │
│  Slowest   │   Global  │ Accessible by all threads and host      │
│    ▼       └───────────┘                                         │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

#### Memory Type Characteristics

| Memory Type | Access Scope | Access Speed | Size      | Lifetime      | Declaration                        |
|-------------|--------------|--------------|-----------|---------------|------------------------------------|
| Register    | Thread       | Fastest      | Limited   | Thread        | Automatic variables (non-arrays)   |
| Shared      | Block        | Very fast    | ~100KB/SM | Block         | `__shared__ int s;`                |
| Constant    | Grid (read)  | Fast (cached)| ~64KB     | Application   | `__constant__ int c;`              |
| Local       | Thread       | Slow         | Large     | Thread        | Automatic arrays, large structures |
| Global      | Host & Grid  | Slowest      | GB range  | Application   | `__device__ int g;`                |

#### Global Memory

Global memory:
- Accessible by both host and device
- Largest capacity (several GB)
- Highest latency (400-800 cycles)
- Persists for the entire application lifetime
- Primary means of communication between host and device

Usage pattern:
```c
// Host allocates and transfers data
float *d_data;
cudaMalloc((void**)&d_data, size);
cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

// Kernel accesses global memory
__global__ void kernel(float *data) {
    // Read/write global memory
    float value = data[threadIdx.x];
    data[threadIdx.x] = value * 2.0f;
}
```

#### Constant Memory

Constant memory:
- Read-only from device perspective
- Cached and optimized for broadcast (all threads reading same address)
- Limited size (~64KB total)
- Declared outside any function with `__constant__` qualifier

Example:
```c
// Declare in global scope
__constant__ float constData[256];

// Host code to initialize
cudaMemcpyToSymbol(constData, h_data, size);

// Kernel using constant memory
__global__ void kernel() {
    // All threads reading same index is efficient
    float value = constData[5];
    
    // Different threads reading different indices uses cache
    float myValue = constData[threadIdx.x];
}
```

#### Shared Memory

Shared memory:
- On-chip memory shared by all threads in a block
- Much faster than global memory (100x lower latency)
- Limited size (up to ~100KB per SM in modern GPUs)
- Declared with `__shared__` qualifier
- Ideal for data reuse within a block

Example:
```c
__global__ void kernel() {
    // Static allocation
    __shared__ float sharedData[256];
    
    // Or dynamic allocation (size set at kernel launch)
    extern __shared__ float dynamicShared[];
    
    // Each thread loads one element from global to shared
    sharedData[threadIdx.x] = globalData[threadIdx.x];
    
    // Synchronize to ensure all data is loaded
    __syncthreads();
    
    // Now threads can access each other's data efficiently
    float value = sharedData[255 - threadIdx.x];
}
```

> ⚠️ **Important**: Always use `__syncthreads()` after writing to shared memory before other threads read those values.

#### Local Memory

Local memory:
- Thread-private storage but physically located in global memory
- Used for automatic arrays and register spilling
- Has same high latency as global memory
- Compiler-managed; not directly controlled by programmer

When local memory is used:
```c
__global__ void kernel() {
    // Large array likely placed in local memory
    float largeArray[1000];
    
    // Complex function with many variables might spill to local memory
    for (int i = 0; i < 100; i++) {
        float temp1, temp2, temp3, /*...many more local variables*/;
        // Complex calculations causing register pressure
    }
}
```

#### Register Memory

Register memory:
- Fastest memory type
- Thread-private variables
- Limited quantity (e.g., 65,536 32-bit registers per SM)
- Allocated by compiler automatically for scalar variables

> 🔍 **Tip**: Use the `--ptxas-options=-v` compiler flag to see register usage for your kernels.

### 5.3 Tiling for reduced memory traffic

> 💡 **Core Concept**: Tiling partitions data into chunks that fit in shared memory, allowing multiple threads to reuse the same data and dramatically reducing global memory traffic.

#### The Tiling Principle

Tiling leverages the memory hierarchy by:
1. Loading a subset of data from global memory into shared memory
2. Having multiple threads reuse this data for calculations
3. Moving to the next subset until all data is processed

```
┌──────────────────── Global Memory ─────────────────────┐
│                                                         │
│  ┌─────┬─────┬─────┬─────┐                             │
│  │Tile1│Tile2│Tile3│Tile4│ ... and so on               │
│  └─────┴─────┴─────┴─────┘                             │
│     │                                                   │
│     │ Load one tile at a time                           │
│     ▼                                                   │
│  ┌─────┐                                                │
│  │Tile │ ◄── Multiple threads process using shared mem  │
│  └─────┘                                                │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

#### Benefits of Tiling

1. **Reduced global memory traffic**: Data is loaded once into shared memory and reused many times
2. **Improved memory bandwidth utilization**: Coalesced accesses when loading tiles
3. **Higher arithmetic intensity**: More operations per global memory access
4. **Better cache utilization**: Working with subsets that fit in cache

#### Memory Traffic Reduction Analysis

Consider matrix multiplication where each output element requires N multiplications:

| Approach | Global Memory Accesses | Compute Ops | Ratio (Ops/Access) |
|----------|------------------------|-------------|-------------------|
| Naive    | 2N + 1 per element    | N           | ~0.5              |
| Tiled (T×T) | 2N/T + 1 per element | N           | ~T/2              |

For a tile size of 16×16, memory traffic is reduced by a factor of 16, increasing arithmetic intensity by 16×.

#### Tiling Requirements

For tiling to be effective:
1. Computation must have **data reuse** opportunities
2. The reused data must fit in **shared memory**
3. Tiles must be **independent** or have minimal dependencies
4. Thread block dimensions should match tile dimensions for efficient access

> 🔑 **Key insight**: The ideal tile size balances memory usage, thread count, and memory access patterns. Too small tiles underutilize shared memory; too large tiles reduce occupancy.

#### Common Tiling Applications

Tiling works well for:
- Matrix operations (multiplication, transposition)
- Convolutions and stencil operations
- Image processing filters
- Reduction operations with intermediate results

> 🔍 **Practical tip**: Start with 16×16 or 32×8 tiles and adjust based on profiling results. The tile size should be a multiple of 32 (warp size) for optimal performance.


### 5.4 A tiled matrix multiplication kernel

> 💡 **Core Concept**: Tiling uses shared memory to reduce global memory traffic by loading small blocks of input matrices that multiple threads can reuse.

#### How Tiling Improves Performance

The basic matrix multiplication kernel from Chapter 3 required each thread to:
- Read N elements from matrix M (one row)
- Read N elements from matrix N (one column)
- Perform N multiply-add operations
- Write 1 element to matrix P

This resulted in (2N+1) memory operations for N computations, giving a compute-to-memory ratio of 1:2, which is memory-bound.

With tiling, a block of threads collaboratively:
1. Loads TILE_WIDTH×TILE_WIDTH elements from M and N into shared memory
2. Each thread uses these cached elements for partial dot product calculations
3. Moves to the next tile until the full dot product is complete

This reduces global memory accesses by a factor of TILE_WIDTH, potentially increasing performance by the same factor.

#### Key Implementation Aspects

- **Shared Memory Arrays**: `Mds` and `Nds` hold tiles of matrices M and N
- **Thread Indexing**: Each thread calculates its P element's location using block and thread indices
- **Phased Computation**: Outer loop processes one tile at a time (strip-mining)
- **Barrier Synchronization**: Required before and after using shared memory to ensure data is loaded/used properly
- **Memory Access Patterns**: Structured to achieve coalesced access for better performance

Tiling changes the compute-to-memory ratio from 0.25 OP/B to 4 OP/B with a 16×16 tile, potentially increasing performance by 16×.

> ⚠️ **Limitations**: This implementation assumes matrices are square and have dimensions that are multiples of the tile width.

#### Tiled Matrix Multiplication Code

```c
__global__ void MatrixMulKernel(float* M, float* N, float* P, int Width) {
    // Shared memory for the tiles
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];
    
    // Thread and block indices
    // These are placed into registers and are terminated after use
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    // Row and column indices for the output element
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;
    
    // Accumulator for the dot product
    float Pvalue = 0;
    
    // Loop over tiles
    for (int ph = 0; ph < Width/TILE_WIDTH; ++ph) {
        // Load M tile into shared memory
        Mds[ty][tx] = M[Row*Width + ph*TILE_WIDTH + tx];
        // Load N tile into shared memory
        Nds[ty][tx] = N[(ph*TILE_WIDTH+ty)*Width + Col];
        
        // Ensure all threads have loaded their elements
        __syncthreads();
        
        // Compute partial dot product using the tile
        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        
        // Ensure all threads have finished using the tile
        __syncthreads();
    }
    
    // Write result to global memory
    P[Row*Width + Col] = Pvalue;
}
```

#### Launching the Kernel

```c
// Calculate grid and block dimensions
dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
dim3 dimGrid(Width/TILE_WIDTH, Width/TILE_WIDTH);

// Launch kernel
MatrixMulKernel<<<dimGrid, dimBlock>>>(d_M, d_N, d_P, Width);
```

> 💡 **Performance Note**: With a 16×16 tile on an A100 GPU, this optimization allows achieving up to 6220 GFLOPS versus 389 GFLOPS for the non-tiled version, though this is still only 32% of the A100's peak 19,500 GFLOPS.

### 5.5 Boundary checks

> ⚠️ **Problem**: When matrix dimensions aren't multiples of tile width, threads may try to access non-existent elements, causing incorrect results or crashes.

#### Handling Matrix Boundaries

When working with matrices whose dimensions aren't multiples of `TILE_WIDTH`, three issues arise:

1. Threads might access elements past the end of a row (accessing incorrect data)
2. Threads might access elements past the end of a column (accessing memory outside the allocated array)
3. These boundary issues occur in all phases of execution, not just the last phase

#### Boundary Check Solution

The solution requires three separate checks:

1. **For loading M tile elements**: `Row < Width && (ph*TILE_WIDTH+tx) < Width`
2. **For loading N tile elements**: `(ph*TILE_WIDTH+ty) < Width && Col < Width`
3. **For storing P results**: `Row < Width && Col < Width`

When a thread would load an invalid element, it should put 0.0 in shared memory instead, which won't affect dot product calculations.

#### Tiled Matrix Multiplication With Boundary Checks

```c
__global__ void MatrixMulKernel(float* M, float* N, float* P, int Width) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;
    
    float Pvalue = 0;
    
    for (int ph = 0; ph < (Width+TILE_WIDTH-1)/TILE_WIDTH; ++ph) {
        // Load M tile with boundary check
        if (Row < Width && ph*TILE_WIDTH+tx < Width)
            Mds[ty][tx] = M[Row*Width + ph*TILE_WIDTH + tx];
        else
            Mds[ty][tx] = 0.0;
            
        // Load N tile with boundary check
        if (ph*TILE_WIDTH+ty < Width && Col < Width)
            Nds[ty][tx] = N[(ph*TILE_WIDTH+ty)*Width + Col];
        else
            Nds[ty][tx] = 0.0;
        
        __syncthreads();
        
        // Calculate partial dot product
        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        
        __syncthreads();
    }
    
    // Store result with boundary check
    if (Row < Width && Col < Width)
        P[Row*Width + Col] = Pvalue;
}
```

> 💡 **Key insight**: Every memory access needs its own boundary check to ensure indices are within array bounds. This makes the kernel work with arbitrary matrix dimensions, not just multiples of the tile width.

This implementation is almost a general matrix multiplication kernel, with one limitation remaining: it only works for square matrices (future optimizations would handle rectangular matrices with different dimensions).


### 5.6 Impact of memory usage on occupancy

> 💡 **Core Concept**: Memory resources (registers and shared memory) directly affect occupancy, which influences performance through latency hiding. Finding the optimal balance is critical for kernel optimization.

#### Understanding Occupancy and Resource Limits

**Occupancy** is the ratio of active warps to the maximum possible warps on a Streaming Multiprocessor (SM). It's constrained by four main factors:

1. **Register usage**: Registers are allocated per thread
2. **Shared memory usage**: Shared memory is allocated per block
3. **Block size**: Number of threads per block
4. **Hardware limits**: Maximum warps/threads/blocks per SM

```
┌─────────────────── Resource-Occupancy Relationship ────────────────────┐
│                                                                       │
│                             ┌─────────────┐                             │
│         ┌─────────────────►│   Maximum   │◄─────────────────┐          │
│         │                   │ SM Capacity │                  │          │
│         │                   └─────────────┘                  │          │
│         │                          ▲                         │          │
│         │                          │                         │          │
│         │                          │                         │          │
│  ┌──────────────┐          ┌──────────────┐          ┌──────────────┐  │
│  │   Register   │          │    Shared    │          │    Block     │  │
│  │    Usage     │          │    Memory    │          │    Size      │  │
│  └──────────────┘          └──────────────┘          └──────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Register Usage Impact

Every thread requires registers for its variables. Register usage affects occupancy as follows:

- More registers per thread → fewer threads can be active simultaneously
- Example: On an SM with 65,536 registers and 2,048 maximum threads:
  - Using 32 registers/thread → 100% occupancy (65,536 ÷ 32 = 2,048 threads)
  - Using 64 registers/thread → 50% occupancy (65,536 ÷ 64 = 1,024 threads)
  - Using 128 registers/thread → 25% occupancy (65,536 ÷ 128 = 512 threads)

> ⚠️ **Important**: The compiler decides register allocation, not the programmer. Use `--ptxas-options=-v` to see how many registers your kernel uses.

#### Shared Memory Impact

Shared memory is allocated per block, impacting how many blocks can reside on an SM:

- More shared memory per block → fewer blocks can be active simultaneously
- Example: On an SM with 64KB shared memory:
  - Using 16KB/block → 4 blocks can reside (64KB ÷ 16KB = 4 blocks)
  - Using 32KB/block → 2 blocks can reside (64KB ÷ 32KB = 2 blocks)

If this block count becomes the limiting factor, it directly affects occupancy.

#### Occupancy Calculator Example

Suppose we have a kernel with these characteristics:
- 64 registers per thread
- 8KB shared memory per block
- 256 threads per block
- Hardware: SM with 65,536 registers, 64KB shared memory, 2,048 max threads

The limiting factors would be:
- Register limit: 65,536 ÷ 64 = 1,024 threads (8 blocks of 128 threads)
- Shared memory limit: 64KB ÷ 8KB = 8 blocks (8 × 256 = 2,048 threads)
- Thread limit: 2,048 threads (8 blocks of 256 threads)

The most restrictive is the register limit, allowing only 1,024 threads (50% occupancy).

#### Occupancy vs. Performance Curve

The relationship between occupancy and performance is not linear:

```
┌───────────────── Typical Occupancy-Performance Curve ────────────────┐
│                                                                       │
│ Performance                                                           │
│     ▲                                                                 │
│     │                                      ┌─────────────────────     │
│     │                                ┌─────┘                          │
│     │                         ┌──────┘                                │
│     │                    ┌────┘                                       │
│     │               ┌────┘                                            │
│     │          ┌────┘                                                 │
│     │      ┌───┘                                                      │
│     │  ┌───┘                                                          │
│     │┌─┘                                                              │
│     └─────────┬─────────┬─────────┬─────────┬─────────► Occupancy    │
│              20%        40%        60%        80%       100%         │
│                                                                       │
│                Performance often plateaus before 100% occupancy       │
└───────────────────────────────────────────────────────────────────────┘
```

> 🔑 **Key insight**: Many kernels achieve peak performance at 40-70% occupancy. Beyond this point, increasing occupancy may not improve performance and could even be counterproductive if it requires sacrificing other optimizations.

#### Tools for Analyzing Occupancy

1. **NVIDIA Nsight Compute**: Provides detailed occupancy analysis
2. **CUDA Occupancy Calculator**: Spreadsheet tool to experiment with different configurations
3. **`cudaOccupancyMaxPotentialBlockSize`**: Runtime API to find optimal block size
   ```c
   int minGridSize;
   int blockSize;
   cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, 
                                     MyKernel, 0, 0);
   ```

#### Strategies for Optimizing Memory Usage

1. **Register optimization**:
   - Use compiler flags like `--maxrregcount=N` to limit register usage
   - Break complex kernels into smaller ones
   - Recompute values instead of storing in registers when beneficial

2. **Shared memory optimization**:
   - Use appropriate tile sizes that balance occupancy with memory efficiency
   - Consider using multiple smaller tiles instead of one large tile
   - Dynamically allocate only what's needed with `extern __shared__`

3. **Thread block size selection**:
   - Choose sizes that are multiples of 32 (warp size)
   - Consider using rectangular blocks (e.g., 32×8) rather than square ones (16×16)
   - Experiment with different configurations, as optimal values vary by kernel

> 🔍 **Practical tip**: Profile before optimizing. Measure the actual impact of your changes, as intuition about performance can be misleading. Sometimes using more registers to avoid recomputation is better despite reducing occupancy.

#### Balancing Memory Resources

Finding the optimal configuration is often a balancing act:

1. **Register vs. recomputation**: Using more registers reduces occupancy but may be faster than recomputing values
2. **Shared memory vs. global memory**: Using shared memory reduces occupancy but significantly speeds up memory access
3. **Block size vs. resources**: Larger blocks improve shared memory utilization but consume more resources per block

> 💡 **Understanding**: There's no universal "best" configuration - the optimal balance depends on the specific algorithm, device capabilities, and memory access patterns.

#### Optimization Workflow

1. **Measure**: Profile kernel with NSight Compute to identify limiting resources
2. **Analyze**: Determine if occupancy is actually limiting performance
3. **Experiment**: Try different configurations of block size, shared memory, and compiler flags
4. **Benchmark**: Measure performance under each configuration
5. **Iterate**: Refine based on results to find the optimal configuration

> 🔑 **Key insight**: Don't optimize occupancy in isolation. Consider the overall goal of maximizing throughput, which might involve trading lower occupancy for better memory access patterns or instruction-level optimizations.


### Exercises

1. Consider matrix addition. Can one use shared memory to reduce the global memory bandwidth consumtion? Hint: Analyze the elements that are accessed by each thread an see whether there is any commonality between threads.
I guess you could, but i dont think it would have any performance difference because in a matrix addition it is just an element wise op that only contains one op. The whole point of using shared for
mat mul is because there are elements in each matrix that are "reused" across different ops, i.e. you need to multiply element i with the entire col j so there is reuse amongst elements in a handwavy sense.

2. Draw the equivalent of Fig. 5.7 for a 8x8 matrix mult. with 2x2 tiling and 4x4 tiling. Verify that the reduction in global memory bandwidth is indeed proportional to the dimension size of the tiles. 
For ease of understanding lets just look at total number of tiles we need to load and see how that changes as we scale tile size.
for 8x8 matrix at a tile size of 2 we have (width / tile)^2 * 2 for (P = A @ B) which gives us (8 / 2)^2 * 2 = 32 tiles to load
for 8x8 matrix at a tile size of 4 we have (8 / 4)^2 * 2 = 8 tiles to load
So yes, the reduction in global memory bandwidth is proportional to the dimension of the tile size. 

3. What type of incorrect execution behavior can happen if one forgot to use one or both __syncthreads() in the kernel of fig. 5.9?
it would be a race condition on either the loading of the tiles or the dot prod itself.
If its in the tile loading then some tile in P would be an incorrect comp between tiles in A or B
If its in the dot prod op itself then there could be a situation where a thread has already executed its op and then goes back and redoes its op because it doesn't know it needs to wait for the remaining threads. 

4. Assuming that capacity is not an issue for registers or shared memory, give one important reason why it would be valuable to use shared memory instead of registers to hold values fetched from global memory?
One important reason is that all threads in a block have access to the data in shared memory, while registers only have access to the data for that thread. This would probably be very important for a sorting algorithm where
threads need to iteratively access data across the entire block, rather than just their data in the register. 

5. For our tiled matrix multiplication kernel, if we use a 32x32 tile, what isthe reduction of memory bandwith usage for input matrices M and N?
By a factor of 32?

6. Assume that a CUDA kernel is launched with 1000 thread blocks, each of which as 512 threads. If a variable is declared as a local variable in the kernel, how many versions of the variable will be created through the lifetime of the execution of the kernel?
Each thread will have its own copy so there will be 512,000 versions of the variable throughout the kernel. 

7. In the previous question, if a variable is declared as a shared memory variable, how many versions of the variable will be created through the lifetime of the execution of the kernel?
1000 because shared memory is blockwise so there will be one variable for each block. 

8. Consider performing a matrix multiplication of two input matrices with dimensions NxN. How many times is each element in the input matrices requested from global memory when:
a. There is not tiling?
2n
b. Tiles of size TxT are used?
2n/T

9. A kernel performs 36 floating-point operations and seven 32-bit global memory accesses per thread. For each of the following device properties, indicate whether this kernel is compute-bound of memory-bound.
a. Peak FLOPS = 200 GFLOPS, peak memory bandwidth = 100 GB/second
memory bound
b. Peak FLOPS = 300 GFLOPS, peak memory bandwidth = 250 GB/second
compute bound
```
Calculate the kernel's arithmetic intensity
Arithmetic intensity = Computation / Memory traffic
Computation: 36 floating-point operations per thread
Memory traffic: 7 × 32-bit (4-byte) accesses = 28 bytes per thread
Arithmetic intensity = 36 FLOPs / 28 bytes = 1.29 FLOPs/byte
Calculate each device's compute-to-memory ratio
This represents how many floating-point operations each device can perform per byte accessed:
Device A:
Peak compute: 200 GFLOPS = 200 × 10^9 FLOPs/second
Peak memory bandwidth: 100 GB/second
Device ratio = 200 × 10^9 / 100 × 10^9 = 2 FLOPs/byte
Device B:
Peak compute: 300 GFLOPS = 300 × 10^9 FLOPs/second
Peak memory bandwidth: 250 GB/second
Device ratio = 300 × 10^9 / 250 × 10^9 = 1.2 FLOPs/byte
3. Compare and determine the bottleneck
If kernel's arithmetic intensity < device ratio: memory-bound
If kernel's arithmetic intensity > device ratio: compute-bound
```

10. To manipulate tiles, a new CUDA programmer has written a device kernel that will transpose each tile in a matrix. THe tiles are of size BLOCK_WIDTH by BLOCK_WIDTH, and each of the dimensions of the matrix A
is known to be a multiple of BLOCK_WIDTH. The kernel invocation and code are shown below. BLOCK_WIDTH is known at compile time and could be set anywhere from 1 to 20.
```c
dim3 blockDim(BLOCK_WIDTH, BLOCK_WIDTH);
dim3 gridDim(A_width/blockDim.x, A_height/blockDim.y);
BlockTranspose<<<gridDim, blockDim>>>(A, A_width, A_height);

__global__ void BlockTranspose(float* A_elements, int A_width, int A_height) {
    __shared__ float blockA[BLOCK_WIDTH][BLOCK_WIDTH];

    int baseIdx = blockIdx.x * BLOCK_WIDTH + threadIdx.x;
    baseIdx += (blockIdx.y * BLOCK_WIDTH + threadIdx.y) * A_width;

    blockA[threadIdx.y][threadIdx.x] = A_elements[baseIdx];

    A_elements[baseIdx] = blockA[threadIdx.x][threadIdx.y];
}
```

a. Out of the possible range of values for BLOCK_WIDTH, for what values of BLOCK_WIDTH will this kernel function execute correctly on the device?
None of them will run because of a race condition
b. If the code does not execute correctly for all BLOCK_WIDTH values, what is the root cause of this incorrect execution behavior? Suggest a fix to the code to make it work for all BLOCK_WIDTH values.
you'll need to introduce a temp array to store values in becuase right now the transpose is reading and writing to the same matrice in global memory so each thread has no idea if its grabbing the correct value to transpose or not

11. Consider the following CUDA kernel and the corresponding host function that calls it:
```c
__global__ void foo_kernel(float* a, float* b) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    float x[4];
    __shared__ float y_s;
    __shared__ float b_s[128];
    for(unsigned int j = 0; j < 4; ++j) {
        x[j] = a[j*blockDim.x*gridDim.x + i];
    }
    if(threadIdx.x == 0) {
        y_s = 7.4f;
    }
    b_s[threadIdx.x] = b[i];
    __syncthreads();
    b[i] = 2.5f*x[0] + 3.7f*x[1] + 6.3f*x[2] + 8.5f*x[3]
            + y_s*b_s[threadIdx.x] + b_s[(threadIdx.x + 3)%128];
}
void foo(int* a_d, int* b_d) {
    unsigned int N = 1024;
    foo_kernel <<< (N + 128 - 1)/128, 128  >>>(a_d, b_d)
}
```

a. How many versions of the variable i are there?
1024, one for each thread
b. How many versions of the array x[] are there?
1024, one for each thread
c. How many versions of the variable y_s are there?
8, one for each block
d. How many versions of the array b_s[] are there?
8, one for each block
e. What is the amount of shared memory used per block (in bytes)?
4 bytes per float
1 floats for y_s per block
128 floats for b_s per block
4 * (1 + 128) = 516 bytes per block
f. What is the floating-point to global memory access ratio of the kernel in (OP/B)?
10FLOPs/24B = .42 OP/B

12. Consider a GPU with the following hardware limits: 2048 threads/SM, 32 blocks/SM, 64k registers/SM, and 96 KB of shared memory/SM. For each of the following kernel characteristics, specify whether the kernel
can achieve full occupancy. If not, specify the limiting factor.
a. The kernel uses 64 threads/block, 27 registers/thread, and 4KB of shared memory/SM.
yes the kernel can fit in the SM.
b. The kernel uses 256 threads/block, 32 registers/thread, and 8KB of shared memory/SM.
No the kernel cannot fit in the device. there are too many threads and registers to fit on this device. 


## 6. Performance Considerations

### 6.1 Memory coalescing

> **Core Concept**: Memory coalescing combines multiple memory accesses from threads in a warp into a single, efficient DRAM request, significantly improving global memory bandwidth utilization.

Global memory bandwidth is often a performance bottleneck in CUDA applications. Memory coalescing is a technique to efficiently utilize this bandwidth by organizing thread memory access patterns to match DRAM's burst-oriented architecture. This technique is often used alongside the tiling approach from Chapter 5 to maximize memory efficiency.

#### Why Coalescing Matters

The global memory of CUDA devices is implemented with DRAM, which has relatively high access latency compared to the GPU's computational speed. Modern DRAM designs use parallelism to deliver data in "bursts" - accessing a range of consecutive locations at once rather than individual addresses. Memory coalescing takes advantage of this architecture.

When threads in a warp execute a load instruction, the hardware detects whether they access consecutive global memory locations. If so, these accesses are combined into a single request for consecutive locations. For example, if thread 0 accesses location X, thread 1 accesses X+1, thread 2 accesses X+2, and so on, all these accesses will be coalesced into a single memory transaction.

#### Coalesced vs. Uncoalesced Access Patterns

**Row-Major Matrix (Coalesced):**
- In a matrix multiplication example with row-major storage, the array index is typically `k*Width+col`
- Since consecutive threads have consecutive values of `col`, they access consecutive memory addresses
- This creates coalesced access where threads in a warp access adjacent memory locations

**Column-Major Matrix (Uncoalesced):**
- With column-major storage, the array index becomes `col*Width+k`
- Consecutive threads now access memory locations that are Width elements apart
- This pattern cannot be coalesced, significantly reducing memory bandwidth utilization

#### Optimization Strategies

When memory access patterns aren't naturally coalesced, several strategies can help:

1. **Rearrange thread mapping**: Change how threads are assigned to data elements to create coalesced access patterns

2. **Modify data layout**: Store data in formats (usually row-major for CUDA) that enable coalesced access

3. **Corner turning**: Use shared memory as an intermediate buffer:
   - Load data from global memory in a coalesced pattern
   - Reorganize the data in shared memory
   - Access the reorganized data for computation

**Example: Corner Turning for Matrix Multiplication**
When multiplying matrices where one is in row-major and one in column-major format:
- For the row-major matrix, threads load elements as usual with coalesced access
- For the column-major matrix, assign consecutive threads to load consecutive elements in the same column rather than row
- After loading into shared memory, threads can access the data in any pattern without penalty

These techniques allow CUDA applications to effectively utilize the available global memory bandwidth, which is crucial for achieving high performance in memory-bound applications.


### 6.2 Hiding memory latency

> **Core Concept**: Modern DRAM systems use multiple parallel structures (bursts, banks, and channels) to hide memory access latency and maximize bandwidth utilization.

As we explained in Section 6.1, DRAM bursting is a form of parallel organization: Multiple locations are accessed in the DRAM core array in parallel. However, bursting alone is not sufficient to realize the level of DRAM access bandwidth required by modern processors. DRAM systems typically employ two more forms of parallel organization: **banks** and **channels**.

#### DRAM System Organization

At the highest level, a processor contains one or more channels. Each channel is a memory controller with a bus that connects a set of DRAM banks to the processor. Modern systems typically have one to eight channels, with multiple banks connected to each channel.

The data transfer bandwidth of a bus is defined by its width and clock frequency. Modern double data rate (DDR) busses perform two data transfers per clock cycle: one at the rising edge and one at the falling edge. For example:

- A 64-bit DDR bus with a 1 GHz clock has a bandwidth of 8B × 2 × 1 GHz = 16 GB/s
- A modern CPU might require at least 32 GB/s (2 channels)
- A modern GPU might require 256 GB/s (16 channels)

#### The Role of Banks in Memory Performance

For each channel, the number of banks that connect to it is determined by the need to fully utilize the data transfer bandwidth of the bus. Each bank contains:
- An array of DRAM cells
- Sensing amplifiers for accessing cells
- Interface for delivering bursts of data to the bus

When a single bank is connected to a channel, memory access is highly inefficient:

1. The long latency for cell access (decoder enabling cells and charge sharing with amplifiers) must complete before data transfer
2. The data transfer time is typically much shorter than the access latency
3. If the ratio of access latency to data transfer time is 20:1, channel utilization would be only 4.8%

#### Banking for Improved Bandwidth Utilization

When multiple banks are connected to a channel, accesses can be initiated in parallel:

- While one bank is transferring data, other banks can be accessing their cell arrays
- With two banks, channel utilization can potentially double
- If the ratio of cell array access latency to data transfer time is R, at least R+1 banks are needed to fully utilize channel bandwidth

More banks are beneficial for two key reasons:
1. Reducing **bank conflicts** (multiple accesses targeting the same bank)
2. Providing sufficient memory capacity while maintaining reasonable access latencies

#### Thread Parallelism and Memory Organization

There is a critical connection between thread execution and DRAM organization:

- To achieve high memory bandwidth, sufficient threads must make simultaneous memory accesses
- Maximizing occupancy ensures enough threads are resident on SMs to:
  - Hide core pipeline latency (utilizing instruction throughput)
  - Hide DRAM access latency (utilizing memory bandwidth)
- Optimal performance requires memory accesses to be distributed across channels and banks, with coalesced access to each bank

#### Interleaved Data Distribution

Modern memory systems distribute array elements across channels and banks in an interleaved pattern:

- Elements are spread sequentially across channels first, then banks
- This ensures even small arrays utilize multiple channels and banks
- Example: With 4 channels and 2 banks per channel:
  - Elements [0-1] → Channel 0, Bank 0
  - Elements [2-3] → Channel 1, Bank 0
  - Elements [4-5] → Channel 2, Bank 0
  - Elements [6-7] → Channel 3, Bank 0
  - Elements [8-9] → Channel 0, Bank 1
  - And so on...

#### Practical Example: Tiled Matrix Multiplication

In tiled matrix multiplication:

- During each phase, thread blocks access different tiles of input matrices
- These accesses are distributed across channels and banks based on memory layout
- Multiple blocks executing in parallel create simultaneous memory requests to different memory subsystems
- GPU caches can combine duplicate accesses from different thread blocks
- As matrix size increases, memory accesses utilize more channels and banks

This demonstrates the symbiotic relationship between thread parallelism and memory organization:
- Thread parallelism creates simultaneous memory requests needed to utilize parallel DRAM structures
- Effective utilization of DRAM channels and banks is essential for achieving high execution throughput

> **Performance Insight**: For optimal memory performance, applications should maximize thread occupancy while ensuring memory accesses are well-distributed across channels and banks. This requires attention to both thread organization and data layout.



