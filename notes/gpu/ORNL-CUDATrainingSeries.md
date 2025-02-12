# Oak Ridge National Lab CUDA Training Series

## Contents

CUDA C++ Basics
CDUA Shared Memory
CUDA Optimization Part 1
CUDA Optimization Part 2
Atomics, Reductions, and Warp Shuffle

## CUDA C++ Basics

CUDA: Compute Unified Device Architecture

Goals:
- vector addition
- write and launch CUDA C++ kernels
- Manage GPU memory

Heterogeneous Programming:
- CPU and GPU are different types of processors
- CPU is general purpose
- GPU is specialized for parallel computations

Vector Addition - an example of "embarrassingly parallel" problem
- each element in the output vector is independent of the others (a_i + b_i = c_i)

GPU kernels: device code
__global__ void mykernel(void) {

}
__global__ indicates a function that:
- runs on the GPU
- is called from host code

mykernel<<<1, 1>>>(...);
Triple angle brackets mark a call to device code
- also called a "kernel launch"
- params inside <<<>>> are cuda kernel execution parameters

Host and device memory are separate entities
- Device pointers point to GPU memory
    - Typically passed to device code
    - Typically ot dereferenced in host
- Host pointers point to CPU memory
    - Typically not passed to device code
    - Typically not dereferenced in device

Running code in parallel:

add<<<1, 1>>>(a, b, c); // launch 1 thread in 1 block

add<<<1024, 1>>>(a, b, c); // launch 1024 threads in 1 block

add<<<1, 2048>>>(a, b, c); // launch 1 thread in 2048 blocks

add<<<1024, 1024>>>(a, b, c); // launch 1024 threads in 1024 blocks

Each parallel invocation of add() is referred to as a block

Set of blocks is referred to as a grid

Each invocation can refer to its block index with blockIdx.x

Each thread can refer to its thread index with threadIdx.x

vector addition on device:

__global__ void add(float *a, float *b, float *c) {
    c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
}

### CUDA Threads

A block can be split into parallel threads

vector addition parallelized over threads:

__global__ void add(float *a, float *b, float *c) {
    c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
}

add<<<1, 1024>>>(a, b, c);
add<<<blocks, threads>>>(a, b, c);

Indexing arrays with blocks and threads:

- With M threads/block a unique index for each thread is given by:
int index = threadIdx.x + blockIdx.x * blockDim.x;
where blockDim.x is the number of threads per block (M)

Handling arbitrary vector sizes

Avoid accessing beyond arrays when array size doesn't equal total thread-block count

__global__ void add(float *a, float *b, float *c, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n) {
        c[index] = a[index] + b[index];
    }
}

add<<<(N + M - 1) / M, M>>>(d_a, d_b, d_c, N);

Why bother with threads?
Unlike parallel blocks, threads have mechanisms to:
- communicate
- synchronize

## CUDA Shared Memory

1D stencil example:

Consider applying a 1D stencil to a 1D array of elements
- each output element is the sum of the input elements within a radius r

If radius is 3, then each output element is the sum of 7 input elements

For example, with radius r=3:

Input array indices: 0  1  2  3  4  5  6  7  8
Input array values: [2, 1, 4, 3, 7, 2, 8, 5, 9]
                     |--------↓--------|
                            3 (center)
                            
Output[3] = 2 + 1 + 4 + 3 + 7 + 2 + 8 = 27

When calculating output[3], we sum elements from index (3-3)=0 to (3+3)=6:
- Input[0] = 2
- Input[1] = 1
- Input[2] = 4
- Input[3] = 3 (center)
- Input[4] = 7
- Input[5] = 2
- Input[6] = 8

### Sharing data between threads

- Within a block threads share data via "shared memory"
- extremely fast on-chip memory which is user managed (different than a cache)
- declared in kernel with __shared__ keyword, and is allocated per block
- data is not visible to threads in other blocks
- bandwidth and latency are higher & lower, respectively, than global memory

### 1D Stencil Implementation

- Cache data in shared memory
    - read (blockDim.x + 2*radius) input elements from global to shared memory
    - compute blockDim.x output elements
    - write blockDim.x output elements from shared to global memory

- Each block needs a "halo" of radius elements at each boundary

__global__ void stencil_1d(float *in, float *out, int N) {
    __shared__ float temp[BLOCK_SIZE + 2*RADIUS];
    int gindex = threadIdx.x + blockIdx.x * blockDim.x;
    int lindex = threadIdx.x + RADIUS;

    // read input elements into shared memory
    temp[lindex] = in[gindex];
    if (threadIdx.x < RADIUS) {
        temp[lindex - RADIUS] = in[gindex - RADIUS];
        temp[lindex + BLOCK_SIZE] = in[gindex + BLOCK_SIZE];
    }

    // wait for all threads to finish reading
    __syncthreads();
    // apply the stencil
    float result = 0;
    for (int offset = -RADIUS; offset <= RADIUS; offset++) {
        result += temp[lindex + offset];
    }
    // store the result
    out[gindex] = result; 
}

### Review

- use __shared__ to declare a variable/array in shared memory
    - data is shared within a block, amongst threads
    - not visible to threads in other blocks
- use __syncthreads() to synchronize threads within a block (barrier op)


## CUDA Optimization Part 1

### Warps

Warp: a collection of 32 threads that execute instructions simultaneously

A thread block can be composed of multiple warps

A wapr is executed physically in parallel (SIMD) on a multiprocessor

### Launch Configurations

- gridDim.x = number of blocks
- blockDim.x = number of threads per block
- gridDim.y = number of blocks
- blockDim.y = number of threads per block

Key to understanding:
- Instructions are issued warp-wide
- Instructions are issued in order
- A threads stalls when one of the operands isn't ready:
    - Memory read by itself doesn't stall execution
- Latency is hidden by switching threads
    - GMEM latency > 100 cycles
    - Arithmetic latency < 100 cycles
- Need enough threads to hide latency

Latency: difference in clock cycles between when code is executed and when it is completed

### GPU Latency Hiding

- CUDA C source code
int idx = threadIdx.x + blockDim.x * blockIdx.x;
c[idx] = a[idx] * b[idx];

- Machine code
L0: LD R0, a[idx]
L1: LD R1, b[idx]
L2: MUL R2, R0, R1
L3: ST R2, c[idx]

- Latency: 3 cycles
- Issue rate: 1 per cycle

### Launch Configurations: Summary
- Need enough threads to hide latency (keep GPU busy)
    - Typically 512+ threads per SM (aim for 2048)
        - More if processing for fp32 element per thread

- Thread block configuration
    - Threads per block should be multiple of warp size (32)
    - SM can concurrently execute at least 16 threads per block
        - Generaly use 128-256 threads per block

## CUDA Optimization Part 2

### Memory Hierarchy Review

- Local storage
    - Each thread has own local storage
    - Typically registers (managed by compiler)

- Shared memory
    - Program configurable: typically up to 48KB per SM
    - Shared memory is accessible by threads in the same thread block
    - very low latency/high throughput

- L2 cache
    - All access to global memory go through L2 cache including copies to/from host

- Global memory
    - Accessible by all threads and host


### GMEM Operations
- Loads
    - caching
        - default mode
        - Attempts to hit in L1 then L2 then global memory
        - Load granularity is 128 byte line
- Stores
    - Invalidate L1, write-back for L2
    - Write-allocate
        - If data is not present in L2, data is loaded from global to L2
        - Data is loaded into cache line (128 bytes)

### Load Operation

- Memory operation are issued per warp (32 threads)
    - Just like other instructions

- Operation
    - Threads in a warp provide memory addresses
    - Determine which lines/segments are needed
    - Request the needed lines/segments

### Caching Load

- Warp requests 32 aligned, consecutive 4 byte words
- Addresses fall within 1 cache line (128 bytes)
    - warp needs 128 bytes
    - 128 bytes move across the buss on a miss
    - Bus utilization is 128/128 = 100%
    - int c = a[idx]

Warp requests 32 misaligned, consecutive 4 byte words
- Addressed fall within 2 cache lines
    - warp needs 128 bytes
    - 256 bytes move across the buss on a miss
    - Bus utilization is 128/256 = 50%



### Shared Memory

- Uses
    - Inter-thread communication within a block
    - Cache data to reduce redundant GMEM accesses
    - Use it to improve global memory access patterns

- Organization
    - 32 banks, 4-byte wide banks
    - Successive 4-byte words belong to different banks

- Performance
    - Typically: 4 bytes per bank 1 or 2 clocks per multiprocessor
    - Shared accesses are issued per 32 threads (warp)
    - Serialization: if N threads of 32 access different 4 byte words in teh same bank, N accesses are executed serially
    - Multicast: N threads access the same 4 byte word in the same bank 
        - Could be different bytes within same word


## Atomics, Reductions, and Warp Shuffle

Motivating example: Sum reduction

- Sum reduction
    - Sum all elements in an array
    - Result is a single value

const int size = 100000
float a[size] = {...};
float sum = 0;
for (int i = 0; i < size; i++) {
    sum += a[i];
}

This doesn't work because of race conditions.

Actual code the GPU executes:
LD R2, a[i]
LD R1, sum
ADD R3, R1, R2
ST sum, R3

This is a race condition because multiple threads are trying to update the same memory location.


### Atomics to the rescue

atomicAdd(&sum, a[i]);

LD R2, a[i]    // (thread independent)
LD R1, sum     // (read)
ADD R3, R1, R2 // (modify)
ST R3, sum     // (write)

read, modify, write sequence becomes one indivisible operation/instruction

Facilitated by special hardware in the L2 cache
May have performance implications

### Other atomics

- atomicMax/Min - choose the max or min of two values
- atomicAdd/Sub - add to or subtract from a value
- atomicInc/Dec - increment or decrement a value and account for rollover/underflow
- atomic Exch/CAS - swap values, or conditionally swap values
- atomic And/Or/Xor - bitwise operations
- atomics have different datatypes they can work on (e.g. int, float, double, etc.)

### Atomic tips and tricks

- Could be used to determine next work item, queue slot, etc.
- int my_position = atomicAdd(order, 1);
- Most atomics return a value that is the "old" value that was in the location receiving the atomic operation update

Reserve space in a buffer

- Each thread in my kernel may produce a variable amoutn of data. How to collect all of this in one buffer, in parallel?

int my_dsize = var;
float local_buffer[my_dsize] = {...};
int my_offset = atomicAdd(buffer_idx, my_dsize);
// buffer_ptr+my_offset now points to the first reserved location, of length my_dsize
memcpy(buffer_ptr+my_offset, local_buffer, my_dsize*sizeof(float));


### The classical parallel reduction
Atomics dont run at full memory bandwidth...

- We would like a reduction method that is not limited by atomics throughput
- We would liek to effectively use all threads, as much as possible
- Parallel reduction is a common and important data parallel primitive
- Naive implementations will often run into bottlenecks
- Basic methodology is a tree-based reduction

#### Problem: Global Synchronization

- If we could synchronize all threads in the block, could we easily reduce very large arrays, right?
    - Global sync after each block produces its result
    - Once all blocks reach sync, continue recursively






    
















