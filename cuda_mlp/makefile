# Makefile

# compile
NVCC = nvcc

# compiler flags
CFLAGS = -O2

# executables
TARGETS = main hello_cuda

# Source files for main executable
SRCS_main = main.cu cifar_loader.cu mlp.cu cuda_utils.cu

# source files for hello_cuda executable
SRCS_hello_cuda = hello_cuda.cu

# default targets
all: $(TARGETS)

# rule to build 'main' executable
main: $(SRCS_main)
	$(NVCC) $(CFLAGS) -o main $(SRCS_main)

# rule to build 'hello_cuda' exec
hello_cuda: $(SRCS_hello_cuda)
	$(NVCC) $(CFLAGS) -o hello_cuda $(SRCS_hello_cuda)

# clean up build files
clean:
	rm -f $(TARGETS) *.o