NVCC = nvcc
CUDA_FLAGS = -std=c++11 -O3 -arch=sm_70
INCLUDES = -I/usr/local/cuda/include
LIBS = -lcublas -lcurand -lcudnn

SRCDIR = .
CDIR = ../c
SOURCES = $(wildcard $(SRCDIR)/*.cu)
OBJECTS = $(SOURCES:.cu=.o)
TARGET = cuda_gpt2

# Default target
all: $(TARGET)

# Build target
$(TARGET): $(OBJECTS)
	$(NVCC) $(CUDA_FLAGS) $(OBJECTS) -o $@ $(LIBS)

# Compile .cu files
%.o: %.cu
	$(NVCC) $(CUDA_FLAGS) $(INCLUDES) -c $< -o $@

# Clean
clean:
	rm -f $(OBJECTS) $(TARGET)

# Install dependencies (Ubuntu/Debian)
install-deps:
	sudo apt-get update
	sudo apt-get install nvidia-cuda-toolkit libcublas-dev libcurand-dev libcudnn8-dev

# Run
run: $(TARGET)
	./$(TARGET)

.PHONY: all clean install-deps run