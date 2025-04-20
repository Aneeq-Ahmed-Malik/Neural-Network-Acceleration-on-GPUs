# Neural Network Acceleration on GPUs: MNIST Classification

## Project Overview

This project, developed for CS-4110: High Performance Computing with GPU, optimizes a 784-128-10 neural network for MNIST digit classification using CUDA.

The MNIST dataset (60,000 training and 10,000 test images, 28x28 pixels) is processed with a batch size of 64 over 3 epochs. Four versions demonstrate GPU acceleration:

- **Version 1 (V1)**: Sequential CPU implementation
- **Version 2 (V2)**: Naive GPU implementation with CUDA kernels
- **Version 3 (V3)**: Optimized GPU implementation with progressive improvements:
  - **V3.1**: Shared memory and CUDA streams
  - **V3.2**: Batch processing
  - **V3.3**: Float precision and batch evaluation kernel
- **Version 4 (V4)**: Tensor Core implementation with cuBLAS FP32 matrix operations (6.05s, 3.81x speedup)

Test accuracy ranges from 96.80% to 96.85%, showcasing significant performance improvements with minimal accuracy loss.

**Authors:** Aneeq Ahmed Malik, Abdullah Mehmood, Hamza Saleem  
**Date:** April 20, 2025

## Prerequisites

To compile and run this project, ensure the following are installed:

- **CUDA Toolkit**: Version 11.x or later (includes nvcc compiler and cuBLAS library)
- **MNIST Dataset**: Download the dataset files from [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/):
  - train-images-idx3-ubyte
  - train-labels-idx1-ubyte
  - t10k-images-idx3-ubyte
  - t10k-labels-idx1-ubyte
- **Hardware**: NVIDIA GPU with CUDA support (e.g., compute capability 5.0 or higher)
- **Operating System**: Linux (recommended) or Windows with WSL2 for CUDA compatibility

**Optional tools for profiling:**
- NVIDIA Nsight Systems or Visual Profiler for GPU performance analysis

## Repository Structure

```
├── src/
│   ├── V1/
│   │   ├── nn.cu               # Version 1: CPU implementation
│   │   ├── Makefile            # Makefile for V1
│   ├── V2/
│   │   ├── nn.cu               # Version 2: Naive GPU implementation
│   │   ├── Makefile            # Makefile for V2
│   ├── V3/
│   │   ├── V3.1/
│   │   │   ├── nn.cu           # V3.1: Shared memory & Cuda Streams
│   │   │   ├── Makefile        # Makefile for V3.1
│   │   ├── V3.2/
│   │   │   ├── nn.cu           # V3.2: Batch Processing
│   │   │   ├── Makefile        # Makefile for V3.2
│   │   ├── V3.3/
│   │   │   ├── nn.cu           # V3.3: Float precision and batch eval kernel
│   │   │   ├── Makefile        # Makefile for V3.3
│   ├── V4/
│   │   ├── nn.cu               # Version 4: cuBLAS implementation
│   │   ├── Makefile            # Makefile for V4
├── data/
│   └── datafiles               # MNIST dataset files
├── HPC_Report.pdf              # Project report
├── HPC_Slides.pdf              # Presentation slides
└── README.md                   # This file
```

## Installation

### Clone the Repository:
```bash
git clone https://github.com/Aneeq-Ahmed-Malik/Neural-Network-Acceleration-on-GPUs
```

### Install Dependencies:
1. Ensure the CUDA Toolkit is installed. Verify with:
   ```bash
   nvcc --version
   ```
2. Install GCC if not already present (e.g., on Ubuntu):
   ```bash
   sudo apt update
   sudo apt install build-essential
   ```

## Compilation

Each version (and V3 sub-version) has its own Makefile in its respective directory (`src/V1/`, `src/V2/`, `src/V3/V3.1/`, etc.). The Makefile compiles the `nn.cu` file, which includes dataset loading and neural network logic.

To compile a specific version, navigate to its directory and run `make`. For example:

```bash
cd src/V1/
make
```

This will generate executables in the respective directories.

To clean build artifacts for a specific version, run:
```bash
make clean
```

**Note:** Ensure `nvcc` is in your PATH and the cuBLAS library is accessible (required for V4, linked with `-lcublas`). For V1, `nn.cu` is compiled with `nvcc` but uses CPU logic. Check CUDA include/library paths in the Makefile if compilation fails.

## Usage

Each executable will:
- Load the MNIST dataset
- Train the neural network for 3 epochs
- Evaluate on the test set
- Output training/evaluation times and test accuracy

## Troubleshooting

- **CUDA Errors**: Ensure GPU drivers and CUDA Toolkit are up to date. Check `nvidia-smi` for GPU status.
- **Dataset Issues**: Verify MNIST files are in `data/` and not corrupted.
- **Compilation Errors**: Confirm `nvcc` compatibility. Check CUDA include/library paths in the Makefile. For V4, ensure cuBLAS is linked.
- **Performance Discrepancies**: Results may vary by GPU model (e.g., RTX vs. Tesla). Use Nsight for profiling.
