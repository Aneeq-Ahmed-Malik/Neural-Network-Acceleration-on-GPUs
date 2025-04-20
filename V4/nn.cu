#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda.h>
#include <cublas_v2.h>

#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.01
#define EPOCHS 3
#define BATCH_SIZE 1
#define NUM_CLASSES 10  // Digits 0-9

#define H2D cudaMemcpyHostToDevice
#define D2H cudaMemcpyDeviceToHost

// Timer function
double get_time(clock_t start) {
    return (double)(clock() - start) / CLOCKS_PER_SEC;
}

template <typename T>
T** allocateMatrix(int rows, int cols) {
    T** mat = (T**)malloc(rows * sizeof(T*));
    for (int i = 0; i < rows; i++) {
        mat[i] = (T*)malloc(cols * sizeof(T));
    }
    return mat;
}

void freeMatrix(double** mat, int rows) {
    for (int i = 0; i < rows; i++) {
        free(mat[i]);
    }
    free(mat);
}

template <typename T>
struct NeuralNetwork{
    T* W1;   // 2D matrix: HIDDEN_SIZE x INPUT_SIZE
    T* W2;   // 2D matrix: OUTPUT_SIZE x HIDDEN_SIZE
    T* b1;   // Bias vector: HIDDEN_SIZE
    T* b2;   // Bias vector: OUTPUT_SIZE
} ;

struct NeuralNetworkCPU{
    double** W1;   // 2D matrix: HIDDEN_SIZE x INPUT_SIZE
    double** W2;   // 2D matrix: OUTPUT_SIZE x HIDDEN_SIZE
    double* b1;   // Bias vector: HIDDEN_SIZE
    double* b2;   // Bias vector: OUTPUT_SIZE
} ;

template <typename T>
NeuralNetwork<T>* createNetwork() {
    // --- Host (CPU) allocations ---
    T* h_W1 = (T*)malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(T));
    T* h_W2 = (T*)malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(T));
    T* h_b1 = (T*)calloc(HIDDEN_SIZE, sizeof(T)); // Zero-init biases
    T* h_b2 = (T*)calloc(OUTPUT_SIZE, sizeof(T));

    // Random weight initialization
    srand(time(NULL));
    for (int i = 0; i < HIDDEN_SIZE * INPUT_SIZE; i++) {
        h_W1[i] = ((T)rand() / RAND_MAX) * 0.01f;
    }
    for (int i = 0; i < OUTPUT_SIZE * HIDDEN_SIZE; i++) {
        h_W2[i] = ((T)rand() / RAND_MAX) * 0.01f;
    }

    // --- GPU (Device) allocations ---
    NeuralNetwork<T> *d_net;
    cudaMalloc((void**)&d_net, sizeof(NeuralNetwork<T>));

    T *d_W1, *d_W2, *d_b1, *d_b2;
    cudaMalloc((void**)&d_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(T));
    cudaMalloc((void**)&d_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(T));
    cudaMalloc((void**)&d_b1, HIDDEN_SIZE * sizeof(T));
    cudaMalloc((void**)&d_b2, OUTPUT_SIZE * sizeof(T));

    cudaMemcpy(d_W1, h_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W2, h_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b1, h_b1, HIDDEN_SIZE * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b2, h_b2, OUTPUT_SIZE * sizeof(T), cudaMemcpyHostToDevice);

    NeuralNetwork<T> h_net;
    h_net.W1 = d_W1;
    h_net.W2 = d_W2;
    h_net.b1 = d_b1;
    h_net.b2 = d_b2;

    cudaMemcpy(d_net, &h_net, sizeof(NeuralNetwork<T>), cudaMemcpyHostToDevice);

    free(h_W1);
    free(h_W2);
    free(h_b1);
    free(h_b2);

    return d_net;
}

template <typename T>
__global__ void apply_bias_relu(T* hidden, NeuralNetwork<T>* net) {
    int idx = threadIdx.x;
    int batch_idx = blockIdx.y; // Batch index

    if (idx < HIDDEN_SIZE && batch_idx < BATCH_SIZE) {
        T sum = net->b1[idx] + hidden[batch_idx * HIDDEN_SIZE + idx];
        if (sum < 0.0)
            sum = 0.0;
        hidden[batch_idx * HIDDEN_SIZE + idx] = sum;
    }
}

template <typename T>
__global__ void compute_hidden_layer(NeuralNetwork<T>* net, const T* input, T* hidden) {
    int i = threadIdx.x + blockIdx.x * blockDim.x; // Hidden neuron index
    int batch_idx = blockIdx.y; // Batch index

    __shared__ T s_input[INPUT_SIZE]; // Shared memory for input (INPUT_SIZE = 784)
    if (threadIdx.x < INPUT_SIZE) {
        s_input[threadIdx.x] = input[batch_idx * INPUT_SIZE + threadIdx.x];
    }
    __syncthreads();

    if (i < HIDDEN_SIZE && batch_idx < gridDim.y) {
        T sum = net->b1[i];
        for (int j = 0; j < INPUT_SIZE; j++) {
            sum += net->W1[i * INPUT_SIZE + j] * s_input[j];
        }
        hidden[batch_idx * HIDDEN_SIZE + i] = (sum > 0.0) ? sum : 0.0;
    }
}

template <typename T>
__global__ void compute_output_layer(NeuralNetwork<T>* net, const T* hidden, T* output) {
    int i = threadIdx.x + blockIdx.x * blockDim.x; // Output neuron index
    int batch_idx = blockIdx.y; // Batch index

    __shared__ T s_hidden[HIDDEN_SIZE]; // Shared memory for hidden (HIDDEN_SIZE = 128)
    if (threadIdx.x < HIDDEN_SIZE) {
        s_hidden[threadIdx.x] = hidden[batch_idx * HIDDEN_SIZE + threadIdx.x];
    }
    __syncthreads();

    if (i < OUTPUT_SIZE && batch_idx < gridDim.y) {
        T sum = net->b2[i];
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            sum += net->W2[i * HIDDEN_SIZE + j] * s_hidden[j];
        }
        output[batch_idx * OUTPUT_SIZE + i] = exp(sum); // Store exp for softmax
    }
}

template <typename T>
__global__ void normalize_softmax(T* output) {
    __shared__ T s_output[OUTPUT_SIZE];
    __shared__ T sum;

    int tid = threadIdx.x;
    int batch_idx = blockIdx.y; // Batch index

    if (tid < OUTPUT_SIZE)
        s_output[tid] = output[batch_idx * OUTPUT_SIZE + tid];  
    __syncthreads();


    if (batch_idx < gridDim.y) { // One thread per sample computes sum
        if (tid == 0){
            sum = 0.0;
            for (int k = 0; k < OUTPUT_SIZE; k++)
                sum += s_output[k];
        }
        __syncthreads();
        // Normalize all outputs for this sample
        output[batch_idx * OUTPUT_SIZE + tid] /= sum;   
    }
}




template <typename T>
__global__ void compute_d_output(const T* output, const T* target, T* d_output) {
    int i = threadIdx.x + blockIdx.x * blockDim.x; // Output neuron index
    int b = blockIdx.y; // Batch index

    if (i < OUTPUT_SIZE && b < BATCH_SIZE) {
        int idx = b * OUTPUT_SIZE + i;
        d_output[idx] = output[idx] - target[idx];
    }
}

template <typename T>
__global__ void compute_d_hidden(NeuralNetwork<T>* net, const T* d_output, const T* hidden, T* d_hidden) {
    int i = threadIdx.x + blockIdx.x * blockDim.x; // Hidden neuron index
    int b = blockIdx.y; // Batch index

    __shared__ T s_output[OUTPUT_SIZE]; // Shared memory for d_output (OUTPUT_SIZE = 10)
    if (threadIdx.x < OUTPUT_SIZE)
        s_output[threadIdx.x] = d_output[b * OUTPUT_SIZE + threadIdx.x];
    __syncthreads();

    if (i < HIDDEN_SIZE && b < BATCH_SIZE) {
        T grad = 0.0;
        for (int j = 0; j < OUTPUT_SIZE; j++) 
            grad += net->W2[j * HIDDEN_SIZE + i] * s_output[j];

        d_hidden[b * HIDDEN_SIZE + i] = (hidden[b * HIDDEN_SIZE + i] > 0.0) ? grad : 0.0;
    }
}

template <typename T>
__global__ void update_output_weights(NeuralNetwork<T>* net, const T* d_output, const T* hidden) {
    int i = blockIdx.x; // OUTPUT neuron index
    int j = threadIdx.x; // HIDDEN neuron index
    int b = blockIdx.y; // Batch index

    if (i < OUTPUT_SIZE && j < HIDDEN_SIZE && b < BATCH_SIZE) {
        int idx = i * HIDDEN_SIZE + j;
        atomicAdd(&net->W2[idx], -LEARNING_RATE * d_output[b * OUTPUT_SIZE + i] * hidden[b * HIDDEN_SIZE + j] / BATCH_SIZE);
    }

    if (j == 0 && i < OUTPUT_SIZE && b < BATCH_SIZE) 
        atomicAdd(&net->b2[i], -LEARNING_RATE * d_output[b * OUTPUT_SIZE + i] / BATCH_SIZE);
}

template <typename T>
__global__ void update_hidden_weights(NeuralNetwork<T>* net, const T* d_hidden, const T* input) {
    int i = blockIdx.x; // HIDDEN neuron index
    int j = threadIdx.x; // INPUT neuron index
    int b = blockIdx.y; // Batch index

    if (i < HIDDEN_SIZE && j < INPUT_SIZE && b < BATCH_SIZE) {
        int idx = i * INPUT_SIZE + j;
        atomicAdd(&net->W1[idx], -LEARNING_RATE * d_hidden[b * HIDDEN_SIZE + i] * input[b * INPUT_SIZE + j] / BATCH_SIZE);
    }

    if (j == 0 && i < HIDDEN_SIZE && b < BATCH_SIZE) {
        atomicAdd(&net->b1[i], -LEARNING_RATE * d_hidden[b * HIDDEN_SIZE + i] / BATCH_SIZE);
    }
}


template <typename T>
void forward_cuda(NeuralNetwork<T>* net, NeuralNetwork<T>& h_net, T* d_input, T* d_output, T* d_hidden, cudaStream_t stream, cublasHandle_t handle) {

    cublasSetStream(handle, stream);

    const T alpha = 1.0f;
    const T beta = 0.0f; 
    
    cublasSgemm(handle,
        CUBLAS_OP_T, CUBLAS_OP_T,
        HIDDEN_SIZE, BATCH_SIZE, INPUT_SIZE,
        &alpha,
        h_net.W1, INPUT_SIZE, // W1^T [INPUT_SIZE x HIDDEN_SIZE] in column-major
        d_input, BATCH_SIZE,  // input^T [INPUT_SIZE x BATCH_SIZE] in column-major
        &beta,
        d_hidden, HIDDEN_SIZE); // hidden^T [HIDDEN_SIZE x BATCH_SIZE] in column-major

    dim3 block(HIDDEN_SIZE);
    dim3 grid(1, BATCH_SIZE);
    apply_bias_relu<<<grid, block, 0, stream>>>(d_hidden, net);


    dim3 blockOutput(HIDDEN_SIZE); // Threads per block (matches OUTPUT_SIZE)
    dim3 gridOutput(1, BATCH_SIZE);
    compute_output_layer<<<gridOutput, blockOutput, 0, stream>>>(net, d_hidden, d_output);

    // Softmax normalization
    dim3 blockSoftmax(OUTPUT_SIZE); // One thread per sample
    dim3 gridSoftmax(1, BATCH_SIZE);
    normalize_softmax<<<gridSoftmax, blockSoftmax, 0, stream>>>(d_output);
    
}

template <typename T>
void backward_cuda(NeuralNetwork<T>* net,  T* input, T* hidden, T* output, T* target, cudaStream_t stream, cublasHandle_t handle) {
    T *d_output_grad, *d_hidden_grad;
    cudaMalloc(&d_output_grad, BATCH_SIZE * OUTPUT_SIZE * sizeof(T));
    cudaMalloc(&d_hidden_grad, BATCH_SIZE * HIDDEN_SIZE * sizeof(T));
    cublasSetStream(handle, stream);
    // Compute output gradients
    dim3 blockDOutput(OUTPUT_SIZE);
    dim3 gridDOutput(1, BATCH_SIZE);
    compute_d_output<<<gridDOutput, blockDOutput, 0, stream>>>(output, target, d_output_grad);

    dim3 blockDHidden(HIDDEN_SIZE);
    dim3 gridDHidden(1, BATCH_SIZE);
    compute_d_hidden<<<gridDHidden, blockDHidden, 0, stream>>>(net, d_output_grad, hidden, d_hidden_grad);

    // Update output weights
    dim3 blockOutputWeights(HIDDEN_SIZE); // 128 threads
    dim3 gridOutputWeights(OUTPUT_SIZE, BATCH_SIZE); // 10 blocks per sample
    update_output_weights<<<gridOutputWeights, blockOutputWeights, 0, stream>>>(net, d_output_grad, hidden);

    // Update hidden weights

    dim3 blockHiddenWeights(INPUT_SIZE); // 784 threads
    dim3 gridHiddenWeights(HIDDEN_SIZE, BATCH_SIZE); // 128 blocks per sample
    update_hidden_weights<<<gridHiddenWeights, blockHiddenWeights, 0, stream>>>(net, d_hidden_grad, input);

    cudaFree(d_output_grad);
    cudaFree(d_hidden_grad);
}

template <typename T>
void train(NeuralNetwork<T>* net, T** images, T** labels, int numImages) {
    
    clock_t total_start = clock();

    cudaStream_t streams[2];
    cudaStreamCreate(&streams[0]);
    cudaStreamCreate(&streams[1]);

    T *d_input[2], *d_hidden[2], *d_output[2], *d_label[2];
    for (int s = 0; s < 2; s++) {
        cudaMalloc(&d_input[s], BATCH_SIZE * INPUT_SIZE * sizeof(T));
        cudaMalloc(&d_hidden[s], BATCH_SIZE * HIDDEN_SIZE * sizeof(T));
        cudaMalloc(&d_output[s], BATCH_SIZE * OUTPUT_SIZE * sizeof(T));
        cudaMalloc(&d_label[s], BATCH_SIZE * NUM_CLASSES * sizeof(T));
    }

    T *pinned_output;
    cudaHostAlloc(&pinned_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(T), cudaHostAllocDefault);

    NeuralNetwork<T> h_net;
    cudaMemcpy(&h_net, net, sizeof(NeuralNetwork<T>), D2H);

    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);

    int first_batch_size = min(BATCH_SIZE, numImages);
    for (int b = 0; b < first_batch_size; b++) {
        cudaMemcpyAsync(d_input[0] + b * INPUT_SIZE, images[b], INPUT_SIZE * sizeof(T), H2D, streams[0]);
        cudaMemcpyAsync(d_label[0] + b * NUM_CLASSES, labels[b], NUM_CLASSES * sizeof(T), H2D, streams[0]);
    }

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        clock_t epoch_start = clock();
        T loss = 0.0;
        int correct = 0;
        forward_cuda(net, h_net, d_input[0], d_output[0], d_hidden[0], streams[0], handle);

        for (int i = 0; i < numImages; i += BATCH_SIZE) {
            int current_stream = (i / BATCH_SIZE) % 2;
            int next_stream = ((i / BATCH_SIZE) + 1) % 2;
            int batch_size = min(BATCH_SIZE, numImages - i);

            backward_cuda(net, d_input[current_stream], d_hidden[current_stream], d_output[current_stream], d_label[current_stream], streams[current_stream], handle);

            cudaMemcpyAsync(pinned_output, d_output[current_stream], batch_size * OUTPUT_SIZE * sizeof(T), D2H, streams[current_stream]);

            if (i + BATCH_SIZE < numImages) {
                int next_batch_size = min(BATCH_SIZE, numImages - (i + BATCH_SIZE));
                for (int b = 0; b < next_batch_size; b++) {
                    cudaMemcpyAsync(d_input[next_stream] + b * INPUT_SIZE, images[i + BATCH_SIZE + b], INPUT_SIZE * sizeof(T), H2D, streams[next_stream]);
                    cudaMemcpyAsync(d_label[next_stream] + b * NUM_CLASSES, labels[i + BATCH_SIZE + b], NUM_CLASSES * sizeof(T), H2D, streams[next_stream]);
                }
                forward_cuda(net, h_net,  d_input[next_stream], d_output[next_stream], d_hidden[next_stream], streams[next_stream], handle);
            }

            cudaStreamSynchronize(streams[current_stream]);

            // Compute metrics
            for (int b = 0; b < batch_size; b++) {
                for (int k = 0; k < OUTPUT_SIZE; k++)
                    loss -= labels[i + b][k] * log(pinned_output[b * OUTPUT_SIZE + k]);
                int pred = 0, actual = 0;
                for (int j = 0; j < OUTPUT_SIZE; j++) {
                    if (pinned_output[b * OUTPUT_SIZE + j] > pinned_output[b * OUTPUT_SIZE + pred]) pred = j;
                    if (labels[i + b][j] > labels[i + b][actual]) actual = j;
                }
                if (pred == actual) correct++;
            }
        }

        printf("Epoch %d - Loss: %.4f - Accuracy: %.2f%% - Time: %.3fs\n",
               epoch + 1, loss / numImages, (correct / (T)numImages) * 100,
               (T)(clock() - epoch_start)/CLOCKS_PER_SEC);
    }

    // Cleanup
    cudaStreamDestroy(streams[0]);
    cudaStreamDestroy(streams[1]);
    cudaFreeHost(pinned_output);
    for (int s = 0; s < 2; s++) {
        cudaFree(d_input[s]);
        cudaFree(d_hidden[s]);
        cudaFree(d_output[s]);
        cudaFree(d_label[s]);
    }

    printf("Total training time: %.3fs\n", (T)(clock() - total_start)/CLOCKS_PER_SEC);
}


template <typename T>
void forward_cuda_eval(NeuralNetwork<T>* net, T* d_input, T* d_output, T* d_hidden, int numImages) {
    // Hidden layer
    dim3 blockHidden(INPUT_SIZE); // Threads per block (matches HIDDEN_SIZE)
    dim3 gridHidden(1, numImages); // One block per sample
    compute_hidden_layer<<<gridHidden, blockHidden>>>(net, d_input, d_hidden);

    // Output layer
    dim3 blockOutput(HIDDEN_SIZE); // Threads per block (matches OUTPUT_SIZE)
    dim3 gridOutput(1, numImages);
    compute_output_layer<<<gridOutput, blockOutput>>>(net, d_hidden, d_output);

    // Softmax normalization
    dim3 blockSoftmax(OUTPUT_SIZE); // One thread per sample
    dim3 gridSoftmax(1, numImages);
    normalize_softmax<<<gridSoftmax, blockSoftmax>>>(d_output);
}


template <typename T>
void evaluate(NeuralNetwork<T>* net, T** images, T** labels, int numImages) {
    int correct = 0;

    // Host-side flat output buffer
    T* output = (T*)malloc(sizeof(T) * OUTPUT_SIZE * numImages);

    T *d_input, *d_hidden, *d_output;
    cudaMalloc((void**)&d_input, sizeof(T) * INPUT_SIZE * numImages);
    cudaMalloc((void**)&d_hidden, sizeof(T) * HIDDEN_SIZE * numImages);
    cudaMalloc((void**)&d_output, sizeof(T) * OUTPUT_SIZE * numImages);

    // Copy all images in one go
    for (int i = 0; i < numImages; i++) {
        cudaMemcpyAsync(d_input + i * INPUT_SIZE, images[i], sizeof(T) * INPUT_SIZE, H2D);
    }

    forward_cuda_eval(net, d_input, d_output, d_hidden, numImages);

    // Copy full output back to host
    cudaMemcpy(output, d_output, sizeof(T) * OUTPUT_SIZE * numImages, D2H);

    // Accuracy calculation
    for (int i = 0; i < numImages; i++) {
        int pred = 0, actual = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            T val = output[i * OUTPUT_SIZE + j];
            if (val > output[i * OUTPUT_SIZE + pred]) pred = j;
            if (labels[i][j] > labels[i][actual]) actual = j;
        }
        if (pred == actual) correct++;
    }

    printf("Test Accuracy: %.2f%%\n", (correct / (double)numImages) * 100.0);

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_hidden);
    cudaFree(d_output);
    free(output);

}



void freeNetwork(NeuralNetworkCPU* net) {
    freeMatrix(net->W1, HIDDEN_SIZE);
    freeMatrix(net->W2, OUTPUT_SIZE);
    free(net->b1);
    free(net->b2);
    free(net);
}

template <typename T>
void freeNetwork(NeuralNetwork<T>* net) {
    cudaFree(net->W1);
    cudaFree(net->W2);
    cudaFree(net->b1);
    cudaFree(net->b2);
    cudaFree(net);
}




template <typename T>
T** loadMNISTImages(const char* filename, int numImages);
template <typename T>
T** loadMNISTLabels(const char* filename, int numLabels) ;

NeuralNetworkCPU* createNetworkCPU();
void forward(NeuralNetworkCPU* net, double* input, double* hidden, double* output);
void backward(NeuralNetworkCPU* net, double* input, double* hidden, double* output, double* target);
void trainCPU(NeuralNetworkCPU* net, double** images, double** labels, int numImages);
void evaluateCPU(NeuralNetworkCPU* net, double** images, double** labels, int numImages);
void relu(double* x, int size);
void softmax(double* x, int size) ;

// Main function
int main() {
    printf("MNIST Neural Network\n\n");

    float** train_images = loadMNISTImages<float>("../data/train-images.idx3-ubyte", 60000);
    float** train_labels = loadMNISTLabels<float>("../data/train-labels.idx1-ubyte", 60000);
    float** test_images = loadMNISTImages<float>("../data/t10k-images.idx3-ubyte", 10000);
    float** test_labels = loadMNISTLabels<float>("../data/t10k-labels.idx1-ubyte", 10000);

    double** train_images_cpu = loadMNISTImages<double>("../data/train-images.idx3-ubyte", 60000);
    double** train_labels_cpu = loadMNISTLabels<double>("../data/train-labels.idx1-ubyte", 60000);
    double** test_images_cpu = loadMNISTImages<double>("../data/t10k-images.idx3-ubyte", 10000);
    double** test_labels_cpu = loadMNISTLabels<double>("../data/t10k-labels.idx1-ubyte", 10000);


    // Timing for GPU training
    printf("Batch size: %d\n", BATCH_SIZE);
    printf("\nStarting GPU training...\n");
    clock_t total_gpu_train = clock();
    NeuralNetwork<float>* net = createNetwork<float>();
    train<float>(net, train_images, train_labels, 60000);
    double gpu_train_time = get_time(total_gpu_train);
    printf("\nGPU Training time: %.3fs\n", gpu_train_time);

    // Timing for GPU evaluation
    clock_t total_gpu_eval = clock();
    evaluate<float>(net, test_images, test_labels, 10000);
    double gpu_eval_time = get_time(total_gpu_eval);
    printf("GPU Evaluation time: %.3fs\n", gpu_eval_time);

    // Overall GPU time (Training + Evaluation)
    double gpu_total_time = gpu_train_time + gpu_eval_time;
    printf("\nTotal GPU time (Training + Evaluation): %.3fs\n\n", gpu_total_time);


    // Timing for CPU training
    printf("\nStarting CPU training...\n");
    clock_t total_cpu_train = clock();
    NeuralNetworkCPU* netCPU = createNetworkCPU();
    trainCPU(netCPU, train_images_cpu, train_labels_cpu, 60000);
    double cpu_train_time = get_time(total_cpu_train);
    printf("\nCPU Training time: %.3fs\n", cpu_train_time);

    // Timing for CPU evaluation
    clock_t total_cpu_eval = clock();
    evaluateCPU(netCPU, test_images_cpu, test_labels_cpu, 10000);
    double cpu_eval_time = get_time(total_cpu_eval);
    printf("CPU Evaluation time: %.3fs\n", cpu_eval_time);

    // Overall CPU time (Training + Evaluation)
    double cpu_total_time = cpu_train_time + cpu_eval_time;
    printf("\nTotal CPU time (Training + Evaluation): %.3fs\n\n", cpu_total_time);

    // Speedup Calculations
    double train_speedup = cpu_train_time / gpu_train_time;
    double eval_speedup = cpu_eval_time / gpu_eval_time;
    double total_speedup = cpu_total_time / gpu_total_time;

    printf("Speedup (Training): %.3f\n", train_speedup);
    printf("Speedup (Evaluation): %.3f\n", eval_speedup);
    printf("Overall Speedup (CPU / GPU): %.3f\n\n", total_speedup);


    freeNetwork(netCPU);



    // freeNetwork(net);
    return 0;
}

template <typename T>
T** loadMNISTImages(const char* filename, int numImages) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening %s\n", filename);
        exit(1);
    }
    fseek(file, 16, SEEK_SET);
    T** images = allocateMatrix<T>(numImages, INPUT_SIZE);
    for (int i = 0; i < numImages; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            unsigned char pixel;

            // fread(&pixel, sizeof(unsigned char), 1, file);
            if (fread(&pixel, sizeof(unsigned char), 1, file) != 1) {
                fprintf(stderr, "Error: Failed to read pixel\n");
                fclose(file);
                exit(EXIT_FAILURE);
            }

            images[i][j] = pixel / 255.0;
        }
    }
    fclose(file);
    return images;
}

template <typename T>
T** loadMNISTLabels(const char* filename, int numLabels) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening %s\n", filename);
        exit(1);
    }
    fseek(file, 8, SEEK_SET);
    T** labels = allocateMatrix<T>(numLabels, OUTPUT_SIZE);
    for (int i = 0; i < numLabels; i++) {
        unsigned char label;
        // fread(&label, sizeof(unsigned char), 1, file);
        if (fread(&label, sizeof(unsigned char), 1, file) != 1) {
            fprintf(stderr, "Error: Failed to read label\n");
            fclose(file);
            exit(EXIT_FAILURE);
        }

        for (int j = 0; j < OUTPUT_SIZE; j++) {
            labels[i][j] = (j == label) ? 1.0 : 0.0;
        }
    }
    fclose(file);
    return labels;
}


NeuralNetworkCPU* createNetworkCPU() {
    NeuralNetworkCPU* net = (NeuralNetworkCPU*)malloc(sizeof(NeuralNetworkCPU));
    net->W1 = allocateMatrix<double>(HIDDEN_SIZE, INPUT_SIZE);
    net->W2 = allocateMatrix<double>(OUTPUT_SIZE, HIDDEN_SIZE);
    net->b1 = (double*)calloc(HIDDEN_SIZE, sizeof(double));
    net->b2 = (double*)calloc(OUTPUT_SIZE, sizeof(double));

    srand(time(NULL));
    for (int i = 0; i < HIDDEN_SIZE; i++)
        for (int j = 0; j < INPUT_SIZE; j++)
            net->W1[i][j] = ((double)rand() / RAND_MAX) * 0.01;

    for (int i = 0; i < OUTPUT_SIZE; i++)
        for (int j = 0; j < HIDDEN_SIZE; j++)
            net->W2[i][j] = ((double)rand() / RAND_MAX) * 0.01;

    return net;
}

void relu(double* x, int size) {
    for (int i = 0; i < size; i++) {
        x[i] = (x[i] > 0) ? x[i] : 0;
    }
}

void softmax(double* x, int size) {
    double sum = 0;
    for (int i = 0; i < size; i++) {
        x[i] = exp(x[i]);
        sum += x[i];
    }
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

void forward(NeuralNetworkCPU* net, double* input, double* hidden, double* output) {
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        hidden[i] = net->b1[i];
        for (int j = 0; j < INPUT_SIZE; j++)
            hidden[i] += net->W1[i][j] * input[j];
    }
    relu(hidden, HIDDEN_SIZE);

    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output[i] = net->b2[i];
        for (int j = 0; j < HIDDEN_SIZE; j++)
            output[i] += net->W2[i][j] * hidden[j];
    }
    softmax(output, OUTPUT_SIZE);
}

void backward(NeuralNetworkCPU* net, double* input, double* hidden, double* output, double* target) {
    double d_output[OUTPUT_SIZE], d_hidden[HIDDEN_SIZE];

    // Compute output layer gradient
    for (int i = 0; i < OUTPUT_SIZE; i++)
        d_output[i] = output[i] - target[i];

    // Compute hidden layer gradient
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        d_hidden[i] = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++)
            d_hidden[i] += net->W2[j][i] * d_output[j];
        d_hidden[i] *= (hidden[i] > 0);
    }

    // Update weights (gradient descent)
    for (int i = 0; i < OUTPUT_SIZE; i++)
        for (int j = 0; j < HIDDEN_SIZE; j++)
            net->W2[i][j] -= LEARNING_RATE * d_output[i] * hidden[j];

    for (int i = 0; i < HIDDEN_SIZE; i++)
        for (int j = 0; j < INPUT_SIZE; j++)
            net->W1[i][j] -= LEARNING_RATE * d_hidden[i] * input[j];

    for (int i = 0; i < OUTPUT_SIZE; i++)
        net->b2[i] -= LEARNING_RATE * d_output[i];

    for (int i = 0; i < HIDDEN_SIZE; i++)
        net->b1[i] -= LEARNING_RATE * d_hidden[i];
}

void trainCPU(NeuralNetworkCPU* net, double** images, double** labels, int numImages) {
    clock_t total_start = clock();
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        clock_t epoch_start = clock();
        double loss = 0.0;
        int correct = 0;

        for (int i = 0; i < numImages; i++) {
            double hidden[HIDDEN_SIZE], output[OUTPUT_SIZE];
            forward(net, images[i], hidden, output);
            backward(net, images[i], hidden, output, labels[i]);

            // Compute loss & accuracy
            for (int k = 0; k < OUTPUT_SIZE; k++) loss -= labels[i][k] * log(output[k]);
            int pred = 0, actual = 0;
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                if (output[j] > output[pred]) pred = j;
                if (labels[i][j] > labels[i][actual]) actual = j;
            }
            if (pred == actual) correct++;
        }

        printf("Epoch %d - Loss: %.4f - Train Accuracy: %.2f%% - Time: %.3fs\n",
               epoch + 1, loss / numImages, (correct / (double)numImages) * 100, get_time(epoch_start));
    }
    printf("Total training time: %.3fs\n", get_time(total_start));
}

void evaluateCPU(NeuralNetworkCPU* net, double** images, double** labels, int numImages) {
    clock_t total_start = clock();
    int correct = 0;
    for (int i = 0; i < numImages; i++) {
        double hidden[HIDDEN_SIZE], output[OUTPUT_SIZE];
        forward(net, images[i], hidden, output);
        int pred = 0, actual = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            if (output[j] > output[pred]) pred = j;
            if (labels[i][j] > labels[i][actual]) actual = j;
        }
        if (pred == actual) correct++;
    }
    printf("Test Accuracy: %.2f%%\n", (correct / (double)numImages) * 100);
    printf("Total Evaluation time: %.3fs\n", get_time(total_start));

}
