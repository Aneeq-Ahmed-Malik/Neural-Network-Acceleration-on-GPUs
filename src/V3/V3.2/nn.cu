#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda.h>
#include <nvToolsExt.h>

#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.01
#define EPOCHS 3
#define BATCH_SIZE 64
#define NUM_CLASSES 10  // Digits 0-9

#define H2D cudaMemcpyHostToDevice
#define D2H cudaMemcpyDeviceToHost

// Timer function
double get_time(clock_t start) {
    return (double)(clock() - start) / CLOCKS_PER_SEC;
}

double** allocateMatrix(int rows, int cols) {
    double** mat = (double**)malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++) {
        mat[i] = (double*)malloc(cols * sizeof(double));
    }
    return mat;
}

void freeMatrix(double** mat, int rows) {
    for (int i = 0; i < rows; i++) {
        free(mat[i]);
    }
    free(mat);
}

struct NeuralNetwork{
    double* W1;   // 2D matrix: HIDDEN_SIZE x INPUT_SIZE
    double* W2;   // 2D matrix: OUTPUT_SIZE x HIDDEN_SIZE
    double* b1;   // Bias vector: HIDDEN_SIZE
    double* b2;   // Bias vector: OUTPUT_SIZE
} ;

struct NeuralNetworkCPU{
    double** W1;   // 2D matrix: HIDDEN_SIZE x INPUT_SIZE
    double** W2;   // 2D matrix: OUTPUT_SIZE x HIDDEN_SIZE
    double* b1;   // Bias vector: HIDDEN_SIZE
    double* b2;   // Bias vector: OUTPUT_SIZE
} ;

NeuralNetwork* createNetwork() {
    // --- Host (CPU) allocations ---
    double* h_W1 = (double*)malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(double));
    double* h_W2 = (double*)malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double));
    double* h_b1 = (double*)calloc(HIDDEN_SIZE, sizeof(double)); // Zero-init biases
    double* h_b2 = (double*)calloc(OUTPUT_SIZE, sizeof(double));

    // Random weight initialization
    srand(time(NULL));
    for (int i = 0; i < HIDDEN_SIZE * INPUT_SIZE; i++) {
        h_W1[i] = ((double)rand() / RAND_MAX) * 0.01;  // Small random values
    }
    for (int i = 0; i < OUTPUT_SIZE * HIDDEN_SIZE; i++) {
        h_W2[i] = ((double)rand() / RAND_MAX) * 0.01;
    }

    // --- GPU (Device) allocations ---
    NeuralNetwork *d_net;
    cudaMalloc((void**)&d_net, sizeof(NeuralNetwork));

    // Allocate GPU memory for weights/biases
    double *d_W1, *d_W2, *d_b1, *d_b2;
    cudaMalloc((void**)&d_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(double));
    cudaMalloc((void**)&d_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double));
    cudaMalloc((void**)&d_b1, HIDDEN_SIZE * sizeof(double));
    cudaMalloc((void**)&d_b2, OUTPUT_SIZE * sizeof(double));

    // Copy host data to GPU
    cudaMemcpy(d_W1, h_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W2, h_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b1, h_b1, HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b2, h_b2, OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);

    // Create a temporary host struct with GPU pointers
    NeuralNetwork h_net;
    h_net.W1 = d_W1;
    h_net.W2 = d_W2;
    h_net.b1 = d_b1;
    h_net.b2 = d_b2;

    // Copy the struct to GPU
    cudaMemcpy(d_net, &h_net, sizeof(NeuralNetwork), cudaMemcpyHostToDevice);
    // Free host memory
    free(h_W1);
    free(h_W2);
    free(h_b1);
    free(h_b2);

    return d_net;
}


__global__ void compute_hidden_layer(NeuralNetwork* net, const double* input, double* hidden) {
    int i = threadIdx.x + blockIdx.x * blockDim.x; // Hidden neuron index
    int batch_idx = blockIdx.y; // Batch index

    __shared__ double s_input[INPUT_SIZE]; // Shared memory for input (INPUT_SIZE = 784)
    if (threadIdx.x < INPUT_SIZE) {
        s_input[threadIdx.x] = input[batch_idx * INPUT_SIZE + threadIdx.x];
    }
    __syncthreads();

    if (i < HIDDEN_SIZE && batch_idx < BATCH_SIZE) {
        double sum = net->b1[i];
        for (int j = 0; j < INPUT_SIZE; j++) {
            sum += net->W1[i * INPUT_SIZE + j] * s_input[j];
        }
        hidden[batch_idx * HIDDEN_SIZE + i] = (sum > 0.0) ? sum : 0.0;
    }
}

__global__ void compute_output_layer(NeuralNetwork* net, const double* hidden, double* output) {
    int i = threadIdx.x + blockIdx.x * blockDim.x; // Output neuron index
    int batch_idx = blockIdx.y; // Batch index

    __shared__ double s_hidden[HIDDEN_SIZE]; // Shared memory for hidden (HIDDEN_SIZE = 128)
    if (threadIdx.x < HIDDEN_SIZE) {
        s_hidden[threadIdx.x] = hidden[batch_idx * HIDDEN_SIZE + threadIdx.x];
    }
    __syncthreads();

    if (i < OUTPUT_SIZE && batch_idx < BATCH_SIZE) {
        double sum = net->b2[i];
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            sum += net->W2[i * HIDDEN_SIZE + j] * s_hidden[j];
        }
        output[batch_idx * OUTPUT_SIZE + i] = exp(sum); // Store exp for softmax
    }
}

__global__ void normalize_softmax(double* output) {
    __shared__ double s_output[OUTPUT_SIZE];
    __shared__ double sum;

    int tid = threadIdx.x;
    int batch_idx = blockIdx.y; // Batch index

    if (tid < OUTPUT_SIZE)
        s_output[tid] = output[batch_idx * OUTPUT_SIZE + tid];  
    __syncthreads();


    if (batch_idx < BATCH_SIZE) { // One thread per sample computes sum
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



__global__ void compute_d_output(const double* output, const double* target, double* d_output) {
    int i = threadIdx.x + blockIdx.x * blockDim.x; // Output neuron index
    int b = blockIdx.y; // Batch index

    if (i < OUTPUT_SIZE && b < BATCH_SIZE) {
        int idx = b * OUTPUT_SIZE + i;
        d_output[idx] = output[idx] - target[idx];
    }
}

__global__ void compute_d_hidden(NeuralNetwork* net, const double* d_output, const double* hidden, double* d_hidden) {
    int i = threadIdx.x + blockIdx.x * blockDim.x; // Hidden neuron index
    int b = blockIdx.y; // Batch index

    __shared__ double s_output[OUTPUT_SIZE]; // Shared memory for d_output (OUTPUT_SIZE = 10)
    if (threadIdx.x < OUTPUT_SIZE)
        s_output[threadIdx.x] = d_output[b * OUTPUT_SIZE + threadIdx.x];
    __syncthreads();

    if (i < HIDDEN_SIZE && b < BATCH_SIZE) {
        double grad = 0.0;
        for (int j = 0; j < OUTPUT_SIZE; j++) 
            grad += net->W2[j * HIDDEN_SIZE + i] * s_output[j];

        d_hidden[b * HIDDEN_SIZE + i] = (hidden[b * HIDDEN_SIZE + i] > 0.0) ? grad : 0.0;
    }
}

__global__ void update_output_weights(NeuralNetwork* net, const double* d_output, const double* hidden) {
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

__global__ void update_hidden_weights(NeuralNetwork* net, const double* d_hidden, const double* input) {
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


void forward_cuda(NeuralNetwork* net, double* d_input, double* d_output, double* d_hidden, cudaStream_t stream) {
    // Hidden layer
    dim3 blockHidden(INPUT_SIZE); // Threads per block (matches HIDDEN_SIZE)
    dim3 gridHidden(1, BATCH_SIZE); // One block per sample
    compute_hidden_layer<<<gridHidden, blockHidden, 0, stream>>>(net, d_input, d_hidden);

    // Output layer
    dim3 blockOutput(HIDDEN_SIZE); // Threads per block (matches OUTPUT_SIZE)
    dim3 gridOutput(1, BATCH_SIZE);
    compute_output_layer<<<gridOutput, blockOutput, 0, stream>>>(net, d_hidden, d_output);

    // Softmax normalization
    dim3 blockSoftmax(OUTPUT_SIZE); // One thread per sample
    dim3 gridSoftmax(1, BATCH_SIZE);
    normalize_softmax<<<gridSoftmax, blockSoftmax, 0, stream>>>(d_output);
}


void backward_cuda(NeuralNetwork* net, double* input, double* hidden, double* output, double* target, cudaStream_t stream) {
    double *d_output_grad, *d_hidden_grad;
    cudaMalloc(&d_output_grad, BATCH_SIZE * OUTPUT_SIZE * sizeof(double));
    cudaMalloc(&d_hidden_grad, BATCH_SIZE * HIDDEN_SIZE * sizeof(double));

    // Compute output gradients
    dim3 blockDOutput(10);
    dim3 gridDOutput(1, BATCH_SIZE);
    compute_d_output<<<gridDOutput, blockDOutput, 0, stream>>>(output, target, d_output_grad);

    // Compute hidden gradients
    dim3 blockDHidden(128);
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


void train(NeuralNetwork* net, double** images, double** labels, int numImages) {
    cudaStream_t streams[2];
    cudaStreamCreate(&streams[0]);
    cudaStreamCreate(&streams[1]);

    double *d_input[2], *d_hidden[2], *d_output[2], *d_label[2];
    for (int s = 0; s < 2; s++) {
        cudaMalloc(&d_input[s], BATCH_SIZE * INPUT_SIZE * sizeof(double));
        cudaMalloc(&d_hidden[s], BATCH_SIZE * HIDDEN_SIZE * sizeof(double));
        cudaMalloc(&d_output[s], BATCH_SIZE * OUTPUT_SIZE * sizeof(double));
        cudaMalloc(&d_label[s], BATCH_SIZE * NUM_CLASSES * sizeof(double));
    }

    double *pinned_output;
    cudaHostAlloc(&pinned_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(double), cudaHostAllocDefault);

    clock_t total_start = clock();

    int first_batch_size = min(BATCH_SIZE, numImages);
    for (int b = 0; b < first_batch_size; b++) {
        cudaMemcpyAsync(d_input[0] + b * INPUT_SIZE, images[b], INPUT_SIZE * sizeof(double), H2D, streams[0]);
        cudaMemcpyAsync(d_label[0] + b * NUM_CLASSES, labels[b], NUM_CLASSES * sizeof(double), H2D, streams[0]);
    }

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        clock_t epoch_start = clock();
        double loss = 0.0;
        int correct = 0;
        forward_cuda(net, d_input[0], d_output[0], d_hidden[0], streams[0]);

        for (int i = 0; i < numImages; i += BATCH_SIZE) {
            int current_stream = (i / BATCH_SIZE) % 2;
            int next_stream = ((i / BATCH_SIZE) + 1) % 2;
            int batch_size = min(BATCH_SIZE, numImages - i);

            backward_cuda(net, d_input[current_stream], d_hidden[current_stream], d_output[current_stream], d_label[current_stream], streams[current_stream]);

            cudaMemcpyAsync(pinned_output, d_output[current_stream], batch_size * OUTPUT_SIZE * sizeof(double), D2H, streams[current_stream]);

            if (i + BATCH_SIZE < numImages) {
                int next_batch_size = min(BATCH_SIZE, numImages - (i + BATCH_SIZE));
                for (int b = 0; b < next_batch_size; b++) {
                    cudaMemcpyAsync(d_input[next_stream] + b * INPUT_SIZE, images[i + BATCH_SIZE + b], INPUT_SIZE * sizeof(double), H2D, streams[next_stream]);
                    cudaMemcpyAsync(d_label[next_stream] + b * NUM_CLASSES, labels[i + BATCH_SIZE + b], NUM_CLASSES * sizeof(double), H2D, streams[next_stream]);
                }
                forward_cuda(net, d_input[next_stream], d_output[next_stream], d_hidden[next_stream], streams[next_stream]);
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
               epoch + 1, loss / numImages, (correct / (double)numImages) * 100,
               (double)(clock() - epoch_start)/CLOCKS_PER_SEC);
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

    printf("Total training time: %.3fs\n", (double)(clock() - total_start)/CLOCKS_PER_SEC);
}


void evaluate(NeuralNetwork* net, double** images, double** labels, int numImages) {
    clock_t total_start = clock();
    int correct = 0;
    double* hidden = (double*)malloc(sizeof(double) * HIDDEN_SIZE);
    double* output = (double*)malloc(sizeof(double) * OUTPUT_SIZE);
    double* d_hidden, *d_output, *d_input;
    
    cudaMalloc((void **)&d_hidden, sizeof(double) * HIDDEN_SIZE);
    cudaMalloc((void **)&d_output, sizeof(double) * OUTPUT_SIZE);
    cudaMalloc((void **)&d_input, sizeof(double) * INPUT_SIZE);

    for (int i = 0; i < numImages; i++) {

        cudaMemcpy(d_input, images[i], sizeof(double) * INPUT_SIZE, H2D);
        forward_cuda(net, d_input, d_output, d_hidden, 0);
        cudaMemcpy(output, d_output, sizeof(double) * OUTPUT_SIZE, D2H);

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


void freeNetwork(NeuralNetworkCPU* net) {
    freeMatrix(net->W1, HIDDEN_SIZE);
    freeMatrix(net->W2, OUTPUT_SIZE);
    free(net->b1);
    free(net->b2);
    free(net);
}

void freeNetwork(NeuralNetwork* net) {
    cudaFree(net->W1);
    cudaFree(net->W2);
    cudaFree(net->b1);
    cudaFree(net->b2);
    cudaFree(net);
}



double** loadMNISTImages(const char* filename, int numImages);
double** loadMNISTLabels(const char* filename, int numLabels) ;

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

    double** train_images = loadMNISTImages("../../../data/train-images.idx3-ubyte", 60000);
    double** train_labels = loadMNISTLabels("../../../data/train-labels.idx1-ubyte", 60000);
    double** test_images = loadMNISTImages("../../../data/t10k-images.idx3-ubyte", 10000);
    double** test_labels = loadMNISTLabels("../../../data/t10k-labels.idx1-ubyte", 10000);

    // Timing for GPU training
    printf("Batch size: %d\n", BATCH_SIZE);
    printf("\nStarting GPU training...\n");
    clock_t total_gpu_train = clock();
    NeuralNetwork* net = createNetwork();
    train(net, train_images, train_labels, 60000);
    double gpu_train_time = get_time(total_gpu_train);
    printf("\nGPU Training time: %.3fs\n", gpu_train_time);

    // Timing for GPU evaluation
    clock_t total_gpu_eval = clock();
    evaluate(net, test_images, test_labels, 10000);
    double gpu_eval_time = get_time(total_gpu_eval);
    printf("GPU Evaluation time: %.3fs\n", gpu_eval_time);

    // Overall GPU time (Training + Evaluation)
    double gpu_total_time = gpu_train_time + gpu_eval_time;
    printf("\nTotal GPU time (Training + Evaluation): %.3fs\n\n", gpu_total_time);


    // Timing for CPU training
    printf("\nStarting CPU training...\n");
    clock_t total_cpu_train = clock();
    NeuralNetworkCPU* netCPU = createNetworkCPU();
    trainCPU(netCPU, train_images, train_labels, 60000);
    double cpu_train_time = get_time(total_cpu_train);
    printf("\nCPU Training time: %.3fs\n", cpu_train_time);

    // Timing for CPU evaluation
    clock_t total_cpu_eval = clock();
    evaluateCPU(netCPU, test_images, test_labels, 10000);
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


double** loadMNISTImages(const char* filename, int numImages) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening %s\n", filename);
        exit(1);
    }
    fseek(file, 16, SEEK_SET);
    double** images = allocateMatrix(numImages, INPUT_SIZE);
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

double** loadMNISTLabels(const char* filename, int numLabels) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening %s\n", filename);
        exit(1);
    }
    fseek(file, 8, SEEK_SET);
    double** labels = allocateMatrix(numLabels, OUTPUT_SIZE);
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
    net->W1 = allocateMatrix(HIDDEN_SIZE, INPUT_SIZE);
    net->W2 = allocateMatrix(OUTPUT_SIZE, HIDDEN_SIZE);
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
