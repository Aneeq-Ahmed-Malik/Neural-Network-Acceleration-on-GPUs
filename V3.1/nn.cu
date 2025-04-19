#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda.h>

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

// Allocate memory for a matrix
double** allocateMatrix(int rows, int cols) {
    double** mat = (double**)malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++) {
        mat[i] = (double*)malloc(cols * sizeof(double));
    }
    return mat;
}

// Free allocated matrix memory
void freeMatrix(double** mat, int rows) {
    for (int i = 0; i < rows; i++) {
        free(mat[i]);
    }
    free(mat);
}

// Activation functions
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
    
    int i = threadIdx.x;

    __shared__ double s_input[INPUT_SIZE];
    s_input[i] = input[i];
    __syncthreads();

    if (i < HIDDEN_SIZE) {
        double sum = net->b1[i];
        for (int j = 0; j < INPUT_SIZE; j++) {
            sum += net->W1[i * INPUT_SIZE + j] * s_input[j];
        }
        hidden[i] = (sum > 0.0) ? sum : 0.0;
    }
}

// Softmax kernel
__global__ void compute_output_layer(NeuralNetwork* net, const double* hidden, double* output) {
    int i = threadIdx.x;

    __shared__ double s_hidden[HIDDEN_SIZE];
    s_hidden[i] = hidden[i];
    
    __syncthreads();
    if (i < OUTPUT_SIZE) {
        double sum = net->b2[i];
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            sum += net->W2[i * HIDDEN_SIZE + j] * s_hidden[j];
        }
        output[i] = exp(sum);
    }
}

// Normalization kernel for softmax
__global__ void normalize_softmax(double* output) {
    __shared__ double s_output[OUTPUT_SIZE];
    __shared__ double sum;

    int tid = threadIdx.x;

    if (tid < OUTPUT_SIZE)
        s_output[tid] = output[tid];  
    __syncthreads();

    if (tid == 0) {
        sum = 0.0;
        for (int i = 0; i < OUTPUT_SIZE; i++)
            sum += s_output[i];
    }
    __syncthreads();

    if (tid < OUTPUT_SIZE)
        output[tid] = s_output[tid] / (sum + 1e-8);  // Avoid div-by-zero
}



// Kernel 1: d_output = output - target
__global__ void compute_d_output(const double* output, const double* target, double* d_output) {
    int i = threadIdx.x;
    if (i < OUTPUT_SIZE) {
        d_output[i] = output[i] - target[i];
    }
}

// Kernel 2: d_hidden = (W2^T * d_output) * ReLU'(hidden)
__global__ void compute_d_hidden(NeuralNetwork* net, const double* d_output, const double* hidden, double* d_hidden) {
    int i = threadIdx.x;
    __shared__ double s_output[OUTPUT_SIZE];

    if (i< OUTPUT_SIZE)
        s_output[i] = d_output[i];
    __syncthreads();

    if (i < HIDDEN_SIZE) {
        double grad = 0.0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            grad += net->W2[j * HIDDEN_SIZE + i] * s_output[j];
        }
        d_hidden[i] = (hidden[i] > 0.0) ? grad : 0.0;
    }
}

// Kernel 3: update W2 and b2
__global__ void update_output_weights(NeuralNetwork* net, const double* d_output, const double* hidden) {
    int i = blockIdx.x; // OUTPUT neuron index
    int j = threadIdx.x; // HIDDEN neuron index

    if (i < OUTPUT_SIZE && j < HIDDEN_SIZE) {
        int idx = i * HIDDEN_SIZE + j;
        net->W2[idx] -= LEARNING_RATE * d_output[i] * hidden[j];
    }

    if (j == 0 && i < OUTPUT_SIZE) {
        net->b2[i] -= LEARNING_RATE * d_output[i];
    }
}

// Kernel 4: update W1 and b1
__global__ void update_hidden_weights(NeuralNetwork* net, const double* d_hidden, const double* input) {
    int i = blockIdx.x; // HIDDEN neuron index
    int j = threadIdx.x; // INPUT neuron index

    if (i < HIDDEN_SIZE && j < INPUT_SIZE) {
        int idx = i * INPUT_SIZE + j;
        net->W1[idx] -= LEARNING_RATE * d_hidden[i] * input[j];
    }

    if (j == 0 && i < HIDDEN_SIZE) {
        net->b1[i] -= LEARNING_RATE * d_hidden[i];
    }
}


void forward_cuda(NeuralNetwork* net, double* d_input, double* d_output, double* d_hidden, cudaStream_t stream) {
    compute_hidden_layer<<<1, INPUT_SIZE , 0, stream>>>(net, d_input, d_hidden);
    compute_output_layer<<<1, HIDDEN_SIZE, 0, stream>>>(net, d_hidden, d_output);
    normalize_softmax<<<1, OUTPUT_SIZE, 0, stream>>>(d_output);
}


void backward_cuda(NeuralNetwork* net, double* input, double* hidden, double* output, double* target, cudaStream_t stream) {
    double *d_output_grad, *d_hidden_grad;
    cudaMalloc(&d_output_grad, OUTPUT_SIZE * sizeof(double));
    cudaMalloc(&d_hidden_grad, HIDDEN_SIZE * sizeof(double));

    compute_d_output<<<1, OUTPUT_SIZE, 0, stream>>>(output, target, d_output_grad);
    compute_d_hidden<<<1, HIDDEN_SIZE, 0, stream>>>(net, d_output_grad, hidden, d_hidden_grad);
    update_output_weights<<<OUTPUT_SIZE, HIDDEN_SIZE, 0, stream>>>(net, d_output_grad, hidden);
    update_hidden_weights<<<HIDDEN_SIZE, INPUT_SIZE, 0, stream>>>(net, d_hidden_grad, input);

    cudaFree(d_output_grad);
    cudaFree(d_hidden_grad);
}


void train(NeuralNetwork* net, double** images, double** labels, int numImages) {
    // Create 2 streams
    cudaStream_t streams[2];
    cudaStreamCreate(&streams[0]);
    cudaStreamCreate(&streams[1]);

    // Allocate device memory
    double *d_hidden, *d_output, *d_input, *d_label;
    cudaMalloc(&d_hidden, HIDDEN_SIZE * sizeof(double));
    cudaMalloc(&d_output, OUTPUT_SIZE * sizeof(double));
    cudaMalloc(&d_input, INPUT_SIZE * sizeof(double));
    cudaMalloc(&d_label, NUM_CLASSES * sizeof(double));

    // Pinned memory for async transfers
    double *pinned_output;
    cudaHostAlloc(&pinned_output, OUTPUT_SIZE * sizeof(double), cudaHostAllocDefault);

    clock_t total_start = clock();
    double *loss;
    cudaMallocManaged((void**)&loss, sizeof(double));
    // SPECIAL CASE: First forward pass (stream 0)
    cudaMemcpyAsync(d_input, images[0], INPUT_SIZE * sizeof(double), H2D, streams[0]);
    cudaMemcpyAsync(d_label, labels[0], NUM_CLASSES * sizeof(double), H2D, streams[0]);

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        clock_t epoch_start = clock();
        *loss = 0;
        forward_cuda(net, d_input, d_output, d_hidden, streams[0]);
        int correct = 0;

        for (int i = 0; i < numImages; i++) {
            int current_stream = i % 2;
            int next_stream = (i + 1) % 2;

            backward_cuda(net, d_input, d_hidden, d_output, d_label, streams[current_stream]);
            cudaMemcpyAsync(pinned_output, d_output, OUTPUT_SIZE * sizeof(double), D2H, streams[current_stream]);


            if (i + 1 < numImages) {
                cudaMemcpyAsync(d_input, images[i+1], INPUT_SIZE * sizeof(double), H2D, streams[next_stream]);
                cudaMemcpyAsync(d_label, labels[i+1], NUM_CLASSES * sizeof(double), H2D, streams[next_stream]);
                forward_cuda(net, d_input, d_output, d_hidden, streams[next_stream]);
            }

            cudaStreamSynchronize(streams[current_stream]);  // Only needed for metrics

            // 4. Sync current stream for metrics
            for (int k = 0; k < OUTPUT_SIZE; k++)
                *loss -= labels[i][k] * log(pinned_output[k]);


            int pred = 0, actual = 0;
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                if (pinned_output[j] > pinned_output[pred]) pred = j;
                if (labels[i][j] > labels[i][actual]) actual = j;
            }
            if (pred == actual) correct++;

            // Compute metrics

        }

        printf("Epoch %d - Loss: %.4f - Accuracy: %.2f%% - Time: %.3fs\n",
               epoch + 1, *loss / numImages, (correct / (double)numImages) * 100,
               (double)(clock() - epoch_start)/CLOCKS_PER_SEC);

    }

    // Cleanup
    cudaStreamDestroy(streams[0]);
    cudaStreamDestroy(streams[1]);
    cudaFreeHost(pinned_output);
    cudaFree(d_hidden);
    cudaFree(d_output);
    cudaFree(d_input);
    cudaFree(d_label);

    printf("Total training time: %.3fs\n", (double)(clock() - total_start)/CLOCKS_PER_SEC);
}

/*
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
        forward_cuda(net, d_input, d_output, d_hidden);
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
*/

// Read MNIST dataset
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



NeuralNetworkCPU* createNetworkCPU();
void forward(NeuralNetworkCPU* net, double* input, double* hidden, double* output);
void backward(NeuralNetworkCPU* net, double* input, double* hidden, double* output, double* target);
void trainCPU(NeuralNetworkCPU* net, double** images, double** labels, int numImages);
void evaluateCPU(NeuralNetworkCPU* net, double** images, double** labels, int numImages);

// Main function
int main() {
    printf("MNIST Neural Network\n\n");

    double** train_images = loadMNISTImages("../data/train-images.idx3-ubyte", 60000);
    double** train_labels = loadMNISTLabels("../data/train-labels.idx1-ubyte", 60000);
    double** test_images = loadMNISTImages("../data/t10k-images.idx3-ubyte", 10000);
    double** test_labels = loadMNISTLabels("../data/t10k-labels.idx1-ubyte", 10000);


    // NeuralNetworkCPU* netCPU = createNetworkCPU();
    // trainCPU(netCPU, train_images, train_labels, 60000);
    // evaluateCPU(netCPU, test_images, test_labels, 10000);

    NeuralNetwork* net = createNetwork();
    train(net, train_images, train_labels, 60000);
    // evaluate(net, test_images, test_labels, 10000);



    // freeNetwork(net);
    return 0;
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

// Forward pass
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

// Backpropagation
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

// Train network
void trainCPU(NeuralNetworkCPU* net, double** images, double** labels, int numImages) {
    clock_t total_start = clock();
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        clock_t epoch_start = clock();
        double loss = 0.0;
        int correct = 0;

        for (int i = 0; i < numImages; i++) {
            double hidden[HIDDEN_SIZE], output[OUTPUT_SIZE];
            forward(net, images[i], hidden, output);
            for (int k = 0; k < OUTPUT_SIZE; k++) loss -= labels[i][k] * log(output[k]);
            backward(net, images[i], hidden, output, labels[i]);

            // Compute loss & accuracy
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
