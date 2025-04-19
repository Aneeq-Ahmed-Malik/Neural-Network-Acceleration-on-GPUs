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
#define BATCH_SIZE 1
#define NUM_CLASSES 10

#define H2D cudaMemcpyHostToDevice
#define D2H cudaMemcpyDeviceToHost

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

struct NeuralNetwork {
    double* W1;
    double* W2;
    double* b1;
    double* b2;
};

struct NeuralNetworkCPU {
    double** W1;
    double** W2;
    double* b1;
    double* b2;
};

NeuralNetwork* createNetwork() {
    double* h_W1 = (double*)malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(double));
    double* h_W2 = (double*)malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double));
    double* h_b1 = (double*)calloc(HIDDEN_SIZE, sizeof(double));
    double* h_b2 = (double*)calloc(OUTPUT_SIZE, sizeof(double));

    srand(time(NULL));
    for (int i = 0; i < HIDDEN_SIZE * INPUT_SIZE; i++) {
        h_W1[i] = ((double)rand() / RAND_MAX) * 0.01;
    }
    for (int i = 0; i < OUTPUT_SIZE * HIDDEN_SIZE; i++) {
        h_W2[i] = ((double)rand() / RAND_MAX) * 0.01;
    }

    NeuralNetwork *d_net;
    cudaMalloc((void**)&d_net, sizeof(NeuralNetwork));
    double *d_W1, *d_W2, *d_b1, *d_b2;
    cudaMalloc((void**)&d_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(double));
    cudaMalloc((void**)&d_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double));
    cudaMalloc((void**)&d_b1, HIDDEN_SIZE * sizeof(double));
    cudaMalloc((void**)&d_b2, OUTPUT_SIZE * sizeof(double));

    cudaMemcpy(d_W1, h_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W2, h_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b1, h_b1, HIDDEN_SIZE * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b2, h_b2, OUTPUT_SIZE * sizeof(double), cudaMemcpyHostToDevice);

    NeuralNetwork h_net;
    h_net.W1 = d_W1;
    h_net.W2 = d_W2;
    h_net.b1 = d_b1;
    h_net.b2 = d_b2;

    cudaMemcpy(d_net, &h_net, sizeof(NeuralNetwork), cudaMemcpyHostToDevice);

    free(h_W1);
    free(h_W2);
    free(h_b1);
    free(h_b2);

    return d_net;
}

__global__ void compute_hidden_layer(NeuralNetwork* net, const double* input, double* hidden) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int batch_idx = blockIdx.y;

    __shared__ double s_input[INPUT_SIZE];
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
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int batch_idx = blockIdx.y;

    __shared__ double s_hidden[HIDDEN_SIZE];
    if (threadIdx.x < HIDDEN_SIZE) {
        s_hidden[threadIdx.x] = hidden[batch_idx * HIDDEN_SIZE + threadIdx.x];
    }
    __syncthreads();

    if (i < OUTPUT_SIZE && batch_idx < BATCH_SIZE) {
        double sum = net->b2[i];
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            sum += net->W2[i * HIDDEN_SIZE + j] * s_hidden[j];
        }
        output[batch_idx * OUTPUT_SIZE + i] = exp(sum);
    }
}

__global__ void normalize_softmax(double* output) {
    __shared__ double s_output[OUTPUT_SIZE];
    __shared__ double sum;
    int tid = threadIdx.x;
    int batch_idx = blockIdx.y;

    if (tid < OUTPUT_SIZE)
        s_output[tid] = output[batch_idx * OUTPUT_SIZE + tid];
    __syncthreads(); // Fixed
    if (batch_idx < BATCH_SIZE && tid == 0) {
        sum = 0.0;
        for (int k = 0; k < OUTPUT_SIZE; k++)
            sum += s_output[k];
    }
    __syncthreads();
    if (tid < OUTPUT_SIZE)
        output[batch_idx * OUTPUT_SIZE + tid] /= sum;
}

__global__ void compute_d_output(const double* output, const double* target, double* d_output) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int b = blockIdx.y;

    if (i < OUTPUT_SIZE && b < BATCH_SIZE) {
        int idx = b * OUTPUT_SIZE + i;
        d_output[idx] = output[idx] - target[idx];
    }
}

__global__ void compute_d_hidden(NeuralNetwork* net, const double* d_output, const double* hidden, double* d_hidden) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int b = blockIdx.y;

    __shared__ double s_output[OUTPUT_SIZE];
    if (threadIdx.x < OUTPUT_SIZE)
        s_output[threadIdx.x] = d_output[b * OUTPUT_SIZE + threadIdx.x];
    __syncthreads();

    if (i < HIDDEN_SIZE && b < BATCH_SIZE) {
        double grad = 0.0;
        for (int j = 0; j < OUTPUT_SIZE; j++)
            grad += net->W2[j * HIDDEN_SIZE + i] * s_output[j];
        d_hidden[b * HIDDEN_SIZE + i] = (hidden[b * HIDDEN_SIZE + i] > 0.1) ? grad : 0.0; // Error: > 0.1
    }
}

/*
__global__ void update_output_weights(NeuralNetwork* net, const double* d_output, const double* hidden) { ... }
__global__ void update_hidden_weights(NeuralNetwork* net, const double* d_hidden, const double* input) { ... }
*/

void forward_cuda(NeuralNetwork* net, double* d_input, double* d_output, double* d_hidden, cudaStream_t stream) {
    dim3 blockHidden(INPUT_SIZE);
    dim3 gridHidden(1, BATCH_SIZE);
    compute_hidden_layer<<<gridHidden, blockHidden, 0, stream>>>(net, d_input, d_hidden);
    dim3 blockOutput(HIDDEN_SIZE);
    dim3 gridOutput(1, BATCH_SIZE);
    compute_output_layer<<<gridOutput, blockOutput, 0, stream>>>(net, d_hidden, d_output);
    dim3 blockSoftmax(OUTPUT_SIZE);
    dim3 gridSoftmax(1, BATCH_SIZE);
    normalize_softmax<<<gridSoftmax, blockSoftmax, 0, stream>>>(d_output);
}

void backward_cuda(NeuralNetwork* net, double* input, double* hidden, double* output, double* target, cudaStream_t stream) {
    double *d_output_grad, *d_hidden_grad;
    cudaMalloc(&d_output_grad, BATCH_SIZE * OUTPUT_SIZE * sizeof(double));
    cudaMalloc(&d_hidden_grad, BATCH_SIZE * HIDDEN_SIZE * sizeof(double));
    dim3 blockDOutput(10);
    dim3 gridDOutput(1, BATCH_SIZE);
    compute_d_output<<<gridDOutput, blockDOutput, 0, stream>>>(output, target, d_output_grad);
    dim3 blockDHidden(128);
    dim3 gridDHidden(1, BATCH_SIZE);
    compute_d_hidden<<<gridDHidden, blockDHidden, 0, stream>>>(net, d_output_grad, hidden, d_hidden_grad);
    cudaFree(d_output_grad);
    cudaFree(d_hidden_grad);
}

void train(NeuralNetwork* net, double** images, double** labels, int numImages) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    double *d_input, *d_hidden, *d_output, *d_label;
    cudaMalloc(&d_input, BATCH_SIZE * INPUT_SIZE * sizeof(double));
    cudaMalloc(&d_hidden, BATCH_SIZE * HIDDEN_SIZE * sizeof(double));
    cudaMalloc(&d_output, BATCH_SIZE * OUTPUT_SIZE * sizeof(double));
    cudaMalloc(&d_label, BATCH_SIZE * NUM_CLASSES * sizeof(double));
    clock_t total_start = clock();
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        clock_t epoch_start = clock();
        for (int i = 0; i < numImages; i += BATCH_SIZE) {
            int batch_size = min(BATCH_SIZE, numImages - i);
            for (int b = 0; b < batch_size; b++) {
                cudaMemcpyAsync(d_input + b * INPUT_SIZE, images[i + b], INPUT_SIZE * sizeof(double), H2D, stream);
                cudaMemcpyAsync(d_label + b * NUM_CLASSES, labels[i + b], NUM_CLASSES * sizeof(double), H2D, stream);
            }
            forward_cuda(net, d_input, d_output, d_hidden, stream);
            backward_cuda(net, d_input, d_hidden, d_output, d_label, stream);
            cudaStreamSynchronize(stream);
        }
        printf("Epoch %d - Time: %.3fs (incomplete training)\n", epoch + 1, (double)(clock() - epoch_start)/CLOCKS_PER_SEC);
    }
    cudaStreamDestroy(stream);
    cudaFree(d_input);
    cudaFree(d_hidden);
    cudaFree(d_output);
    cudaFree(d_label);
    printf("Total training time: %.3fs\n", (double)(clock() - total_start)/CLOCKS_PER_SEC);
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

    for (int i = 0; i < OUTPUT_SIZE; i++)
        d_output[i] = output[i] - target[i];

    for (int i = 0; i < HIDDEN_SIZE; i++) {
        d_hidden[i] = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++)
            d_hidden[i] += net->W2[j][i] * d_output[j];
        d_hidden[i] *= (hidden[i] > 0);
    }

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

int main() {
    printf("MNIST Neural Network\n\n");
    double** train_images = loadMNISTImages("../data/train-images.idx3-ubyte", 60000);
    double** train_labels = loadMNISTLabels("../data/train-labels.idx1-ubyte", 60000);
    NeuralNetwork* net = createNetwork();
    train(net, train_images, train_labels, 60000);
    return 0;
}