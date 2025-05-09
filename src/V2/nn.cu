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
    if (i < HIDDEN_SIZE) {
        double sum = net->b1[i];
        for (int j = 0; j < INPUT_SIZE; j++) {
            sum += net->W1[i * INPUT_SIZE + j] * input[j];
        }
        hidden[i] = (sum > 0.0) ? sum : 0.0;
    }
}

__global__ void compute_output_layer(NeuralNetwork* net, const double* hidden, double* output) {
    int i = threadIdx.x;
    if (i < OUTPUT_SIZE) {
        double sum = net->b2[i];
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            sum += net->W2[i * HIDDEN_SIZE + j] * hidden[j];
        }
        output[i] = exp(sum);
    }
}

__global__ void normalize_softmax(double* output) {
    double sum = 0.0;
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        sum += output[i];
    }
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output[i] /= sum;
    }
}


__global__ void compute_d_output(const double* output, const double* target, double* d_output) {
    int i = threadIdx.x;
    if (i < OUTPUT_SIZE) {
        d_output[i] = output[i] - target[i];
    }
}

__global__ void compute_d_hidden(NeuralNetwork* net, const double* d_output, const double* hidden, double* d_hidden) {
    int i = threadIdx.x;
    if (i < HIDDEN_SIZE) {
        double grad = 0.0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            grad += net->W2[j * HIDDEN_SIZE + i] * d_output[j];
        }
        d_hidden[i] = (hidden[i] > 0.0) ? grad : 0.0;
    }
}

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



void forward_cuda(NeuralNetwork* net, double* d_input, double* d_output, double* d_hidden) {
    
    compute_hidden_layer<<<1, HIDDEN_SIZE>>>(net, d_input, d_hidden);
    compute_output_layer<<<1, OUTPUT_SIZE>>>(net, d_hidden, d_output);
    normalize_softmax<<<1, 1>>>(d_output);

}


void backward_cuda(NeuralNetwork* net, double* input, double* hidden, double* output, double* target) {
    double *d_output_grad, *d_hidden_grad;

    cudaMalloc(&d_output_grad, OUTPUT_SIZE * sizeof(double));
    cudaMalloc(&d_hidden_grad, HIDDEN_SIZE * sizeof(double));

    compute_d_output<<<1, OUTPUT_SIZE>>>(output, target, d_output_grad);
    compute_d_hidden<<<1, HIDDEN_SIZE>>>(net, d_output_grad, hidden, d_hidden_grad);
    update_output_weights<<<OUTPUT_SIZE, HIDDEN_SIZE>>>(net, d_output_grad, hidden);
    update_hidden_weights<<<HIDDEN_SIZE, INPUT_SIZE>>>(net, d_hidden_grad, input);

    cudaFree(d_output_grad);
    cudaFree(d_hidden_grad);
}

void train(NeuralNetwork* net, double** images, double** labels, int numImages) {
    // clock_t total_start = clock();
    double* hidden = (double*)malloc(sizeof(double) * HIDDEN_SIZE);
    double* output = (double*)malloc(sizeof(double) * OUTPUT_SIZE);
    double* d_hidden, *d_output, *d_input, *d_label;
    cudaMalloc((void **)&d_label, sizeof(double) * NUM_CLASSES);
    cudaMalloc((void **)&d_hidden, sizeof(double) * HIDDEN_SIZE);
    cudaMalloc((void **)&d_output, sizeof(double) * OUTPUT_SIZE);
    cudaMalloc((void **)&d_input, sizeof(double) * INPUT_SIZE);

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        clock_t epoch_start = clock();
        double loss = 0.0;
        int correct = 0;

        for (int i = 0; i < numImages; i++) {

            cudaMemcpy(d_input, images[i], sizeof(double) * INPUT_SIZE, H2D);
            forward_cuda(net, d_input, d_output, d_hidden);

            cudaMemcpy(d_label, labels[i], sizeof(double) * NUM_CLASSES, H2D);
            backward_cuda(net, d_input, d_hidden, d_output, d_label);

            cudaMemcpy(output, d_output, sizeof(double) * OUTPUT_SIZE, D2H);

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
    // printf("Total training time: %.3fs\n", get_time(total_start));
}

void evaluate(NeuralNetwork* net, double** images, double** labels, int numImages) {
    // clock_t total_start = clock();
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
    // printf("Total Evaluation time: %.3fs\n", get_time(total_start));

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


NeuralNetworkCPU* createNetworkCPU();
void forward(NeuralNetworkCPU* net, double* input, double* hidden, double* output);
void backward(NeuralNetworkCPU* net, double* input, double* hidden, double* output, double* target);
void trainCPU(NeuralNetworkCPU* net, double** images, double** labels, int numImages);
void evaluateCPU(NeuralNetworkCPU* net, double** images, double** labels, int numImages);

// Main function
int main() {
    printf("MNIST Neural Network\n\n");

    double** train_images = loadMNISTImages("../../data/train-images.idx3-ubyte", 60000);
    double** train_labels = loadMNISTLabels("../../data/train-labels.idx1-ubyte", 60000);
    double** test_images = loadMNISTImages("../../data/t10k-images.idx3-ubyte", 10000);
    double** test_labels = loadMNISTLabels("../../data/t10k-labels.idx1-ubyte", 10000);

    // Timing for GPU training
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
    // clock_t total_start = clock();
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
    // printf("Total training time: %.3fs\n", get_time(total_start));
}

void evaluateCPU(NeuralNetworkCPU* net, double** images, double** labels, int numImages) {
    // clock_t total_start = clock();
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
    // printf("Total Evaluation time: %.3fs\n", get_time(total_start));

}
