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

    cudaMemcpy(d_W1, h_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(double), H2D);
    cudaMemcpy(d_W2, h_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double), H2D);
    cudaMemcpy(d_b1, h_b1, HIDDEN_SIZE * sizeof(double), H2D);
    cudaMemcpy(d_b2, h_b2, OUTPUT_SIZE * sizeof(double), H2D);

    NeuralNetwork h_net;
    h_net.W1 = d_W1;
    h_net.W2 = d_W2;
    h_net.b1 = d_b1;
    h_net.b2 = d_b2;

    cudaMemcpy(d_net, &h_net, sizeof(NeuralNetwork), H2D);
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

void forward_cuda(NeuralNetwork* net, double* d_input, double* d_output, double* d_hidden) {
    double* h_hidden = (double*)malloc(HIDDEN_SIZE * sizeof(double));
    double* h_output = (double*)malloc(OUTPUT_SIZE * sizeof(double));

    compute_hidden_layer<<<1, HIDDEN_SIZE>>>(net, d_input, d_hidden);
    cudaMemcpy(h_hidden, d_hidden, HIDDEN_SIZE * sizeof(double), D2H);

    NeuralNetworkCPU net_cpu;
    net_cpu.W2 = allocateMatrix(OUTPUT_SIZE, HIDDEN_SIZE);
    net_cpu.b2 = (double*)calloc(OUTPUT_SIZE, sizeof(double));
    cudaMemcpy(net_cpu.W2[0], net->W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double), D2H);
    cudaMemcpy(net_cpu.b2, net->b2, OUTPUT_SIZE * sizeof(double), D2H);

    for (int i = 0; i < OUTPUT_SIZE; i++) {
        h_output[i] = net_cpu.b2[i];
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            h_output[i] += net_cpu.W2[i][j] * h_hidden[j];
        }
    }
    softmax(h_output, OUTPUT_SIZE);

    cudaMemcpy(d_output, h_output, OUTPUT_SIZE * sizeof(double), H2D);
    free(h_hidden);
    free(h_output);
    freeMatrix(net_cpu.W2, OUTPUT_SIZE);
    free(net_cpu.b2);
}

void backward_cuda(NeuralNetwork* net, double* d_input, double* d_hidden, double* d_output, double* d_label) {
    double* h_input = (double*)malloc(INPUT_SIZE * sizeof(double));
    double* h_hidden = (double*)malloc(HIDDEN_SIZE * sizeof(double));
    double* h_output = (double*)malloc(OUTPUT_SIZE * sizeof(double));
    double* h_label = (double*)malloc(NUM_CLASSES * sizeof(double));

    cudaMemcpy(h_input, d_input, INPUT_SIZE * sizeof(double), D2H);
    cudaMemcpy(h_hidden, d_hidden, HIDDEN_SIZE * sizeof(double), D2H);
    cudaMemcpy(h_output, d_output, OUTPUT_SIZE * sizeof(double), D2H);
    cudaMemcpy(h_label, d_label, NUM_CLASSES * sizeof(double), D2H);

    NeuralNetworkCPU net_cpu;
    net_cpu.W1 = allocateMatrix(HIDDEN_SIZE, INPUT_SIZE);
    net_cpu.W2 = allocateMatrix(OUTPUT_SIZE, HIDDEN_SIZE);
    net_cpu.b1 = (double*)calloc(HIDDEN_SIZE, sizeof(double));
    net_cpu.b2 = (double*)calloc(OUTPUT_SIZE, sizeof(double));
    cudaMemcpy(net_cpu.W1[0], net->W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(double), D2H);
    cudaMemcpy(net_cpu.W2[0], net->W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double), D2H);
    cudaMemcpy(net_cpu.b1, net->b1, HIDDEN_SIZE * sizeof(double), D2H);
    cudaMemcpy(net_cpu.b2, net->b2, OUTPUT_SIZE * sizeof(double), D2H);

    double d_output[OUTPUT_SIZE], d_hidden[HIDDEN_SIZE];
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        d_output[i] = h_output[i] - h_label[i];
    }

    for (int i = 0; i < HIDDEN_SIZE; i++) {
        d_hidden[i] = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            d_hidden[i] += net_cpu.W2[j][i] * d_output[j];
        }
        d_hidden[i] *= (h_hidden[i] > 0);
    }

    for (int i = 0; i < OUTPUT_SIZE; i++) {
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            net_cpu.W2[i][j] -= LEARNING_RATE * d_output[i] * h_hidden[j];
        }
        net_cpu.b2[i] -= LEARNING_RATE * d_output[i];
    }

    for (int i = 0; i < HIDDEN_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            net_cpu.W1[i][j] -= LEARNING_RATE * d_hidden[i] * h_input[j];
        }
        net_cpu.b1[i] -= LEARNING_RATE * d_hidden[i];
    }

    cudaMemcpy(net->W1, net_cpu.W1[0], HIDDEN_SIZE * INPUT_SIZE * sizeof(double), H2D);
    cudaMemcpy(net->W2, net_cpu.W2[0], OUTPUT_SIZE * HIDDEN_SIZE * sizeof(double), H2D);
    cudaMemcpy(net->b1, net_cpu.b1, HIDDEN_SIZE * sizeof(double), H2D);
    cudaMemcpy(net->b2, net_cpu.b2, OUTPUT_SIZE * sizeof(double), H2D);

    free(h_input);
    free(h_hidden);
    free(h_output);
    free(h_label);
    freeMatrix(net_cpu.W1, HIDDEN_SIZE);
    freeMatrix(net_cpu.W2, OUTPUT_SIZE);
    free(net_cpu.b1);
    free(net_cpu.b2);
}

void train(NeuralNetwork* net, double** images, double** labels, int numImages) {
    clock_t total_start = clock();
    double* d_hidden, *d_output, *d_input, *d_label;
    cudaMalloc((void**)&d_hidden, HIDDEN_SIZE * sizeof(double));
    cudaMalloc((void**)&d_output, OUTPUT_SIZE * sizeof(double));
    cudaMalloc((void**)&d_input, INPUT_SIZE * sizeof(double));
    cudaMalloc((void**)&d_label, NUM_CLASSES * sizeof(double));

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        clock_t epoch_start = clock();
        double loss = 0.0;
        int correct = 0;

        for (int i = 0; i < numImages; i++) {
            cudaMemcpy(d_input, images[i], INPUT_SIZE * sizeof(double), H2D);
            forward_cuda(net, d_input, d_output, d_hidden);
            cudaMemcpy(d_label, labels[i], NUM_CLASSES * sizeof(double), H2D);
            backward_cuda(net, d_input, d_hidden, d_output, d_label);

            double* output = (double*)malloc(OUTPUT_SIZE * sizeof(double));
            cudaMemcpy(output, d_output, OUTPUT_SIZE * sizeof(double), D2H);

            for (int k = 0; k < OUTPUT_SIZE; k++) loss -= labels[i][k] * log(output[k]);
            int pred = 0, actual = 0;
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                if (output[j] > output[pred]) pred = j;
                if (labels[i][j] > labels[i][actual]) actual = j;
            }
            if (pred == actual) correct++;
            free(output);
        }

        printf("Epoch %d - Loss: %.4f - Train Accuracy: %.2f%% - Time: %.3fs\n",
               epoch + 1, loss / numImages, (correct / (double)numImages) * 100, get_time(epoch_start));
    }
    printf("Total training time: %.3fs\n", get_time(total_start));
    cudaFree(d_hidden);
    cudaFree(d_output);
    cudaFree(d_input);
    cudaFree(d_label);
}

void evaluate(NeuralNetwork* net, double** images, double** labels, int numImages) {
    clock_t total_start = clock();
    int correct = 0;
    double* d_hidden, *d_output, *d_input;
    cudaMalloc((void**)&d_hidden, HIDDEN_SIZE * sizeof(double));
    cudaMalloc((void**)&d_output, OUTPUT_SIZE * sizeof(double));
    cudaMalloc((void**)&d_input, INPUT_SIZE * sizeof(double));

    for (int i = 0; i < numImages; i++) {
        cudaMemcpy(d_input, images[i], INPUT_SIZE * sizeof(double), H2D);
        forward_cuda(net, d_input, d_output, d_hidden);
        double* output = (double*)malloc(OUTPUT_SIZE * sizeof(double));
        cudaMemcpy(output, d_output, OUTPUT_SIZE * sizeof(double), D2H);

        int pred = 0, actual = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            if (output[j] > output[pred]) pred = j;
            if (labels[i][j] > labels[i][actual]) actual = j;
        }
        if (pred == actual) correct++;
        free(output);
    }
    printf("Test Accuracy: %.2f%%\n", (correct / (double)numImages) * 100);
    printf("Total Evaluation time: %.3fs\n", get_time(total_start));
    cudaFree(d_hidden);
    cudaFree(d_output);
    cudaFree(d_input);
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
    printf("MNIST Neural Network (Kernel 1)\n\n");

    double** train_images = loadMNISTImages("../data/train-images.idx3-ubyte", 60000);
    double** train_labels = loadMNISTLabels("../data/train-labels.idx1-ubyte", 60000);
    double** test_images = loadMNISTImages("../data/t10k-images.idx3-ubyte", 10000);
    double** test_labels = loadMNISTLabels("../data/t10k-labels.idx1-ubyte", 10000);

    NeuralNetwork* net = createNetwork();
    train(net, train_images, train_labels, 60000);
    evaluate(net, test_images, test_labels, 10000);

    NeuralNetworkCPU* netCPU = createNetworkCPU();
    trainCPU(netCPU, train_images, train_labels, 60000);
    evaluateCPU(netCPU, test_images, test_labels, 10000);

    return 0;
}