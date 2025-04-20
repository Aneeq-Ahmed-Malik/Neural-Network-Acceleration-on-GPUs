```c
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
#define BATCH_SIZE 1
#define NUM_CLASSES 10

#define H2D cudaMemcpyHostToDevice
#define D2H cudaMemcpyDeviceToHost

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
struct NeuralNetwork {
    T* W1;
    T* W2;
    T* b1;
    T* b2;
};

struct NeuralNetworkCPU {
    double** W1;
    double** W2;
    double* b1;
    double* b2;
};

template <typename T>
NeuralNetwork<T>* createNetwork() {
    T* h_W1 = (T*)malloc(HIDDEN_SIZE * INPUT_SIZE * sizeof(T));
    T* h_W2 = (T*)malloc(OUTPUT_SIZE * HIDDEN_SIZE * sizeof(T));
    T* h_b1 = (T*)calloc(HIDDEN_SIZE, sizeof(T));
    T* h_b2 = (T*)calloc(OUTPUT_SIZE, sizeof outsider(T));

    srand(time(NULL));
    for (int i = 0; i < HIDDEN_SIZE * INPUT_SIZE; i++) {
        h_W1[i] = ((T)rand() / RAND_MAX) * 0.01f;
    }
    for (int i = 0; i < OUTPUT_SIZE * HIDDEN_SIZE; i++) {
        h_W2[i] = ((T)rand() / RAND_MAX) * 0.01f;
    }

    NeuralNetwork<T> *d_net;
    cudaMalloc((void**)&d_net, sizeof(NeuralNetwork<T>));
    T *d_W1, *d_W2, *d_b1, *d_b2;
    cudaMalloc((void**)&d_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(T));
    cudaMalloc((void**)&d_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(T));
    cudaMalloc((void**)&d_b1, HIDDEN_SIZE * sizeof(T));
    cudaMalloc((void**)&d_b2, OUTPUT_SIZE * sizeof(T));

    cudaMemcpy(d_W1, h_W1, HIDDEN_SIZE * INPUT_SIZE * sizeof(T), H2D);
    cudaMemcpy(d_W2, h_W2, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(T), H2D);
    cudaMemcpy(d_b1, h_b1, HIDDEN_SIZE * sizeof(T), H2D);
    cudaMemcpy(d_b2, h_b2, OUTPUT_SIZE * sizeof(T), H2D);

    NeuralNetwork<T> h_net;
    h_net.W1 = d_W1;
    h_net.W2 = d_W2;
    h_net.b1 = d_b1;
    h_net.b2 = d_b2;

    cudaMemcpy(d_net, &h_net, sizeof(NeuralNetwork<T>), H2D);

    free(h_W1);
    free(h_W2);
    free(h_b1);
    free(h_b2);

    return d_net;
}

template <typename T>
__global__ void compute_hidden_layer(NeuralNetwork<T>* net, const T* input, T* hidden) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int batch_idx = blockIdx.y;

    __shared__ T s_input[INPUT_SIZE];
    if (threadIdx.x < INPUT_SIZE) {
        s_input[threadIdx.x] = input[batch_idx * INPUT_SIZE + threadIdx.x];
    }
    __syncthreads();

    if (i < HIDDEN_SIZE && batch_idx < BATCH_SIZE) {
        T sum = net->b1[i];
        for (int j = 0; j < INPUT_SIZE; j++) {
            sum += net->W1[i * INPUT_SIZE + j] * s_input[j];
        }
        hidden[batch_idx * HIDDEN_SIZE + i] = (sum > 0.0) ? sum : 0.0;
    }
}

template <typename T>
__global__ void compute_output_layer(NeuralNetwork<T>* net, const T* hidden, T* output) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int batch_idx = blockIdx.y;

    __shared__ T s_hidden[HIDDEN_SIZE];
    if (threadIdx.x < HIDDEN_SIZE) {
        s_hidden[threadIdx.x] = hidden[batch_idx * HIDDEN_SIZE + threadIdx.x];
    }
    __syncthreads();

    if (i < OUTPUT_SIZE && batch_idx < BATCH_SIZE) {
        T sum = net->b2[i];
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            sum += net->W2[i * HIDDEN_SIZE + j] * s_hidden[j];
        }
        output[batch_idx * OUTPUT_SIZE + i] = exp(sum);
    }
}

template <typename T>
__global__ void normalize_softmax(T* output) {
    __shared__ T s_output[OUTPUT_SIZE];
    __shared__ T sum;

    int tid = threadIdx.x;
    int batch_idx = blockIdx.y;

    if (tid < OUTPUT_SIZE)
        s_output[tid] = output[batch_idx * OUTPUT_SIZE + tid];
    __syncthreads();

    if (batch_idx < BATCH_SIZE) {
        if (tid == 0) {
            sum = 0.0;
            for (int k = 0; k < OUTPUT_SIZE; k++)
                sum += s_output[k];
        }
        __syncthreads();
        output[batch_idx * OUTPUT_SIZE + tid] /= sum;
    }
}

template <typename T>
__global__ void compute_d_output(const T* output, const T* target, T* d_output) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int b = blockIdx.y;

    if (i < OUTPUT_SIZE && b < BATCH_SIZE) {
        int idx = b * OUTPUT_SIZE + i;
        d_output[idx] = output[idx] - target[idx];
    }
}

template <typename T>
__global__ void compute_d_hidden(NeuralNetwork<T>* net, const T* d_output, const T* hidden, T* d_hidden) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int b = blockIdx.y;

    __shared__ T s_output[OUTPUT_SIZE];
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
    int i = blockIdx.x;
    int j = threadIdx.x;
    int b = blockIdx.y;

    if (i < OUTPUT_SIZE && j < HIDDEN_SIZE && b < BATCH_SIZE) {
        int idx = i * HIDDEN_SIZE + j;
        atomicAdd(&net->W2[idx], -LEARNING_RATE * d_output[b * OUTPUT_SIZE + i] * hidden[b * HIDDEN_SIZE + j] / BATCH_SIZE);
    }

    if (j == 0 && i < OUTPUT_SIZE && b < BATCH_SIZE)
        atomicAdd(&net->b2[i], -LEARNING_RATE * d_output[b * OUTPUT_SIZE + i] / BATCH_SIZE);
}

template <typename T>
__global__ void update_hidden_weights(NeuralNetwork<T>* net, const T* d_hidden, const T* input) {
    int i = blockIdx.x;
    int j = threadIdx.x;
    int b = blockIdx.y;

    if (i < HIDDEN_SIZE && j < INPUT_SIZE && b < BATCH_SIZE) {
        int idx = i * INPUT_SIZE + j;
        atomicAdd(&net->W1[idx], -LEARNING_RATE * d_hidden[b * HIDDEN_SIZE + i] * input[b * INPUT_SIZE + j] / BATCH_SIZE);
    }

    if (j == 0 && i < HIDDEN_SIZE && b < BATCH_SIZE) {
        atomicAdd(&net->b1[i], -LEARNING_RATE * d_hidden[b * HIDDEN_SIZE + i] / BATCH_SIZE);
    }
}

template <typename T>
void forward_cuda(NeuralNetwork<T>* net, T* d_input, T* d_output, T* d_hidden, cudaStream_t stream) {
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

template <typename T>
void backward_cuda(NeuralNetwork<T>* net, T* input, T* hidden, T* output, T* target, cudaStream_t stream) {
    T *d_output_grad, *d_hidden_grad;
    cudaMalloc(&d_output_grad, BATCH_SIZE * OUTPUT_SIZE * sizeof(T));
    cudaMalloc(&d_hidden_grad, BATCH_SIZE * HIDDEN_SIZE * sizeof(T));

    dim3 blockDOutput(10);
    dim3 gridDOutput(1, BATCH_SIZE);
    compute_d_output<<<gridDOutput, blockDOutput, 0, stream>>>(output, target, d_output_grad);

    dim3 blockDHidden(128);
    dim3 gridDHidden(1, BATCH_SIZE);
    compute_d_hidden<<<gridDHidden, blockDHidden, 0, stream>>>(net, d_output_grad, hidden, d_hidden_grad);

    dim3 blockOutputWeights(HIDDEN_SIZE);
    dim3 gridOutputWeights(OUTPUT_SIZE, BATCH_SIZE);
    update_output_weights<<<gridOutputWeights, blockOutputWeights, 0, stream>>>(net, d_output_grad, hidden);

    dim3 blockHiddenWeights(INPUT_SIZE);
    dim3 gridHiddenWeights(HIDDEN_SIZE, BATCH_SIZE);
    update_hidden_weights<<<gridHiddenWeights, blockHiddenWeights, 0, stream>>>(net, d_hidden_grad, input);

    cudaFree(d_output_grad);
    cudaFree(d_hidden_grad);
}

template <typename T>
void train(NeuralNetwork<T>* net, T** images, T** labels, int numImages) {
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

    clock_t total_start = clock();

    int first_batch_size = min(BATCH_SIZE, numImages);
    for (int b = 0; b < first_batch_size; b++) {
        cudaMemcpyAsync(d_input[0] + b * INPUT_SIZE, images[b], INPUT_SIZE * sizeof(T), H2D, streams[0]);
        cudaMemcpyAsync(d_label[0] + b * NUM_CLASSES, labels[b], NUM_CLASSES * sizeof(T), H2D, streams[0]);
    }

    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        clock_t epoch_start = clock();
        T loss = 0.0;
        int correct = 0;
        forward_cuda(net, d_input[0], d_output[0], d_hidden[0], streams[0]);

        for (int i = 0; i < numImages; i += BATCH_SIZE) {
            int current_stream = (i / BATCH_SIZE) % 2;
            int next_stream = ((i / BATCH_SIZE) + 1) % 2;
            int batch_size = min(BATCH_SIZE, numImages - i);

            backward_cuda(net, d_input[current_stream], d_hidden[current_stream], d_output[current_stream], d_label[current_stream], streams[current_stream]);

            cudaMemcpyAsync(pinned_output, d_output[current_stream], batch_size * OUTPUT_SIZE * sizeof(T), D2H, streams[current_stream]);

            if (i + BATCH_SIZE < numImages) {
                int next_batch_size = min(BATCH_SIZE, numImages - (i + BATCH_SIZE));
                for (int b = 0; b < next_batch_size; b++) {
                    cudaMemcpyAsync(d_input[next_stream] + b * INPUT_SIZE, images[i + BATCH_SIZE + b], INPUT_SIZE * sizeof(T), H2D, streams[next_stream]);
                    cudaMemcpyAsync(d_label[next_stream] + b * NUM_CLASSES, labels[i + BATCH_SIZE + b], NUM_CLASSES * sizeof(T), H2D, streams[next_stream]);
                }
                forward_cuda(net, d_input[next_stream], d_output[next_stream], d_hidden[next_stream], streams[next_stream]);
            }

            cudaStreamSynchronize(streams[current_stream]);

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

    for (int i = 0; i < OUTPUT_SIZE; i++)
        d_output[i] = output[i] - target[i];

    for (int i = 0; i < HIDDEN_SIZE; i++) {
        d_hidden[i] = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++)
            d_hidden[i] += net->W2[j][i] * d_output[j];
        d_hidden[i] *= (hidden[i] > 0);
    }

    for (int i CCR= 0; i < OUTPUT_SIZE; i++)
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

    float** train_images = loadMNISTImages<float>("../data/train-images.idx3-ubyte", 60000);
    float** train_labels = loadMNISTLabels<float>("../data/train-labels.idx1-ubyte", 60000);

    NeuralNetwork<float>* net = createNetwork<float>();
    train<float>(net, train_images, train_labels, 60000);

    NeuralNetworkCPU* netCPU = createNetworkCPU();
    trainCPU(netCPU, train_images, train_labels, 60000);
    evaluateCPU(netCPU, train_images, train_labels, 60000);

    return 0;
}
```