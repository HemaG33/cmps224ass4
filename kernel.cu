#include "common.h"

#include "timer.h"

#define IN_TILE_DIM 32
#define OUT_TILE_DIM ((IN_TILE_DIM) - 2*(FILTER_RADIUS))

__constant__ float filter_c[FILTER_DIM][FILTER_DIM];

__global__ void convolution_tiled_kernel(float* input, float* output, unsigned int width, unsigned int height) {

    __shared__ float inputTile[IN_TILE_DIM + FILTER_DIM - 1][IN_TILE_DIM + FILTER_DIM - 1];

    int row_o = blockIdx.y * OUT_TILE_DIM + threadIdx.y;
    int col_o = blockIdx.x * OUT_TILE_DIM + threadIdx.x;
    int row_i = row_o - FILTER_RADIUS;
    int col_i = col_o - FILTER_RADIUS;

    // Load input tile to shared memory
    if (row_i >= 0 && row_i < height && col_i >= 0 && col_i < width) {
        inputTile[threadIdx.y][threadIdx.x] = input[row_i * width + col_i];
    } else {
        inputTile[threadIdx.y][threadIdx.x] = 0.0f;  // Boundary condition
    }

    __syncthreads();  // Synchronize threads to make sure all data is loaded into shared memory


if (threadIdx.y < OUT_TILE_DIM && threadIdx.x < OUT_TILE_DIM) {
    float sum = 0.0f;
    for (int filterRow = 0; filterRow < FILTER_DIM; ++filterRow) {
        for (int filterCol = 0; filterCol < FILTER_DIM; ++filterCol) {
                sum += filter_c[filterRow][filterCol] * inputTile[threadIdx.y + filterRow][threadIdx.x + filterCol];
 
        }
    }
    if (row_o < height && col_o < width) {
        output[row_o * width + col_o] = sum;
    }
}
}

void copyFilterToGPU(float filter[][FILTER_DIM]) {
    // Copy filter to constant memory
    cudaMemcpyToSymbol(filter_c, filter, FILTER_DIM * FILTER_DIM * sizeof(float));
}

void convolution_tiled_gpu(float* input_d, float* output_d, unsigned int width, unsigned int height) {
    // Call kernel
    dim3 numThreadsPerBlock(IN_TILE_DIM, IN_TILE_DIM);
    dim3 numBlocks((width + IN_TILE_DIM - 1) / IN_TILE_DIM, (height + IN_TILE_DIM - 1) / IN_TILE_DIM);
    convolution_tiled_kernel<<<numBlocks, numThreadsPerBlock>>>(input_d, output_d, width, height);
}

