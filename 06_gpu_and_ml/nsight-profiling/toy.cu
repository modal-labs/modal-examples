#include <iostream>

__device__ void nanosleep_cuda(unsigned int nanoseconds) {
    asm volatile ("nanosleep.u32 %0;" :: "r"(nanoseconds));
}

__global__ void toyKernel(int *d_arr) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    d_arr[idx] *= 2;
    nanosleep_cuda(1000000000); // Sleep for ~1 second
}



int main() {
    const int size = 2 << 16;
    const int ct = 2 << 10;
    int h_arr[size], *d_arr;

    for (int i = 0; i < size; i++) h_arr[i] = i;

    cudaMalloc((void **)&d_arr, size * sizeof(int));
    cudaMemcpy(d_arr, h_arr, size * sizeof(int), cudaMemcpyHostToDevice);

    for (int i = 0; i < ct; i++) {
            if (i > 0 && (i & (i - 1)) == 0) {
                std::cout << i << std::endl;
            }
            toyKernel<<<4, 64>>>(d_arr);
            cudaDeviceSynchronize();
        }

    cudaMemcpy(h_arr, d_arr, size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_arr);

    std::cout << "Computation done!" << std::endl;
    return 0;
}
