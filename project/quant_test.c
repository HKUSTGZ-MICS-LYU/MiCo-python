#ifdef RISCV_VEXII
#include "sim_stdlib.h"
#else
#include <stdio.h>
#endif

#include "profile.h"
#include "mico_nn.h"
#include "mico_qnn.h"
#include "mico_quant.h"

#define DATA_SIZE 1024

int main(){

    Tensor2D_F32 x;
    Tensor2D_Q8 qx;
    Tensor1D_F32 bias;

    x.data = (float*)malloc(1024*sizeof(float));
    x.shape[0] = 1;
    x.shape[1] = DATA_SIZE;

    qx.data = (int8_t*)malloc(1024*sizeof(int8_t));
    qx.shape[0] = 1;
    qx.shape[1] = DATA_SIZE;
    long start_time, end_time;

    // Warm Up
    printf("Warming Up\n");
    MiCo_2D_FP32toQ8(&qx, &x);

    // 8-bit Quantization
    start_time = MiCo_time();
    MiCo_2D_FP32toQ8(&qx, &x);
    end_time = MiCo_time();
    printf("MiCo 8-bit Quantization Time: %ld\n", end_time - start_time);

    // 4-bit Quantization
    start_time = MiCo_time();
    MiCo_2D_FP32toQ4(&qx, &x);
    end_time = MiCo_time();
    printf("MiCo 4-bit Quantization Time: %ld\n", end_time - start_time);

    // 2-bit Quantization
    start_time = MiCo_time();
    MiCo_2D_FP32toQ2(&qx, &x);
    end_time = MiCo_time();
    printf("MiCo 2-bit Quantization Time: %ld\n", end_time - start_time);
    
    // 1-bit Quantization
    start_time = MiCo_time();
    MiCo_2D_FP32toQ1(&qx, &x);
    end_time = MiCo_time();
    printf("MiCo 1-bit Quantization Time: %ld\n", end_time - start_time);

    return 0;
}