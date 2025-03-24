#ifdef RISCV_VEXII
#include "sim_stdlib.h"
#else
#include <stdio.h>
#endif

#include "profile.h"
#include "mico_nn.h"
#include "mico_qnn.h"

#include "matmul_test.h"

int main(){
    Tensor2D_Q8 x, w;

    x.data = (int8_t*)malloc(N*K*sizeof(int8_t));
    w.data = (int8_t*)malloc(K*M*sizeof(int8_t));

    x.shape[0] = N;
    x.shape[1] = K;
    
    w.shape[0] = M;
    w.shape[1] = K;

    int32_t *o = (int32_t*)malloc(N*M*sizeof(int32_t));

    long start_time, end_time;

    // Warming Up
    printf("Warming Up\n");
    MiCo_Q8_MatMul(o, &x, &w);
    printf("Warming Up Done\n");

    // Same Bit-widths Kernel
    start_time = MiCo_time();
    MiCo_Q8_MatMul(o, &x, &w);
    end_time = MiCo_time();
    printf("MiCo 8x8 Time: %ld\n", end_time - start_time);

    start_time = MiCo_time();
    MiCo_Q4_MatMul(o, &x, &w);
    end_time = MiCo_time();
    printf("MiCo 4x4 Time: %ld\n", end_time - start_time);

    start_time = MiCo_time();
    MiCo_Q2_MatMul(o, &x, &w);
    end_time = MiCo_time();
    printf("MiCo 2x2 Time: %ld\n", end_time - start_time);

    start_time = MiCo_time();
    MiCo_Q1_MatMul(o, &x, &w);
    end_time = MiCo_time();
    printf("MiCo 1x1 Time: %ld\n", end_time - start_time);


    // Mixed Bit-widths Kernel
    start_time = MiCo_time();
    MiCo_Q8x4_MatMul(o, &x, &w);
    end_time = MiCo_time();
    printf("MiCo 8x4 Time: %ld\n", end_time - start_time);

    start_time = MiCo_time();
    MiCo_Q8x2_MatMul(o, &x, &w);
    end_time = MiCo_time();
    printf("MiCo 8x2 Time: %ld\n", end_time - start_time);

    start_time = MiCo_time();
    MiCo_Q8x1_MatMul(o, &x, &w);
    end_time = MiCo_time();
    printf("MiCo 8x1 Time: %ld\n", end_time - start_time);

    start_time = MiCo_time();
    MiCo_Q4x2_MatMul(o, &x, &w);
    end_time = MiCo_time();
    printf("MiCo 4x2 Time: %ld\n", end_time - start_time);

    start_time = MiCo_time();
    MiCo_Q4x1_MatMul(o, &x, &w);
    end_time = MiCo_time();
    printf("MiCo 4x1 Time: %ld\n", end_time - start_time);

    start_time = MiCo_time();
    MiCo_Q2x1_MatMul(o, &x, &w);
    end_time = MiCo_time();
    printf("MiCo 2x1 Time: %ld\n", end_time - start_time);

    start_time = MiCo_time();
    MiCo_Q4x8_MatMul(o, &x, &w);
    end_time = MiCo_time();
    printf("MiCo 4x8 Time: %ld\n", end_time - start_time);

    start_time = MiCo_time();
    MiCo_Q2x8_MatMul(o, &x, &w);
    end_time = MiCo_time();
    printf("MiCo 2x8 Time: %ld\n", end_time - start_time);

    start_time = MiCo_time();
    MiCo_Q1x8_MatMul(o, &x, &w);
    end_time = MiCo_time();
    printf("MiCo 1x8 Time: %ld\n", end_time - start_time);

    start_time = MiCo_time();
    MiCo_Q2x4_MatMul(o, &x, &w);
    end_time = MiCo_time();
    printf("MiCo 2x4 Time: %ld\n", end_time - start_time);

    start_time = MiCo_time();
    MiCo_Q1x4_MatMul(o, &x, &w);
    end_time = MiCo_time();
    printf("MiCo 1x4 Time: %ld\n", end_time - start_time);

    start_time = MiCo_time();
    MiCo_Q1x2_MatMul(o, &x, &w);
    end_time = MiCo_time();
    printf("MiCo 1x2 Time: %ld\n", end_time - start_time);

    // Floating Point Kernel
    // float *fx, *fw, *fo;

    // fx = (float*)malloc(N*K*sizeof(float));
    // fw = (float*)malloc(K*M*sizeof(float));
    // fo = (float*)malloc(N*M*sizeof(float));

    // start_time = MiCo_time();
    // MiCo_MatMul_f32(fo, fx, fw, N, K, M);
    // end_time = MiCo_time();

    // printf("Floating Point Time: %ld\n", end_time - start_time);
}