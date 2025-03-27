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
    Tensor2D_F32 x;
    Tensor2D_Q8 w;
    Tensor1D_F32 bias;

    x.data = (float*)malloc(N*K*sizeof(float));
    w.data = (int8_t*)malloc(K*M*sizeof(int8_t));

    x.shape[0] = N;
    x.shape[1] = K;
    
    w.shape[0] = M;
    w.shape[1] = K;


    bias.data = (float*)malloc(M*sizeof(float));
    bias.shape[0] = M;

    Tensor2D_F32 o;

    o.data = (float*)malloc(N*M*sizeof(float));
    o.shape[0] = N;
    o.shape[1] = M;

    long start_time, end_time;

    // Warming Up
    printf("Warming Up\n");
    MiCo_bitlinear_f32(&o, &x, &w, &bias, 8, 8);
    printf("Warming Up Done\n");

    // Same Bit-widths Kernel
    start_time = MiCo_time();
    MiCo_bitlinear_f32(&o, &x, &w, &bias, 8, 8);
    end_time = MiCo_time();
    printf("MiCo 8x8 Time: %ld\n", end_time - start_time);

    start_time = MiCo_time();
    MiCo_bitlinear_f32(&o, &x, &w, &bias, 4, 4);
    end_time = MiCo_time();
    printf("MiCo 4x4 Time: %ld\n", end_time - start_time);

    start_time = MiCo_time();
    MiCo_bitlinear_f32(&o, &x, &w, &bias, 2, 2);

    end_time = MiCo_time();
    printf("MiCo 2x2 Time: %ld\n", end_time - start_time);

    start_time = MiCo_time();
    MiCo_bitlinear_f32(&o, &x, &w, &bias, 1, 1);

    end_time = MiCo_time();
    printf("MiCo 1x1 Time: %ld\n", end_time - start_time);


    // Mixed Bit-widths Kernel
    start_time = MiCo_time();
    MiCo_bitlinear_f32(&o, &x, &w, &bias, 4, 8);

    end_time = MiCo_time();
    printf("MiCo 8x4 Time: %ld\n", end_time - start_time);

    start_time = MiCo_time();
    MiCo_bitlinear_f32(&o, &x, &w, &bias, 2, 8);

    end_time = MiCo_time();
    printf("MiCo 8x2 Time: %ld\n", end_time - start_time);

    start_time = MiCo_time();
    MiCo_bitlinear_f32(&o, &x, &w, &bias, 1, 8);

    end_time = MiCo_time();
    printf("MiCo 8x1 Time: %ld\n", end_time - start_time);

    start_time = MiCo_time();
    MiCo_bitlinear_f32(&o, &x, &w, &bias, 2, 4);

    end_time = MiCo_time();
    printf("MiCo 4x2 Time: %ld\n", end_time - start_time);

    start_time = MiCo_time();
    MiCo_bitlinear_f32(&o, &x, &w, &bias, 1, 4);

    end_time = MiCo_time();
    printf("MiCo 4x1 Time: %ld\n", end_time - start_time);

    start_time = MiCo_time();
    MiCo_bitlinear_f32(&o, &x, &w, &bias, 1, 2);

    end_time = MiCo_time();
    printf("MiCo 2x1 Time: %ld\n", end_time - start_time);

    start_time = MiCo_time();
    MiCo_bitlinear_f32(&o, &x, &w, &bias, 8, 4);

    end_time = MiCo_time();
    printf("MiCo 4x8 Time: %ld\n", end_time - start_time);

    start_time = MiCo_time();
    MiCo_bitlinear_f32(&o, &x, &w, &bias, 8, 2);

    end_time = MiCo_time();
    printf("MiCo 2x8 Time: %ld\n", end_time - start_time);

    start_time = MiCo_time();
    MiCo_bitlinear_f32(&o, &x, &w, &bias, 8, 1);

    end_time = MiCo_time();
    printf("MiCo 1x8 Time: %ld\n", end_time - start_time);

    start_time = MiCo_time();
    MiCo_bitlinear_f32(&o, &x, &w, &bias, 4, 2);

    end_time = MiCo_time();
    printf("MiCo 2x4 Time: %ld\n", end_time - start_time);

    start_time = MiCo_time();
    MiCo_bitlinear_f32(&o, &x, &w, &bias, 4, 1);

    end_time = MiCo_time();
    printf("MiCo 1x4 Time: %ld\n", end_time - start_time);

    start_time = MiCo_time();
    MiCo_bitlinear_f32(&o, &x, &w, &bias, 2, 1);

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