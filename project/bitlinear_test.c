#ifdef RISCV_VEXII
#include "sim_stdlib.h"
#else
#include <stdio.h>
#endif

#include "profile.h"
#include "mico_nn.h"
#include "mico_qnn.h"

#include "bitlinear_test.h"

extern long QMATMUL_TIMER;
extern long QUANT_TIMER;

int main(){
    Tensor2D_F32 x;
    Tensor2D_Q8 w;
    Tensor1D_F32 bias;

    x.data = (float*)malloc(N*K*sizeof(float));
    w.data = (int8_t*)malloc(K*M*sizeof(int8_t));

    x.shape[0] = N;
    x.shape[1] = K;

    for (int i=0; i<N*K; i++){
        x.data[i] = (float)(i % 256) / 256.0f;
    }
    
    w.shape[0] = M;
    w.shape[1] = K;
    w.scale = 0.5;

    bias.data = (float*)malloc(M*sizeof(float));
    bias.shape[0] = M;

    Tensor2D_F32 o;

    o.data = (float*)malloc(N*M*sizeof(float));
    o.shape[0] = N;
    o.shape[1] = M;

    long start_time, end_time;

    // Warming Up
    printf("Warming Up\n");
    MiCo_bitlinear_f32(&o, &x, &w, &bias, 8, 8, 1);
    printf("Warming Up Done\n");

    // Same Bit-widths Kernel
    for (qtype q=8; q>=1; q /= 2){

        QMATMUL_TIMER = 0;
        QUANT_TIMER = 0;
        
        start_time = MiCo_time();
        MiCo_bitlinear_f32(&o, &x, &w, &bias, q, q, 1);
        end_time = MiCo_time();
        printf("MiCo %dx%d Time: %ld\n", q, q, end_time - start_time);
        printf("QMatMul Timer: %ld\n", QMATMUL_TIMER);
        printf("Quant Timer: %ld\n", QUANT_TIMER);
    }
    // Mixed Bit-widths Kernel
    for (qtype qa=8; qa>=1; qa /= 2){
        for (qtype qb=8; qb>=1; qb /= 2){
            if (qa == qb) continue;

            QMATMUL_TIMER = 0;
            QUANT_TIMER = 0;

            start_time = MiCo_time();
            MiCo_bitlinear_f32(&o, &x, &w, &bias, qb, qa, 1);
            end_time = MiCo_time();
            printf("MiCo %dx%d Time: %ld\n", qa, qb, end_time - start_time);
            printf("QMatMul Timer: %ld\n", QMATMUL_TIMER);
            printf("Quant Timer: %ld\n", QUANT_TIMER);
        }
    }

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