#ifdef RISCV_VEXII
#include "sim_stdlib.h"
#else
#include <stdio.h>
#endif

#include "profile.h"
#include "mico_nn.h"
#include "mico_qnn.h"

#include "bitconv2d_test.h"

int main(){
    Tensor4D_F32 x;
    Tensor4D_Q8 w;
    Tensor1D_F32 bias;

    x.data = (float*)malloc(N*INC*INH*INW*sizeof(float));
    w.data = (int8_t*)malloc(M*INC*K*K*sizeof(int8_t));

    x.shape[0] = N;
    x.shape[1] = INC;
    x.shape[2] = INH;
    x.shape[3] = INW;

    for (int i=0; i<N*INC*INH*INW; i++){
        x.data[i] = (float)(i % 256) / 256.0f;
    }

    w.shape[0] = M;
    w.shape[1] = INC;
    w.shape[2] = K;
    w.shape[3] = K;
    w.scale = 0.5;

    bias.data = (float*)malloc(M*sizeof(float));
    bias.shape[0] = M;

    Tensor4D_F32 o;

    size_t out_h = (INH - K) + 1;
    size_t out_w = (INW - K) + 1;

    o.data = (float*)malloc(N*M*out_h*out_w*sizeof(float));
    o.shape[0] = N;
    o.shape[1] = M;
    o.shape[2] = out_h;
    o.shape[3] = out_w;

    long start_time, end_time;

    // Warming Up
    printf("Warming Up\n");
    MiCo_bitconv2d_f32(&o, &x, &w, &bias, 8, 8, 1, 0, 1, 1, 1);
    printf("Warming Up Done\n");

    // Same Bit-widths Kernel
    for (qtype q=8; q>=1; q /= 2){

        start_time = MiCo_time();
        MiCo_bitconv2d_f32(&o, &x, &w, &bias, q, q, 1, 0, 1, 1, 1);
        end_time = MiCo_time();

        printf("MiCo %dx%d Time: %ld\n", q, q, end_time - start_time);
    }

    // Mixed Bit-widths Kernel
    for (qtype qa=8; qa>=1; qa /= 2){
        for (qtype qb=8; qb>=1; qb /= 2){
            if (qa == qb) continue;

            start_time = MiCo_time();
            MiCo_bitconv2d_f32(&o, &x, &w, &bias, qb, qa, 1, 0, 1, 1, 1);
            end_time = MiCo_time();
            printf("MiCo %dx%d Time: %ld\n", qa, qb, end_time - start_time);
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