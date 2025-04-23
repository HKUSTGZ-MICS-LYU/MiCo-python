#ifdef RISCV_VEXII
#include "sim_stdlib.h"
#else
#include <stdio.h>
#endif

#include "profile.h"
#include "mico_nn.h"
#include "mico_qnn.h"

#include "avgpool4d_f32_test.h"

int main(){

    Tensor4D_F32 x;
    Tensor4D_F32 y;

    x.data = (float*)malloc(N*INC*INH*INW*sizeof(float));

    size_t out_h = (INH - K) / S + 1;
    size_t out_w = (INW - K) / S + 1;

    y.data = (float*)malloc(N*INC*out_h*out_w*sizeof(float));

    x.shape[0] = N;
    x.shape[1] = INC;
    x.shape[2] = INH;
    x.shape[3] = INW;

    for (int i=0; i<N*INC*INH*INW; i++){
        x.data[i] = (float)(i % 256) / 256.0f;
    }
    
    y.shape[0] = N;
    y.shape[1] = INC;
    y.shape[2] = out_h;
    y.shape[3] = out_w;

    long start_time, end_time;

    // Warming Up
    printf("Warming Up\n");
    MiCo_avgpool4d_f32(&y, &x, K, S);
    printf("Warming Up Done\n");

    start_time = MiCo_time();
    MiCo_avgpool4d_f32(&y, &x, K, S);
    end_time = MiCo_time();
    printf("MiCo Time: %ld\n",end_time - start_time);

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