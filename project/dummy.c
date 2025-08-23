#ifdef RISCV_VEXII
#include "sim_stdlib.h"
#else
#include <stdio.h>
#endif

#include "profile.h"
#include "mico_nn.h"
#include "mico_qnn.h"

#define N 16
#define K 16
#define M 16

int main(){
    
    Tensor2D_Q8 x, w;
    printf("Hello MiCo!\n");

    x.data = (int8_t*)malloc(N*K*sizeof(int8_t));
    w.data = (int8_t*)malloc(K*M*sizeof(int8_t));

    x.shape[0] = N;
    x.shape[1] = K;
    
    w.shape[0] = M;
    w.shape[1] = K;

    int32_t *o = (int32_t*)malloc(N*M*sizeof(int32_t));
    long start_time, end_time;
    printf("MiCo 8x8 MatMul Test\n");
    start_time = MiCo_time();
    MiCo_Q8_MatMul(o, &x, &w);
    end_time = MiCo_time();
    printf("MiCo 8x8 Time: %ld\n", end_time - start_time);

    return 0;
}