#ifdef RISCV_VEXII
#include "sim_stdlib.h"
#endif

#ifndef REF
#define REF
#endif

#include "profile.h"
#include "mico_nn.h"
#include "mico_qnn.h"

#define N 4
#define M 16
#define K 256

int check(int32_t *o_ref, int32_t *o_cmp){
    int32_t error = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            if(o_ref[i * M + j] != o_cmp[i * M + j]){
                error++;
                printf("Error at (%d, %d): %d != %d\n", i, j, o_ref[i * M + j], o_cmp[i * M + j]);
            }
        }
    }
    return error;
}

int test_func(int32_t* o_ref, int32_t* o_cmp, Tensor2D_Q8 x, Tensor2D_Q8 w, 
    void (*func_ref)(int32_t*, const Tensor2D_Q8*, const Tensor2D_Q8*), 
    void (*func_cmp)(int32_t*, const Tensor2D_Q8*, const Tensor2D_Q8*)){
    
    long start, end;
    start = MiCo_time();
    func_ref(o_ref, &x, &w);
    end = MiCo_time();
    long ref_time = end - start;
    printf("Reference time: %ld cycles\n", ref_time);
    start = MiCo_time();
    func_cmp(o_cmp, &x, &w);
    end = MiCo_time();
    printf("Comparison time: %ld cycles\n", end - start);
    long cmp_time = end - start;
    int32_t errors = check(o_ref, o_cmp);
    if(errors == 0){
        printf("Test passed!\n");
        printf("Speedup: %fx\n", (float)ref_time / (float)cmp_time);
        return 0;
    } else {
        printf("Test failed with %d errors!\n", errors);
        return -1;
    }
}


int main(){
    Tensor2D_Q8 x, w;

    size_t alignment = 32; // N-byte alignment for SIMD

    void *x_raw = malloc(N*K*sizeof(int8_t) + alignment - 1);
    void *w_raw = malloc(K*M*sizeof(int8_t) + alignment - 1);

    x.data = (int8_t*)(((uintptr_t)x_raw + alignment - 1) & ~(alignment - 1));
    w.data = (int8_t*)(((uintptr_t)w_raw + alignment - 1) & ~(alignment - 1));

    // Initialize the data with some values
    for (size_t i = 0; i < N * K; i++) {
        x.data[i] = (int8_t)(i % 107); // Example initialization
        // x.data[i] = (int8_t)(i % 8);
        // x.data[i] = 1;
    }
    for (size_t i = 0; i < K * M; i++) {
        w.data[i] = (int8_t)(176 - (i % 177)); // Example initialization
        // w.data[i] = (int8_t)(7 - i % 8);
        // w.data[i] = 1;
    }

    x.shape[0] = N;
    x.shape[1] = K;
    
    w.shape[0] = M;
    w.shape[1] = K;

    int32_t *o_ref = (int32_t*)malloc(N*M*sizeof(int32_t));
    int32_t *o_cmp = (int32_t*)malloc(N*M*sizeof(int32_t));

    // Test List
    void* func_refs[] = {
        MiCo_Q8_MatMul_Ref, 
        MiCo_Q4_MatMul_Ref, 
        MiCo_Q2_MatMul_Ref, 
        MiCo_Q1_MatMul_Ref,
        MiCo_Q8x4_MatMul_Ref,
        MiCo_Q8x2_MatMul_Ref,
        MiCo_Q8x1_MatMul_Ref,
        MiCo_Q4x2_MatMul_Ref,
        MiCo_Q4x1_MatMul_Ref,
        MiCo_Q2x1_MatMul_Ref,
        MiCo_Q4x8_MatMul_Ref,
        MiCo_Q2x8_MatMul_Ref,
        MiCo_Q2x4_MatMul_Ref,
        MiCo_Q1x8_MatMul_Ref,
        MiCo_Q1x4_MatMul_Ref,
        MiCo_Q1x2_MatMul_Ref,
    };

    void* func_cmps[] = {
        MiCo_Q8_MatMul, 
        MiCo_Q4_MatMul, 
        MiCo_Q2_MatMul, 
        MiCo_Q1_MatMul,
        MiCo_Q8x4_MatMul,
        MiCo_Q8x2_MatMul,
        MiCo_Q8x1_MatMul,
        MiCo_Q4x2_MatMul,
        MiCo_Q4x1_MatMul,
        MiCo_Q2x1_MatMul,
        MiCo_Q4x8_MatMul,
        MiCo_Q2x8_MatMul,
        MiCo_Q2x4_MatMul,
        MiCo_Q1x8_MatMul,
        MiCo_Q1x4_MatMul,
        MiCo_Q1x2_MatMul,
    };
    const char* func_names[] = {
        "MiCo_Q8_MatMul", 
        "MiCo_Q4_MatMul", 
        "MiCo_Q2_MatMul", 
        "MiCo_Q1_MatMul",
        "MiCo_Q8x4_MatMul",
        "MiCo_Q8x2_MatMul",
        "MiCo_Q8x1_MatMul",
        "MiCo_Q4x2_MatMul",
        "MiCo_Q4x1_MatMul",
        "MiCo_Q2x1_MatMul",
        "MiCo_Q4x8_MatMul",
        "MiCo_Q2x8_MatMul",
        "MiCo_Q2x4_MatMul",
        "MiCo_Q1x8_MatMul",
        "MiCo_Q1x4_MatMul",
        "MiCo_Q1x2_MatMul",
    };
    int num_tests = sizeof(func_refs) / sizeof(func_refs[0]);
    for(int i = 0; i < 1; i++){
        printf("Testing %s...\n", func_names[i]);
        int result = test_func(o_ref, o_cmp, x, w, 
            (void (*)(int32_t*, const Tensor2D_Q8*, const Tensor2D_Q8*))func_refs[i],
            (void (*)(int32_t*, const Tensor2D_Q8*, const Tensor2D_Q8*))func_cmps[i]);
        if(result != 0){
            printf("%s test failed!\n", func_names[i]);
            free(x_raw);
            free(w_raw);
            free(o_ref);
            free(o_cmp);
            return -1;
        }
    }
    return 0;
}