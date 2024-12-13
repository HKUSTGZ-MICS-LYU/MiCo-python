#ifdef RISCV
#include "sim_stdlib.h"
#else
#include <stdio.h>
#endif

#include "model.h"
#include "test_cifar10.h"
int main(){

    Model m_ref;
    Model* model_ref = &m_ref;
    model_init(model_ref);

    m_ref.x.data = test_input[0];
    MiCo_conv2d_f32(&model_ref->layers_0, 
        &model_ref->x, 
        &model_ref->layers_0_weight, 
        &model_ref->layers_0_bias, 1, 2, 1, 1);

    Model m_test;
    Model* model_test = &m_test;
    model_init(model_test);

    m_test.x.data = test_input[0];

    MiCo_im2col_conv2d_f32(&model_test->layers_0, 
        &model_test->x, 
        &model_test->layers_0_weight, 
        &model_test->layers_0_bias, 1, 2, 1, 1);

    size_t total = model_ref->layers_0.shape[0]*model_ref->layers_0.shape[1]*model_ref->layers_0.shape[2]*model_ref->layers_0.shape[3];

    for (size_t i=0; i<total; i++){
        float error = model_ref->layers_0.data[i] - model_test->layers_0.data[i];
        error = error > 0 ? error : -error;
        if (error > 1e-5){
            printf("Error: %f, %f\n", 
            model_ref->layers_0.data[i], 
            model_test->layers_0.data[i]);
        }
    }

    return 0;
}