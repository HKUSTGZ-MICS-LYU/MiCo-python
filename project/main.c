#include <stdio.h>

#include "model.h"

// #include "mlp_test_data.h"
// #include "lenet_test_mnist.h"
#include "test_cifar10.h"

int main(){

    Model model;

    printf("Init Model\n");
    model_init(&model);

    int correct = 0;

    for (int t=0; t < TEST_NUM; t++){

        printf("Set Input Data\n");
        // model.x.data = test_data[t];
        model.x.data = test_input[t];

        printf("Forward Model\n");
        model_forward(&model);

        size_t label[1];
        MiCo_argmax2d_f32(label, &model.output);
        printf("Predicted Label: %ld, Correct Label: %d\n", label[0], test_label[t]);
        if (label[0] == test_label[t]){
            correct++;
        }
    }
    printf("Correct: %d / %d\n", correct, TEST_NUM);
    printf("Accuracy: %f\n", (float)correct/TEST_NUM);
    return 0;
}