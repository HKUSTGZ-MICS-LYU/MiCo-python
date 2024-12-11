#include "model.h"

#include <stdio.h>

int main(){

    Model model;

    printf("Init Model\n");
    model_init(&model);

    printf("Forward Model\n");
    model_forward(&model);

    printf("End Forward Model\n");
    return 0;
}