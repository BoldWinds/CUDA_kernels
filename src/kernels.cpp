#include "kernels.h"

float launch_sgemm(int version, const GemmArgs& args, bool validate = false){
    float *A, *B, *C;
    CudaTimer timer = new CudaTimer();
    timer.start();
    switch (version)
    {
    case 0:
        launch_gemm_naive();
        break;
    default:
        break;
    }
    float time = timer.stop();
    delete timer;
    return time;
}

float launch_matrix_add(int version, const MatrixAddArgs& args, bool validate = false){
    CudaTimer timer = new CudaTimer();
    timer.start();
    switch (version)
    {
    case 0:
        launch_matrix_add_naive();
        break;
    default:
        break;
    }
    float time = timer.stop();
    delete timer;
    return time;
}