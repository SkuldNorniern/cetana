
#include <cuda_runtime.h>
extern "C" {
    int cudaInit(unsigned int flags) {
        return cudaSetDevice(0);
    }
}
