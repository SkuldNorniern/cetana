// CUDA wrapper that includes our fixes first
#pragma once

// Include our fixes first
#include "gcc13_compat.h"

// Now include the real CUDA runtime
#include <cuda_runtime.h>
