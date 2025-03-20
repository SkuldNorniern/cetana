// Float types definitions for GCC 13 + CUDA
#pragma once

// Define the types that stdlib.h and math.h expect
typedef float _Float32;
typedef double _Float64;
typedef long double _Float128;
typedef double _Float32x;
typedef long double _Float64x;

// Suppress include of problematic system headers
#define _BITS_FLOATN_H
#define _BITS_FLOATN_COMMON_H

// Define additional macros needed by system headers
#define __f32(x) x ## f
#define __f64(x) x
#define __f32x(x) x
#define __f64x(x) x
#define __builtin_huge_valf() __builtin_huge_val()
#define __builtin_nanf("") __builtin_nan("")
#define __builtin_nansf("") __builtin_nans("")
