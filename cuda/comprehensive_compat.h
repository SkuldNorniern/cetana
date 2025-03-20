// Comprehensive header to fix GCC 13 compatibility issues
#pragma once

// Define internal macro to inform the system that we're handling the float types
#define __HAVE_FLOAT128 0
#define __HAVE_DISTINCT_FLOAT128 0
#define __HAVE_FLOAT64X 0
#define __HAVE_FLOAT64X_LONG_DOUBLE 0
#define __HAVE_FLOAT32 0
#define __HAVE_FLOAT32X 0
#define __HAVE_FLOAT16 0

// Define standard math features to avoid compiler warnings
#define _ISOC99_SOURCE 1
#define _POSIX_C_SOURCE 200809L
#define _XOPEN_SOURCE 700

// Define the float types that GCC 13 expects
typedef float _Float32;
typedef double _Float64;
typedef double _Float32x;
typedef double _Float64x;
typedef double _Float128;
