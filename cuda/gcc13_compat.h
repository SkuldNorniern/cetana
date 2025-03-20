
// Pre-include header to disable GCC 13.3 C23 float types
#define _BITS_FLOATN_H
#define _BITS_FLOATN_COMMON_H
#define __HAVE_FLOAT128 0
#define __HAVE_DISTINCT_FLOAT128 0
#define __HAVE_FLOAT64X 0
#define __HAVE_FLOAT32 0
#define __HAVE_FLOAT32X 0
#define __HAVE_FLOAT16 0

// Define C23 float types as regular types to prevent errors
typedef float _Float32;
typedef double _Float64;
typedef double _Float32x;
typedef double _Float64x;
typedef double _Float128;
