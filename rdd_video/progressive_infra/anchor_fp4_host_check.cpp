// CPU-only CUDA-header conversion check. No CUDA runtime/driver API calls.
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cstdio>

int main() {
  const float x[] = {0.f,-0.f,.5f,-.5f,1.f,1.5f,2.f,3.f,4.f,6.f,
                    .25f,.75f,1.25f,1.75f,2.5f,3.5f,5.f,-100.f,100.f};
  for (float v : x)
    std::printf("%u ", (unsigned)__nv_cvt_float_to_fp4(v,__NV_E2M1,cudaRoundNearest));
  std::printf("\n");
  const float scales[] = {0.f,1.f/512,1.f,448.f};
  for (float v : scales)
    std::printf("%u ", (unsigned)__nv_cvt_float_to_fp8(v,__NV_SATFINITE,__NV_E4M3));
  std::printf("\n");
  return 0;
}
