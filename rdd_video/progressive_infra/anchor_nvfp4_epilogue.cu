// Experimental CUDA producer: parent + 3 details -> 4 fresh GELUs ->
// 3 high bands -> E4M3/group16 scales + packed E2M1.
// No GEMM, no model integration, no performance claim.
// Logical row-major scales REQUIRE a CUTLASS SM120 layout adapter before MMA.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <stdint.h>

__device__ __forceinline__ float fresh_gelu(float x) {
  return .5f*x*(1.f + erff(x * .7071067811865475f));
}

extern "C" __global__ void anchor_gelu_nvfp4(
    const __nv_bfloat16* parent, const __nv_bfloat16* details,
    uint8_t* packed, uint8_t* logical_scales, uint8_t* clip_counts,
    int* nonfinite_flag, int nc, int g, float global_scale) {
  // One 16-lane subgroup owns all three bands of a parent/channel-block.
  // Parent is loaded ONCE, not once per output band. No four-row temporary.
  const int lane = threadIdx.x % 16;
  const int groups_per_row = (g+15)/16;
  const int64_t group = ((int64_t)blockIdx.x*blockDim.x + threadIdx.x)/16;
  const int p = group/groups_per_row;
  const int kblock = group%groups_per_row;
  const int k = kblock*16 + lane;
  float a=0, b=0, c=0, d=0;
  if (p < nc && k < g) {
    const int64_t start = (int64_t)p*3*g+k;
    a = __bfloat162float(parent[(int64_t)p*g+k]);
    b = __bfloat162float(details[start]);
    c = __bfloat162float(details[start+g]);
    d = __bfloat162float(details[start+2*g]);
    if (!isfinite(a) || !isfinite(b) || !isfinite(c) || !isfinite(d)) {
      atomicExch(nonfinite_flag, 1);
      a=b=c=d=0;  // The caller MUST reject outputs when flag is set.
    }
  }
  const float y0=fresh_gelu(((a+b)+c)+d);
  const float y1=fresh_gelu(((a-b)+c)-d);
  const float y2=fresh_gelu(((a+b)-c)-d);
  const float y3=fresh_gelu(((a-b)-c)+d);
  float h[3] = {(((y0-y1)+y2)-y3)*.25f,
                      (((y0+y1)-y2)-y3)*.25f,
                      (((y0-y1)-y2)+y3)*.25f};
  if (!isfinite(h[0]) || !isfinite(h[1]) || !isfinite(h[2])) {
    atomicExch(nonfinite_flag, 1);
    h[0]=h[1]=h[2]=0;
  }
  const unsigned mask = __activemask();
  for (int band=0; band<3; ++band) {
    float maximum = fabsf(h[band]);
    for (int offset=8; offset>0; offset/=2)
      maximum=fmaxf(maximum,__shfl_xor_sync(mask,maximum,offset,16));
    float raw=maximum == 0 ? 0 : fminf(448.f,fmaxf(1.f/512,(maximum/global_scale)/6));
    const uint8_t sf=__nv_cvt_float_to_fp8(raw,__NV_SATFINITE,__NV_E4M3);
    const float scale=__half2float(__nv_cvt_fp8_to_halfraw(sf,__NV_E4M3))*global_scale;
    const float normalized=scale==0 ? 0 : h[band]/scale;
    const uint8_t code=__nv_cvt_float_to_fp4(normalized,__NV_E2M1,cudaRoundNearest);
    const uint8_t next=__shfl_down_sync(mask,(unsigned)code,1,16);
    unsigned clipped=fabsf(h[band])>6*scale ? 1 : 0;
    for (int offset=8; offset>0; offset/=2)
      clipped+=__shfl_xor_sync(mask,clipped,offset,16);
    if (p < nc) {
      const int64_t row=(int64_t)p*3+band;
      if (lane%2==0)
        packed[row*groups_per_row*8+kblock*8+lane/2]=code | (next<<4);
      if (lane==0) {
        logical_scales[row*groups_per_row+kblock]=sf;
        clip_counts[row*groups_per_row+kblock]=(uint8_t)clipped;
      }
    }
  }
}

// Caller must allocate packed[3Nc,ceil(G/16)*8], scales/clips[3Nc,ceil(G/16)],
// zero nonfinite_flag on THIS stream, and validate it before consuming output.
// Positive finite global_scale must come from charged calibration/amax work.
extern "C" cudaError_t launch_anchor_gelu_nvfp4(
    const void* parent, const void* details, uint8_t* packed,
    uint8_t* scales, uint8_t* clips, int* nonfinite_flag,
    int nc, int g, float global_scale, cudaStream_t stream) {
  if (nc<=0 || g<=0 || g>2147483632 || !isfinite(global_scale) || global_scale<=0 ||
      !parent || !details || !packed || !scales || !clips || !nonfinite_flag)
    return cudaErrorInvalidValue;
  int64_t groups=(int64_t)nc*((g+15)/16);
  if (groups>2147483640LL) return cudaErrorInvalidValue;
  anchor_gelu_nvfp4<<<(groups+7)/8,128,0,stream>>>(
      (const __nv_bfloat16*)parent,(const __nv_bfloat16*)details,
      packed,scales,clips,nonfinite_flag,nc,g,global_scale);
  return cudaGetLastError();
}
