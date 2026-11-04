// Offline prototype only. No distributed launcher is provided.
// Host must validate positive B/C/T/H/W, H/W even, all buffer sizes, packet/map
// bounds, mode 0/1/2, FP32 alpha/sigma in [0,1]. All counts must fit int64.
// Packet must contain finite BF16 values. This raw kernel is NOT a safe API.
// Input packet: BF16 [B,C,unique_parent_voxels]. Output BF16 [B,L,C,4].
// One thread computes two adjacent x positions, sharing one Philox invocation.
#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>
#include "stage_birth_rng.h"

extern "C" __global__ void rdd_birth_patch_bf16(
    const __nv_bfloat16* packet, const int64_t* parent_map,
    __nv_bfloat16* output, int64_t batch, int64_t channels,
    int64_t unique_voxels, int64_t local_tokens, int64_t token_start,
    int64_t tf, int64_t hf, int64_t wf, float alpha, float sigma,
    uint64_t seed, uint32_t stage, uint32_t stream, int arithmetic_mode) {
    int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    const int64_t count=batch*local_tokens*channels*2;
    if(i>=count) return;
    int64_t dy=i%2, c=(i/2)%channels;
    int64_t j=(i/(2*channels))%local_tokens;
    int64_t b=i/(2*channels*local_tokens), token=j+token_start;
    int64_t t=token/((hf/2)*(wf/2));
    int64_t yp=(token/(wf/2))%(hf/2), xp=token%(wf/2);
    uint64_t address=uint64_t(((b*channels+c)*tf+t)*hf*wf+(2*yp+dy)*wf+2*xp);
    uint64_t pair=address/2;
    uint32_t ctr[4]={uint32_t(pair),uint32_t(pair>>32),stage,stream};
    rdd_philox(ctr,uint32_t(seed),uint32_t(seed>>32));
    float u=float(ctr[0])*0x1p-32f+0x1p-32f;
    float v=float(ctr[1])*0x1p-32f+0x1p-32f;
    float radius=sqrtf(-2.f*logf(u)), s, co;
    sincosf(6.2831853071795864769f*v,&s,&co);
    float noise[2]={radius*co,radius*s};
    // The host supplies original scheduler sigma, not 1-rounded-alpha.
    float beta=arithmetic_mode==0 ? 1.f-alpha : sigma;
    // 0: FP32 mix + one BF16 store. 1: explicit BF16 sigma coefficients.
    // 2: FP32 scalar coefficients but BF16 noise/products/add (flow.py BF16).
    if(arithmetic_mode==1) {
        beta=__bfloat162float(__float2bfloat16_rn(beta));
        alpha=__bfloat162float(__float2bfloat16_rn(1.f-beta));
    }
    for(int dx=0;dx<2;++dx) {
        int64_t slot=parent_map[j*4+dy*2+dx];
        float parent=__bfloat162float(packet[(b*channels+c)*unique_voxels+slot]);
        float result;
        if(arithmetic_mode==0) result=alpha*parent+beta*noise[dx];
        else {
            float e=__bfloat162float(__float2bfloat16_rn(noise[dx]));
            float a=__bfloat162float(__float2bfloat16_rn(alpha*parent));
            float bnoise=__bfloat162float(__float2bfloat16_rn(beta*e));
            result=a+bnoise;
        }
        output[((b*local_tokens+j)*channels+c)*4+dy*2+dx]=__float2bfloat16_rn(result);
    }
}
