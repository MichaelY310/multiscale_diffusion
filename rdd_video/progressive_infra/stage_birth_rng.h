#pragma once
#include <stdint.h>
#ifdef __CUDACC__
#define RDD_HD __host__ __device__
#else
#define RDD_HD
#endif

// Independently implemented Philox4x32-10. Integer KATs: Random123 tests.
RDD_HD inline void rdd_philox(uint32_t c[4], uint32_t k0, uint32_t k1) {
    for (int r=0;r<10;++r) {
        uint64_t a=uint64_t(0xD2511F53u)*c[0];
        uint64_t b=uint64_t(0xCD9E8D57u)*c[2];
        uint32_t n0=uint32_t(b>>32)^c[1]^k0;
        uint32_t n2=uint32_t(a>>32)^c[3]^k1;
        c[0]=n0; c[1]=uint32_t(b); c[2]=n2; c[3]=uint32_t(a);
        k0+=0x9E3779B9u; k1+=0xBB67AE85u;
    }
}
