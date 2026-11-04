#include <cstdio>
#include "stage_birth_rng.h"
int main() {
    uint32_t cases[3][4]={{0,0,0,0},{0xffffffffu,0xffffffffu,0xffffffffu,0xffffffffu},
        {0x243f6a88u,0x85a308d3u,0x13198a2eu,0x03707344u}};
    uint32_t keys[3][2]={{0,0},{0xffffffffu,0xffffffffu},{0xa4093822u,0x299f31d0u}};
    for(int i=0;i<3;++i) {
        rdd_philox(cases[i],keys[i][0],keys[i][1]);
        for(int j=0;j<4;++j) std::printf("%08x%c",cases[i][j],j==3?'\n':' ');
    }
}
