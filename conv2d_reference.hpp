#ifndef conv2d_REFERENCE_INCLUDED
#define conv2d_REFERENCE_INCLUDED
#include <cstdlib>
#include "archlab.h"
#include <unistd.h>
#include<cstdint>
#include"function_map.hpp"


//template<typename T>
void __attribute__((noinline)) conv2d_reference(int64_t **M, int64_t **K, uint32_t m_size, uint32_t k_size, int64_t **R) {
		      // parameters you can use for whatever purpose you want (e.g., tile sizes)

	for(int32_t i = 0; i < m_size-k_size; i++) {
		for(int32_t j = 0; j < m_size-k_size; j++) {
            int64_t sum=0;
            for(int32_t x = 0; x < k_size; x++) {
                for(int32_t y = 0; y < k_size; y++) {
                    sum += M[i+x][j+y] * K[x][y];
                }
            }
            R[i][j] = sum;
        }
	}
    return;

}

#endif
