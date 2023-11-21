#include"conv2d_reference.hpp"
#include"conv2d_solution.hpp"
#include <vector>

#define ELEMENT_TYPE uint64_t

typedef std::tuple<int, int> Bench;

std::vector<Bench> benches = {
	std::make_tuple(600, 2),
	std::make_tuple(350, 25),
	std::make_tuple(120, 100),
};

extern "C"
void conv2d_reference_c(int64_t **M, int64_t **K, uint32_t m_size, uint32_t k_size, int64_t **R) 
{
//	for(int i = 0; i < ITERATIONS; i++) 
    {
		conv2d_reference(M, K, m_size, k_size, R);
	}
}
FUNCTION(conv2d, conv2d_reference_c);


extern "C"
void conv2d_solution_c(int64_t **M, int64_t **K, uint32_t m_size, uint32_t k_size, int64_t **R) 
{
//	for(int i = 0; i < ITERATIONS; i++) 
    {
		conv2d_solution(M, K, m_size, k_size, R);
	}
}
FUNCTION(conv2d, conv2d_solution_c);
