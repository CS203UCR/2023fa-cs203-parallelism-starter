#include <cstdlib>
#include "archlab.h"
#include <unistd.h>
#include <algorithm>
#include <cstdint>
#include "function_map.hpp"
#include <dlfcn.h>
#include <vector>
#include <sstream>
#include <string>
#include "perfstats.h"
#include <omp.h>

#define ELEMENT_TYPE uint64_t

uint array_size;

typedef void(*conv2d_impl)(int64_t **M, int64_t **K, uint32_t m_size, uint32_t k_size, int64_t **R);

int64_t **init_square_matrix(uint32_t size);
int64_t **zero_square_matrix(uint32_t size);
void free_square_matrix(int64_t **M, uint32_t size);
bool compare_matrices(int64_t **M, int64_t **N, uint32_t size);
int main(int argc, char *argv[])
{

	
	std::vector<int> mhz_s;
	std::vector<int> default_mhz;
	std::vector<int> k_sizes;
	std::vector<int> default_k_sizes;
	int i, reps=1, size, thread_input, iterations=1,mhz, k_size, verify =0;
    char *stat_file = NULL;
    char default_filename[] = "stats.csv";
    char preamble[1024];
    char epilogue[1024];
    char header[1024];
	std::stringstream clocks;
	std::vector<std::string> functions;
	std::vector<std::string> default_functions;
	std::vector<unsigned long int> sizes;
	std::vector<unsigned long int> default_sizes;
    std::vector<unsigned long int> threads;
	std::vector<unsigned long int> default_threads;
	default_threads.push_back(1);
	default_sizes.push_back(1024);
	default_k_sizes.push_back(15);
    int64_t **M, **K, **R, **V;
	
	float minv = -1.0;
	float maxv = 1.0;
	std::vector<uint64_t> seeds;
	std::vector<uint64_t> default_seeds;
	default_seeds.push_back(0xDEADBEEF);
    for(i = 1; i < argc; i++)
    {
            // This is an option.
        if(argv[i][0]=='-')
        {
            switch(argv[i][1])
            {
                case 'o':
                    if(i+1 < argc && argv[i+1][0]!='-')
                        stat_file = argv[i+1];
                    break;
                case 'r':
                    if(i+1 < argc && argv[i+1][0]!='-')
                        reps = atoi(argv[i+1]);
                    break;
                case 's':
                    for(;i+1<argc;i++)
                    {
                        if(argv[i+1][0]!='-')
                        {
                            size = atoi(argv[i+1]);
	                        sizes.push_back(size);
                        }
                        else
                            break;
                    }
                    break;
                case 't':
                    for(;i+1<argc;i++)
                    {
                        if(argv[i+1][0]!='-')
                        {
                            thread_input = atoi(argv[i+1]);
	                        threads.push_back(thread_input);
                        }
                        else
                            break;
                    }
                    break;
                case 'k':
                    for(;i+1<argc;i++)
                    {
                        if(argv[i+1][0]!='-')
                        {
                            k_size = atoi(argv[i+1]);
	                        k_sizes.push_back(k_size);
                        }
                        else
                            break;
                    }
                    break;
                case 'M':
                    for(;i+1<argc;i++)
                    {
                        if(argv[i+1][0]!='-')
                        {
                            mhz = atoi(argv[i+1]);
	                        mhz_s.push_back(mhz);
                        }
                        else
                            break;
                    }
                    break;
                case 'f':
                    for(;i+1<argc;i++)
                    {
                        if(argv[i+1][0]!='-')
                        {
                            functions.push_back(std::string(argv[i+1]));
                        }
                    else
                        break;
                    }
                    break;
                case 'i':
                    if(i+1 < argc && argv[i+1][0]!='-')
                        iterations = atoi(argv[i+1]);
                    break;
                case 'h':
                    std::cout << "-s set the size of the source matrix.\n-k set the kernel matrix size.\n-f what functions to run.\n-d sets the random seed.\n-o sets where statistics should go.\n-i sets the number of iterations.\n-v compares the result with the reference solution.\n";
                    break;
                case 'v':
		            verify = 1;
                    break;
                }
            }
        }
	if(stat_file==NULL)
	    stat_file = default_filename;

	if (std::find(functions.begin(), functions.end(), "ALL") != functions.end()) {
		functions.clear();
		for(auto & f : function_map::get()) {
			functions.push_back(f.first);
		}
	}
	
	for(auto & function : functions) {
		auto t= function_map::get().find(function);
		if (t == function_map::get().end()) {
			std::cerr << "Unknown function: " << function <<"\n";
			exit(1);
		}
		std::cerr << "Gonna run " << function << "\n";
	}
	if(sizes.size()==0)
	    sizes = default_sizes;
	if(seeds.size()==0)
	    seeds = default_seeds;
	if(functions.size()==0)
	    functions = default_functions;
	if(k_sizes.size()==0)
	    k_sizes = default_k_sizes;
	if(verify == 1)
            sprintf(header,"size,k_size,threads,function,IC,Cycles,CPI,CT,ET,L1_dcache_miss_rate,L1_dcache_misses,L1_dcache_accesses,branch_mispred,branches,correctness");
	else
	   sprintf(header,"size,k_size,threads,function,IC,Cycles,CPI,CT,ET,L1_dcache_miss_rate,L1_dcache_misses,L1_dcache_accesses,branch_mispred,branches");
    perfstats_print_header(stat_file, header);
     
	for(auto & seed: seeds ) {
		for(auto & size:sizes) {
			for(auto & k_size: k_sizes ) {
                for(auto & thread: threads ) {
                    omp_set_num_threads(thread);
                    for(auto & function : functions) {
                        M = init_square_matrix(size);
                        K = init_square_matrix(k_size);
                        R = zero_square_matrix(size);
                        std::cerr << "Running: " << function << "\n";
                        function_spec_t f_spec = function_map::get()[function];
                        auto fut = reinterpret_cast<conv2d_impl>(f_spec.second);
                        sprintf(preamble, "%lu,%d,%ld,%s,",size,k_size,thread,function.c_str());
                        perfstats_init();
                        perfstats_enable(1);
                        fut(M, K, size, k_size, R);
                        perfstats_disable(1);
                        if(verify)
                        {
                            if(function.find("bench_solution") != std::string::npos)
                            {
                                function_spec_t t = function_map::get()[std::string("conv2d_reference_c")];
                                auto verify_fut = reinterpret_cast<conv2d_impl>(t.second);
                                V = zero_square_matrix(size);
                                verify_fut(M, K, size, k_size, V);
                                if(compare_matrices(R,V,size))
                                {
                                    std::cerr << "Passed!\n";
                                    sprintf(epilogue,",1\n");
                                }
                                else
                                {
                                    std::cerr << "Reference solution does not agree with your optimization!\n";
                                    sprintf(epilogue,",-1\n");
                                }
                            }

                            else if(function.find("conv2d_solution_c") != std::string::npos)
                            {
                                function_spec_t t = function_map::get()[std::string("conv2d_reference_c")];
                                auto verify_fut = reinterpret_cast<conv2d_impl>(t.second);						
                                V = zero_square_matrix(size);
                                verify_fut(M, K, size, k_size, V);
                                if(compare_matrices(R,V,size))
                                {
                                    std::cerr << "Passed!!\n";
                                    sprintf(epilogue,",1\n");
                                }
                                else
                                {
                                    std::cerr << "Reference solution does not agree with your optimization!\n";
                                    sprintf(epilogue,",-1\n");
                                }
                            }
                            else
                                sprintf(epilogue,",0\n");
                        }
                        else
                            sprintf(epilogue,"\n");
                        perfstats_print(preamble, stat_file, epilogue);
                        perfstats_deinit();
                        std::cerr << "Done execution: " << function << "\n";
                    }
				}
			}
		}
	}
	return 0;
}

//START_INIT
int64_t **init_square_matrix(uint32_t size)
{
    int64_t **M,*data,i,j;
    uint64_t seed;
    data = (int64_t *)malloc(size*size*sizeof(int64_t));
    M = (int64_t **)malloc(size*sizeof(int64_t *));
    for(i=0;i<size;i++)
    {
        M[i] = &data[size*i];
        for(j=0;j<size;j++)
        {
            M[i][j]=fast_rand(&seed);
        }
    }
    return M;
}
//END_INIT

int64_t **zero_square_matrix(uint32_t size)
{
    int64_t **M,*data,i,j;
    data = (int64_t *)calloc(size*size,sizeof(int64_t));
    M = (int64_t **)malloc(size*sizeof(int64_t *));
    for(i=0;i<size;i++)
    {
        M[i] = &data[size*i];
    }
    return M;
}
void free_square_matrix(int64_t **M, uint32_t size)
{
    uint32_t i;
    for(i=0;i<size;i++)
        free(M[i]);
    free(M);
}

bool compare_matrices(int64_t **M, int64_t **N, uint32_t size)
{
    uint32_t i,j;
    for(i=0;i<size;i++)
    {
        for(j=0;j<size;j++)
        {
            if(M[i][j]!=N[i][j])
                return false;
        }
    }
    return true;
}