#include <cstdlib>
#include "archlab.h"
#include <unistd.h>
#include <algorithm>
#include <cstdint>
#include "function_map.hpp"
#include <dlfcn.h>
#include "tensor_t.hpp"
#include "perfstats.h"
#include <omp.h>

#define ELEMENT_TYPE uint64_t
uint64_t  canary_size = 500000000;
uint64_t histogram[256];
extern "C"
void  __attribute__ ((optimize("O3"))) _canary(tensor_t<ELEMENT_TYPE> & dst, const tensor_t<ELEMENT_TYPE> & A, uint32_t power,
					       int64_t p1=0,
					       int64_t p2=0,
					       int64_t p3=0,
					       int64_t p4=0,
					       int64_t p5=0) {
	uint64_t arg1 = 1024;
	uint64_t * data = new uint64_t[canary_size];
	uint64_t histogram[256];
			    
	for(int i =0; i < 256;i++) {
		histogram[i] = 0;
	}

#pragma omp parallel for
	for(uint64_t ii = 0; ii < canary_size; ii+=arg1) {
		uint64_t my_histogram[256];
		for(int i =0; i < 256;i++) {
			my_histogram[i] = 0;
		}
		for(uint64_t i = ii; i < canary_size && i < ii + arg1; i++) {
			
			for(int k = 0; k < 64; k+=8) {
				uint8_t b = (data[i] >> k)& 0xff;
				my_histogram[b]++;
			}
		}

#pragma omp critical 
		for(int i =0; i < 256;i++) {
			histogram[i] += my_histogram[i];
		}
	}


	delete data;
}
FUNCTION(conv2d, _canary);

uint array_size;

typedef void(*conv2d_impl)(tensor_t<ELEMENT_TYPE> & , const tensor_t<ELEMENT_TYPE> &, uint32_t power, uint64_t seed, int iterations);

int main(int argc, char *argv[])
{

	
	std::vector<int> mhz_s;
	std::vector<int> default_mhz;
	std::vector<int> powers;
	std::vector<int> default_powers;
	int i, reps=1, size, iterations=1,mhz, power, verify =0, arg=0;
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
	default_sizes.push_back(16);
	default_powers.push_back(4);
	default_threads.push_back(1);
	
	double minv = -1.0;
	double maxv = 1.0;
	std::vector<uint64_t> seeds;
	std::vector<uint64_t> default_seeds;
	default_seeds.push_back(0xDEADBEEF);
	if (canary_size!= 0) {
		functions.insert(functions.begin(),"_canary");
	}
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
                case 'p':
                    for(;i+1<argc;i++)
                    {
                        if(argv[i+1][0]!='-')
                        {
                            power = atoi(argv[i+1]);
	                        powers.push_back(power);
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
                    std::cout << "-s set the size of the matrix to multiply.\n-p set the power to raise it to.\n-f what functions to run.\n-d sets the random seed.\n-o sets where statistics should go.\n-i sets the number of iterations.\n-v compares the result with the reference solution.\n";
                    break;
                case 'v':
		            verify = 1;
                    break;
		case '-':
		    if(strcmp(&argv[i][2], "threads") == 0)
		    {
		        for(;i+1<argc;i++)
		        {
			    if(argv[i+1][0]!='-')
			    {
		                arg = atoi(argv[i+1]);
	                        threads.push_back(arg);
		            }
			    else
			        break;
		        }
		    }
		    else if(strcmp(&argv[i][2], "canary") == 0)
		    {
			    if(argv[i+1][0]!='-')
			    {
		                canary_size = atoi(argv[i+1]);
		            }
			    else
			        break;
		    }
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
	if(threads.size()==0)
	    threads = default_threads;
	if(powers.size()==0)
	    powers = default_powers;
	if(verify == 1)
            sprintf(header,"size,threads,power,function,IC,Cycles,CPI,CT,ET,L1_dcache_miss_rate,L1_dcache_misses,L1_dcache_accesses,branches,branch_misses,correctness");
	else
	   sprintf(header,"size,threads,power,function,IC,Cycles,CPI,CT,ET,L1_dcache_miss_rate,L1_dcache_misses,L1_dcache_accesses,branches,branch_misses");
    perfstats_print_header(stat_file, header);
    change_cpufrequnecy(mhz);
    for(auto & thread: threads ) {
        omp_set_num_threads(thread);
     
	for(auto & seed: seeds ) {
		for(auto & size:sizes) {
			for(auto & power: powers ) {
				for(auto & function : functions) {
					tensor_t<ELEMENT_TYPE> dst(size,size);
					tensor_t<ELEMENT_TYPE> A(size,size);
					randomize(A, seed, minv, maxv);
							
					std::cerr << "Running: " << function << "\n";
					function_spec_t f_spec = function_map::get()[function];
					auto fut = reinterpret_cast<conv2d_impl>(f_spec.second);
					sprintf(preamble, "%lu,%lu,%d,%s,",size,thread,power,function.c_str());
					perfstats_init();
					perfstats_enable(1);
					fut(dst, A, power, seed, iterations);
					perfstats_disable(1);
					if(verify)
					{
						tensor_t<ELEMENT_TYPE>::diff_prints_deltas = true;
						if(function.find("bench_solution") != std::string::npos)
						{
							function_spec_t t = function_map::get()[std::string("bench_reference")];
							auto verify_fut = reinterpret_cast<conv2d_impl>(t.second);
							tensor_t<ELEMENT_TYPE> v(size,size);
							verify_fut(v, A, power, seed, 1);
							if(v == dst)
							{
								std::cerr << "Passed!\n";
								sprintf(epilogue,",1\n");
							}
							else
							{
								std::cerr << "Check:\n" << A << "\nRAISED TO THE " << power << " SHOULD BE  \n" << v << "\nYOUR CODE GOT\n" << dst<< "\n";
								sprintf(epilogue,",-1\n");
							}
							//std::cerr << diff(v,A);
						}
						
						else if(function.find("conv2d_solution_c") != std::string::npos)
						{
							function_spec_t t = function_map::get()[std::string("conv2d_reference_c")];
							auto verify_fut = reinterpret_cast<conv2d_impl>(t.second);
							tensor_t<ELEMENT_TYPE> v(size,size);							
							verify_fut(v, A, power, seed, 1);
							if(v == dst)
							{
								std::cerr << "Passed!!\n";
								sprintf(epilogue,",1\n");
							}
							else
							{
								std::cerr << diff(v,dst);
							//	std::cerr << "Check:\n" << A << "\nRAISED TO THE " << power << " SHOULD BE  \n" << v << "\nYOUR CODE GOT\n" << dst<< "\n";
								sprintf(epilogue,",-1\n");
							}
							//std::cerr << diff(v,A);
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
