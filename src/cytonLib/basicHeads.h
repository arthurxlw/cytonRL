/*
Copyright 2018 XIAOLIN WANG (xiaolin.wang@nict.go.jp; arthur.xlw@gmail.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef _CYTONLIB_BASICHEADS_H
#define _CYTONLIB_BASICHEADS_H

#ifdef __CDT_PARSER__
#undef __CUDA_RUNTIME_H__
#include <cuda_runtime.h>
#endif

#include <sstream>
#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <time.h>
#include <cmath>
#include <unordered_map>
#include <fstream>
#include <math_constants.h>
#include "xlCLib.h"
#include <cuda.h>
#include <cudnn.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <curand.h>
#include <device_launch_parameters.h>


#ifndef checkError
#define checkError(stat) { checkError_((stat), __FILE__, __LINE__); }
#endif

#define CU1DBLOCK 256
#define CU2DBLOCK 16

namespace cytonLib
{

typedef unsigned char uchar;
typedef int MatrixIndexT;
typedef cublasOperation_t MatrixTransposeType;

using std::cout;
using std::cerr;
using std::string;
using std::ostringstream;
using std::vector;
using std::unordered_map;
using std::ofstream;
using std::ifstream;
using std::pair;
using std::ostream;
using std::istream;

extern cudnnDataType_t cudnnDataType;
typedef float Precision;

cudaError_t checkError_(cudaError_t stat, const char *file, int line);
cudnnStatus_t checkError_(cudnnStatus_t stat, const char *file, int line);
cublasStatus_t checkError_(cublasStatus_t stat, const char *file, int line);
curandStatus_t checkError_(curandStatus_t stat, const char *file, int line);
cusolverStatus_t checkError_(cusolverStatus_t stat, const char *file, int line);

inline int ceil(int a, int b)
{
	return (a-1)/b+1;
}

void checkFile(ifstream& f, const string t);

inline void setOstream(std::ostream& os)
{
	if(sizeof(Precision)==sizeof(double))
	{
		os << std::setprecision(16)<< std::scientific;
	}
	else if(sizeof(Precision)==sizeof(float))
	{
		os << std::setprecision(8)<< std::scientific;
	}
	else
	{
		assert(false);
	}
}

}


#endif
