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



#include "basicHeads.h"
#include "Global.h"


namespace cytonLib
{

cudnnDataType_t cudnnDataType=CUDNN_DATA_FLOAT;

// Define some error checking macros.
cudaError_t checkError_(cudaError_t stat, const char *file, int line)
{
	if (stat != cudaSuccess)
	{
		string tErr=cudaGetErrorString(stat);
		if(tErr!="driver shutting down")
		{
			fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
			assert(false);
			exit(1);
		}
		else
		{
		}
	}
	return stat;
}

cudnnStatus_t checkError_(cudnnStatus_t stat, const char *file, int line)
{
	if (stat != CUDNN_STATUS_SUCCESS)
	{
		fprintf(stderr, "cuDNN Error: %s %s %d\n", cudnnGetErrorString(stat), file, line);
		assert(false);
		exit(1);
	}
	return stat;
}

cublasStatus_t checkError_(cublasStatus_t stat, const char *file, int line)
{
	if (stat != CUBLAS_STATUS_SUCCESS)
	{
		fprintf(stderr, "cublas Error: %d %s %d\n", stat, file, line);
		assert(false);
		exit(1);
	}
	return stat;
}

curandStatus_t checkError_(curandStatus_t stat, const char *file, int line)
{
	if (stat != CURAND_STATUS_SUCCESS)
	{
		fprintf(stderr, "curand Error: %s %d\n",  file, line);
		assert(false);
		exit(1);
	}
	return stat;
}


static const char *_cusolverGetErrorEnum(cusolverStatus_t error)
{
    switch (error)
    {
        case CUSOLVER_STATUS_SUCCESS:
            return "CUSOLVER_SUCCESS";

        case CUSOLVER_STATUS_NOT_INITIALIZED:
            return "CUSOLVER_STATUS_NOT_INITIALIZED";

        case CUSOLVER_STATUS_ALLOC_FAILED:
            return "CUSOLVER_STATUS_ALLOC_FAILED";

        case CUSOLVER_STATUS_INVALID_VALUE:
            return "CUSOLVER_STATUS_INVALID_VALUE";

        case CUSOLVER_STATUS_ARCH_MISMATCH:
            return "CUSOLVER_STATUS_ARCH_MISMATCH";

        case CUSOLVER_STATUS_EXECUTION_FAILED:
            return "CUSOLVER_STATUS_EXECUTION_FAILED";

        case CUSOLVER_STATUS_INTERNAL_ERROR:
            return "CUSOLVER_STATUS_INTERNAL_ERROR";

        case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";

    }

    return "<unknown>";
}

cusolverStatus_t checkError_(cusolverStatus_t stat, const char *file, int line)
{
    if(CUSOLVER_STATUS_SUCCESS != stat) {
        fprintf(stderr, "cusolver error: %s %d, error %d %s\n", file, line,
        		stat, _cusolverGetErrorEnum(stat));
        assert(0);
    }
    return stat;
}

void checkFile(ifstream& f, const string t)
{
	string line;
	while(getline(f, line))
	{
		if(!line.empty())
		{
			break;
		}
	}
	bool right=line==t;
	assert(right);
}

}
