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


#include "utils.h"

namespace cytonLib
{

void writeBinaryTag(FILE* file)
{
	int tag=1;
	fwrite(&tag, sizeof(tag), 1, file);

}

void readBinaryTag(FILE* file)
{
	int tag;
	fread(&tag, sizeof(tag), 1, file);
	assert(tag==1);
}


__global__
void applyMask_kernel(int* mask, Precision* mat, int dim2, int dim1, int dim0,
		bool transpose, Precision value)
{
	size_t i2=blockIdx.y;
	size_t i1=blockIdx.x;
	size_t i0=threadIdx.x;

	int *tMask=mask+i2*dim1+i1;
	if(*tMask==0)
	{
		Precision* tm;
		if(transpose==false)
		{
			tm=mat+i2*dim1*dim0+i1*dim0+i0;
		}
		else
		{
			tm=mat+i1*dim2*dim0+i2*dim0+i0;
		}
		*tm=value;
	}
}

__global__
void applyMask_kernel(int* mask, Precision* mat, int len, Precision value)
{
	int i=blockIdx.x*blockDim.x+threadIdx.x;
	if(i<len)
	{
		if(mask[i]==0)
		{
			mat[i]=value;
		}
	}
}



void applyMask(int* mask, Precision* mat, int dim2, int dim1, int dim0,
		bool transpose, Precision value)
{
	dim3 grid(dim1,dim2);
	applyMask_kernel<<<grid, dim0>>>(mask, mat, dim2, dim1, dim0, transpose, value);
}


void applyMask(int* mask, Precision* mat, int len, Precision value)
{
	int gridDim=ceil(len, blockSize);
	applyMask_kernel<<<gridDim, blockSize>>>(mask, mat, len, value);
}

}


