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

#include "Global.h"

namespace cytonLib
{

Global::Global()
{
	os=NULL;
	one=1.0;
	zero=0.0;
	batch=0;

	initFactor=0.1;

	workSpace=NULL;
	workSpaceSize=0;
}

void Global::ensureWorkSpace(int size)
{
	if(workSpaceSize<size)
	{
		if(size!=0)
		{
			checkError(cudaFree(workSpace));
		}
		workSpaceSize=size;
		checkError(cudaMalloc(&workSpace,workSpaceSize) );
	}
}


bool testMode=false;
int batchSize=64;
int blockSize=128;
int blockSize2d=16;
bool random_=false;

void Global::init()
{
	if(os==NULL)
	{
		os=&std::cout;
	}

	cudaSetDevice(0);
	cudaDeviceSynchronize();
	cudaThreadSynchronize();

	cudnnStatus_t tCudnn = cudnnCreate(&cudnnHandle);
	checkError(tCudnn);

	cublasStatus_t tCublas = cublasCreate(&cublasHandle);
	checkError(tCublas);

	curandStatus_t tCurand=curandCreateGenerator(&curandGenerator, CURAND_RNG_PSEUDO_DEFAULT);
	checkError(tCurand);

	srand (time(NULL));
	unsigned long long t=rand();

	t=rand();
	rnnDropoutSeed = t;

	t=rand();
	checkError(curandSetPseudoRandomGeneratorSeed(curandGenerator, t));
}

template<>
double* Global::onesFD(int size)
{
	DevMatReal<double>* res=&onesDouble;

	if(res->length()<size)
	{
		res->resize(size, 1);
		res->setValue(1.0);
	}
	return res->data;
}

template<>
float* Global::onesFD(int size)
{
	DevMatReal<float>* res=&onesFloat;

	if(res->length()<size)
	{
		res->resize(size, 1);
		res->setValue(1.0);
	}
	return res->data;
}

Precision* Global::ones(int size)
{
	DevMatPrec* res=&ones_;
	if(res->length()<size)
	{
		res->resize(size, 1);
		res->setValue(1.0);
	}
	return res->data;
}

void Global::end()
{
	ones_.freeData();
}



Global global;


} /* namespace cytonLib */
