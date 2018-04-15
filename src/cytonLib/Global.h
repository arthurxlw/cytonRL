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

#ifndef _CYTONLIB_GLOBAL_H_
#define _CYTONLIB_GLOBAL_H_

#include "basicHeads.h"
#include "DevMatReal.h"
#include <curand.h>

namespace cytonLib {

extern bool testMode;
extern int blockSize;
extern int blockSize2d;
extern int batchSize;

class Global
{
public:
	ostream* os;
	Precision one;
	Precision zero;
	DevMatReal<float> onesFloat;
	DevMatReal<double> onesDouble;
	DevMatPrec ones_;

	template<typename T>
	T* onesFD(int size);

	Precision* ones(int size);

	Precision initFactor;
	int epoch;
	int batch;
	int hiddenSize;

	cudnnHandle_t cudnnHandle;
	cublasHandle_t cublasHandle;
	curandGenerator_t curandGenerator;
	cusolverDnHandle_t cusolverHandle;
	unsigned long long rnnDropoutSeed;
	bool dropOutRefresh;

	Global();

	void init();

	void end();

	void* workSpace;
	int workSpaceSize;
	void ensureWorkSpace(int size);
};

extern Global global;
extern vector<cudaStream_t> streams;

} /* namespace cytonLib */

#endif /* GLOBAL_H_ */
