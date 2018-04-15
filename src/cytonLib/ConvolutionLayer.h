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

#ifndef _CYTONLIB_CONVOLUTIONCELL_H_
#define _CYTONLIB_CONVOLUTIONCELL_H_

#include "Weight.h"
#include "Variable.h"
#include "Layer.h"
#include "WeightFactory.h"

namespace cytonLib
{

class ConvolutionLayer: public Layer
{

public:
	Weight weight;
	Weight bias;

	Variable* init(string tag, Variable* x, int outputs, int kernelDimH, int kernelDimW,
				int strideH=1, int strideW=1,int padH=0, int padW=0, WeightFactory* weightFactory_=NULL);

	void forward(const string& tag, Variable& x, Variable& y);

	void backward(const string& tag, Variable& x, Variable& y);

	void calculateGradient(const string& tag, Variable& x, Variable& y);

	void forward();

	void backward();

	void calculateGradient();

protected:
	cudnnConvolutionFwdAlgo_t forwardAlgo;
	cudnnConvolutionBwdDataAlgo_t backwardDataAlgo;
	cudnnConvolutionBwdFilterAlgo_t backwardFilterAlgo;
	cudnnFilterDescriptor_t weightDesc;
	cudnnTensorDescriptor_t biasDesc;
	cudnnConvolutionDescriptor_t convDesc;

};

} /* namespace cytonLib */

#endif /* CONVOLUTIONCELL_H_ */
