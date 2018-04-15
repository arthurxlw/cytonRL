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
#include "ConvolutionLayer.h"
#include "Global.h"
#include "WeightFactory.h"

namespace cytonLib
{

Variable* ConvolutionLayer::init(string tag_, Variable* x_, int outputs, int kernelDimH, int kernelDimW,
		int strideH, int strideW, int padH, int padW, WeightFactory* weightFactory_)
{
	this->tag=tag_;
	this->x=x_;

	WeightFactory* pWF=weightFactory_;
	if(pWF==NULL)
	{
		pWF=&weightFactory;
	}

	pWF->create(weight, tag+string(".weight"), x->c*kernelDimH*kernelDimW, outputs);
	const int tensorDims = 4;
	const int filterDimA[tensorDims] = {outputs, x->c, kernelDimH, kernelDimW};
	checkError(cudnnCreateFilterDescriptor(&weightDesc));
	checkError(cudnnSetFilterNdDescriptor(weightDesc,
			cudnnDataType,
			CUDNN_TENSOR_NCHW,
			tensorDims,
			filterDimA) );

	pWF->create(bias, tag+string(".bias"), 1, outputs);
	checkError(cudnnCreateTensorDescriptor(&biasDesc));
	checkError(cudnnSetTensor4dDescriptor(biasDesc, CUDNN_TENSOR_NCHW, cudnnDataType, 1, outputs, 1, 1));

	const int convDims = 2;
	int filterStrideA[convDims] = {strideH,strideW};
	int padA[convDims] = {padH,padW};
	int upscaleA[convDims] = {1,1};

	checkError(cudnnCreateConvolutionDescriptor(&convDesc))
	checkError(cudnnSetConvolutionNdDescriptor(convDesc,
			convDims,
			padA,
			filterStrideA,
			upscaleA,
			CUDNN_CROSS_CORRELATION,
			cudnnDataType) );

	int tensorOuputDimA[tensorDims];
	// find dimension of convolution output
	checkError( cudnnGetConvolutionNdForwardOutputDim(convDesc,
			x->desc,
			weightDesc,
			tensorDims,
			tensorOuputDimA) );
	int outputN = tensorOuputDimA[0];
	int outputC = tensorOuputDimA[1];
	int outputH = tensorOuputDimA[2];
	int outputW = tensorOuputDimA[3];

	y.resize(outputN, outputC, outputH, outputW);

	assert(outputC==outputs);

	// Choose the best according to the preference
	checkError( cudnnGetConvolutionForwardAlgorithm(global.cudnnHandle,
			x->desc,
			weightDesc,
			convDesc,
			y.desc,
			CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
			0,
			&forwardAlgo
	) );
	std::cout << "Fastest forward algorithm is Algo " << forwardAlgo << "\n";

	size_t sizeInBytes=0;
	checkError( cudnnGetConvolutionForwardWorkspaceSize(global.cudnnHandle,
			x->desc,
			weightDesc,
			convDesc,
			y.desc,
			forwardAlgo,
			&sizeInBytes) );
	global.ensureWorkSpace(sizeInBytes);

	//backward data
	checkError( cudnnGetConvolutionBackwardDataAlgorithm(global.cudnnHandle,
			weightDesc,
			y.desc,
			convDesc,
			x->desc,
			CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
			0,
			&backwardDataAlgo
	) );
	std::cout << "Fastest backward data algorithm is Algo " << backwardDataAlgo << "\n";

	checkError( cudnnGetConvolutionBackwardDataWorkspaceSize(global.cudnnHandle,
			weightDesc,
			y.desc,
			convDesc,
			x->desc,
			backwardDataAlgo,
			&sizeInBytes) );
	global.ensureWorkSpace(sizeInBytes);

	//backward filter
	checkError( cudnnGetConvolutionBackwardFilterAlgorithm(global.cudnnHandle,
			x->desc,
			y.desc,
			convDesc,
			weightDesc,
			CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
			0,
			&backwardFilterAlgo
	) );
	std::cout << "Fastest backward filter algorithm is Algo " << backwardDataAlgo << "\n";

	checkError( cudnnGetConvolutionBackwardFilterWorkspaceSize(global.cudnnHandle,
			x->desc,
			y.desc,
			convDesc,
			weightDesc,
			backwardFilterAlgo,
			&sizeInBytes) );
	global.ensureWorkSpace(sizeInBytes);

	return &y;
}
void ConvolutionLayer::forward(const string& tag, Variable& x, Variable& y)
{
	if(y.n != x.n)
	{
		y.setDesc(x.n);
	}

	checkError( cudnnConvolutionForward(global.cudnnHandle,
			&global.one,
			x.desc,
			x.data,
			weightDesc,
			weight.data,
			convDesc,
			forwardAlgo,
			global.workSpace,
			global.workSpaceSize,
			&global.zero,
			y.desc,
			y.data) );

	checkError( cudnnAddTensor( global.cudnnHandle,
					&global.one, biasDesc, bias.data,
					&global.one, y.desc, y.data) );
}

void ConvolutionLayer::backward(const string& tag, Variable& x, Variable& y)
{
	if(x.grad.data!=NULL)
	{
		checkError( cudnnConvolutionBackwardData(global.cudnnHandle,
				&global.one,
				weightDesc,
				weight.data,
				y.desc,
				y.grad.data,
				convDesc,
				backwardDataAlgo,
				global.workSpace,
				global.workSpaceSize,
				&global.zero,
				x.desc,
				x.grad.data ) );
	}

}

void ConvolutionLayer::calculateGradient(const string& tag, Variable& x, Variable& y)
{
	checkError( cudnnConvolutionBackwardFilter(global.cudnnHandle,
			&global.one,
			x.desc,
			x.data,
			y.desc,
			y.grad.data,
			convDesc,
			backwardFilterAlgo,
			global.workSpace,
			global.workSpaceSize,
			&global.one,
			weightDesc,
			weight.grad.data) );

	checkError( cudnnConvolutionBackwardBias( global.cudnnHandle,
			&global.one,
			y.desc,
			y.grad.data,
			&global.one,
			biasDesc,
			bias.grad.data) );
}

void ConvolutionLayer::forward()
{
	forward(tag, *x, y);
}

void ConvolutionLayer::backward()
{
	backward(tag, *x, y);
}


void ConvolutionLayer::calculateGradient()
{
	calculateGradient(tag, *x, y);
}


} /* namespace cytonVR */
