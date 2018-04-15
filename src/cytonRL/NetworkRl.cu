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

#include "NetworkRl.h"
#include "ParamsRl.h"
#include "Global.h"
#include "WeightFactory.h"
#include "utils.h"

namespace cytonRl
{

void NetworkRl::init(Variable* x, int nTarget)
{
	weightFactory.init(params.optimizer);

	vector<int>& networkSize=params.networkSize;
	int k=0;

	Variable* tx=x;
	vector<Precision> inputDims;
	{
		ConvolutionLayer& layer=conv1;
		inputDims.push_back(tx->c*8*8);
		tx=layer.init("conv1", tx, networkSize.at(k++), 8, 8, 4, 4, 0, 0, &weightFactory);
		layers.push_back(&layer);
	}

	{
		ActivationLayer& layer=act1;
		tx=layer.init("act1", tx, CUDNN_ACTIVATION_RELU);
		layers.push_back(&layer);
	}

	{
		ConvolutionLayer& layer=conv2;
		inputDims.push_back(tx->c*4*4);
		tx=layer.init("conv2", tx, networkSize.at(k++), 4, 4, 2, 2, 0, 0, &weightFactory);
		layers.push_back(&layer);
	}

	{
		ActivationLayer& layer=act2;
		tx=layer.init("act2", tx, CUDNN_ACTIVATION_RELU);
		layers.push_back(&layer);
	}

	{
		ConvolutionLayer& layer=conv3;
		inputDims.push_back(tx->c*3*3);
		tx=layer.init("conv3", tx, networkSize.at(k++), 3, 3, 1, 1, 0, 0, &weightFactory);
		layers.push_back(&layer);
	}

	{
		ActivationLayer& layer=act3;
		tx=layer.init("act3", tx, CUDNN_ACTIVATION_RELU);
		layers.push_back(&layer);
	}

	{
		LinearLayer& layer=lin1;
		inputDims.push_back(tx->c*tx->h*tx->w);
		tx=layer.init("lin1", tx, networkSize.at(k++), true, &weightFactory);
		layers.push_back(&layer);
	}

	{
		ActivationLayer& layer=act4;
		tx=layer.init("act4", tx, CUDNN_ACTIVATION_RELU);
		layers.push_back(&layer);
	}

	if(params.dueling)
	{
		DueLinLayer& layer=dueLin2;
		inputDims.push_back(tx->c*tx->h*tx->w);
		inputDims.push_back(tx->c*tx->h*tx->w);
		tx=layer.init("dueLin2", tx, nTarget, true, &weightFactory);
		layers.push_back(&layer);
	}
	else
	{
		LinearLayer& layer=lin2;
		inputDims.push_back(tx->c*tx->h*tx->w);
		tx=layer.init("lin2", tx, nTarget, true, &weightFactory);
		layers.push_back(&layer);
	}

	assert(k==networkSize.size());

	XLLib::printfln("\nPredictLayer.scale input %s",x->toStringDim().c_str());
	for(int i=0; i<layers.size();i++)
	{
		Layer& t=*layers.at(i);
		XLLib::printfln("%s %s", t.tag.c_str(), t.y.toStringDim().c_str());
	}

	weightFactory.alloc(10.0);

	assert(weightFactory.weights.size()==inputDims.size()*2);
	for(int i=0; i<weightFactory.weights.size(); i++)
	{
		Weight& w=*weightFactory.weights.at(i);
		Precision inputDim=inputDims.at(i/2);
		Precision factor=1.0/sqrt(inputDim);
		w.initRandom(-factor, factor);
		XLLib::printfln("reinitWeight %s %g", w.tag.c_str(), factor);
	}
}



void NetworkRl::forwardQ(Variable* x)
{

	Network::layers.front()->x=x;
	Network::forward();
	y.set(Network::layers.back()->y);
}

__global__
void reinLearnNet_setGrad(Precision*y, Precision* dy, int batchSize, int na, int* actions, Precision* yTarget, Precision* score,
		Precision* diffs, Precision* sampleWeights)
{
	int idx=blockDim.x*blockIdx.x+threadIdx.x;
	if(idx<batchSize)
	{
		int action=actions[idx];
		Precision yt=yTarget[idx];

		int offset=na*idx+action;
		Precision diff=yt-y[offset];
		score[idx]=-diff*diff/2;
		Precision sw=sampleWeights[idx];

		Precision tDy=diff*sw;
		if(tDy>1.0)
		{
			tDy=1.0;
		}
		else if(tDy<-1.0)
		{
			tDy=-1.0;
		}
		dy[offset]=tDy;
		diffs[idx]=tDy;
	}
}

Precision NetworkRl::learnQ(Precision lambda, HostMatPrec& hDiffs, vector<Precision>& hSampleWeights)
{

	Variable& y=layers.back()->y;

	int len=y.n;;
	y.grad.setZero();
	qScore.resize(len, 1);
	diffs.resize(len, 1);
	sampleWeights.resize(len, 1);
	sampleWeights.copyFrom(&hSampleWeights.at(0), len);
	reinLearnNet_setGrad<<<ceil(len, blockSize), blockSize>>>(
			y.data, y.grad.data, len, y.ni, targetAction.data, targetQ.data, qScore.data, diffs.data, sampleWeights.data);
	hDiffs.copyFrom(diffs);

	Precision score=qScore.sum();

	if(lambda>0)
	{
		Network::backward();

		weightFactory.clearGrad();
		Network::calculateGradient();

		weightFactory.update(lambda);
	}

	return score;
}

} /* namespace cytonLib */
