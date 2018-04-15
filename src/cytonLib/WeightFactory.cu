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

#include "WeightFactory.h"
#include "Global.h"
#include "cublasWrapper.h"
#include "utils.h"

namespace cytonLib {

WeightFactory weightFactory;

void WeightFactory::init(const string& method)
{
	if(method=="RMSprop")
	{
		optRmsprop=true;
		RMSpropGamma=0.95;
		RMSpropEpsilon=1e-2;
	}
	else if(method=="SGD")
	{
		optSgd=true;
	}
	else
	{
		assert(false);
	}

	probeNorms.init(1);

}

void WeightFactory::create(Weight& weight, string tag, int ni, int nj)
{
	weight.create(tag, ni, nj);
	weights.push_back(&weight);
}

void WeightFactory::alloc(Precision clipGradient)
{

	int length=0;
	for(int i=0;i<weights.size();i++)
	{
		Weight& w=*weights.at(i);
		XLLib::printfln(global.os, "weight%d %s %d*%d", i, w.tag.c_str(), w.ni, w.nj);
		length+=w.length();
	}
	whole.resize(length, 1);
	whole.clipGrad=clipGradient;
	XLLib::printfln(global.os, "totalWeight %d",length);

	int offset=0;
	for(vector<Weight*>::iterator iw=weights.begin();iw!=weights.end();iw++)
	{
		Weight& w=*(*iw);
		w.set(w.ni, w.ni, w.nj, whole.data+offset, whole.grad.data+offset);
		offset+=w.length();
	}

	whole.initRandom(-global.initFactor, global.initFactor);

	if(optRmsprop)
	{
		momentum.resize(whole.ni, whole.nj);
		momentum.setZero();
		gradientVariance.resize(whole.ni, whole.nj);
		gradientVariance.setZero();
		dWeight.resize(whole.ni, whole.nj);
	}
	else if(optSgd)
	{
	}
	else
	{
		assert(false);
	}

}



void WeightFactory::clearGrad()
{
	whole.grad.setZero();
}


__global__
void weightFactory_update_rmsProp1(Precision* grad, Precision* gradMomentum, Precision* gradVar,  Precision* weight, Precision* dWeight, int len,
		Precision gamma, Precision epsilon, Precision lambda )
{
	int i=blockDim.x*blockIdx.x+threadIdx.x;
	if(i<len)
	{
		Precision& g=grad[i];
		Precision& gm=gradMomentum[i];
		Precision& gv=gradVar[i];
		Precision& w=weight[i];
		Precision& dw=dWeight[i];

		gm=(1-gamma)*g+gamma*gm;
		gv=(1-gamma)*g*g+gamma*gv;

		dw= 1.0/sqrt(gv-gm*gm+epsilon)*g*lambda;
		w += dw;
	}
}


void WeightFactory::update(Precision lambda)
{
	int len=whole.length();

	if(optRmsprop)
	{
		if(numUpdates==0)
		{
			XLLib::printfln("weightFactory.update RMSprop");
		}

		vector<Precision> norms(1);
		norms.at(0)=whole.grad.getNorm();
		if(whole.clipGrad>0)
		{
			if(numUpdates==0)
			{
				XLLib::printfln("weightFactory.clipGrad %g", whole.clipGrad);
			}
			whole.grad.clip(whole.clipGrad);
		}
		weightFactory_update_rmsProp1<<<ceil(len, blockSize), blockSize>>>(whole.grad.data, momentum.data, gradientVariance.data, whole.data, dWeight.data, len,
						RMSpropGamma, RMSpropEpsilon, lambda);
		probeNorms.update(norms);
	}
	else if(optSgd)
	{
		if(numUpdates==0)
		{
			XLLib::printfln("\nweightFactory.update sgd");
		}

		checkError(cublasXaxpy(global.cublasHandle, whole.length(), &lambda, whole.grad.data, 1, whole.data, 1));
	}
	else
	{
		assert(false);
	}

	numUpdates+=1;
}

void WeightFactory::save(const string& fileName)
{
	XLLib::dirPrepare4file(fileName);
	std::ofstream f(fileName.c_str());

	for(vector<Weight*>::iterator iw=weights.begin();iw!=weights.end();iw++)
	{
		Weight& w=*(*iw);
		f<<"##"<<w.tag<<"\n";
		w.save(f);
	}

	f.close();
}

void WeightFactory::load(const string& fileName)
{
	if(!XLLib::fileExists(fileName))
	{
		XLLib::printfln("model file %s does not exist.", fileName.c_str());
		assert(false);
	}
	ifstream f(fileName.c_str());
	for(int i=0; i<weights.size();i++)
	{
		Weight& w=*weights.at(i);
		string tTag=string("##")+w.tag;
		checkFile(f,tTag);
		w.load(f);
	}
	f.close();
}


string WeightFactory::statistics()
{
	ostringstream os;
	os<<probeNorms.toString();
	probeNorms.reset();
	return os.str();
}


} /* namespace cytonLib */
