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

#include "Network.h"
#include "WeightFactory.h"
#include "Global.h"

namespace cytonLib
{

void Network::init()
{
	nGradientFail=0;
}


Precision Network::getScore()
{
	return 0;
}

void Network::forward()
{
	for(int k=0; k<layers.size(); k++)
	{
		Layer* layer=layers.at(k);
		layer->forward();
	}
}

void Network::backward()
{
	for(int k=layers.size()-1; k>=0; k--)
	{
		Layer* layer=layers.at(k);
		layer->backward();
	}
}

void Network::calculateGradient()
{
	int k0=layers.size()-1;
	for(int k=k0; k>=0; k--)
	{
		Layer* layer=layers.at(k);
		layer->calculateGradient();
	}
}

Precision Network::setEmsError(Variable& y, HostMatPrec& target)
{

	HostMatPrec hy;
	hy.copyFrom(y);

	HostMatPrec hDy;
	hDy.resize(hy.ni, hy.nj);

	assert(hy.ni==target.ni & y.nj==target.nj);
	Precision score=0.0;
	for(int i=0; i<hy.length(); i++)
	{
		Precision& ty=hy.at(i);
		Precision& tt=target.at(i);
		Precision& td=hDy.at(i);

		Precision diff=(tt-ty);
		score -= diff*diff/2.0;
		td=diff;
	}

	y.grad.copyFrom(hDy);
	return score;


}



} /* namespace cytonLib */
