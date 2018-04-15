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

#ifndef _CYTONRL_NETWORKRL_H_
#define _CYTONRL_NETWORKRL_H_

#include "Network.h"
#include "Variable.h"
#include "ConvolutionLayer.h"
#include "ActivationLayer.h"
#include "LinearLayer.h"
#include "DueLinLayer.h"
#include "SoftmaxLayer.h"
#include "WeightFactory.h"

namespace cytonRl
{

using namespace cytonLib;

class NetworkRl: public Network
{
public:
	ConvolutionLayer conv1;
	ActivationLayer act1;
	ConvolutionLayer conv2;
	ActivationLayer act2;
	ConvolutionLayer conv3;
	ActivationLayer act3;
	LinearLayer lin1;
	ActivationLayer act4;
	LinearLayer lin2;
	DueLinLayer dueLin2;

	WeightFactory weightFactory;

	DevMatInt targetAction;
	DevMatPrec targetQ;
	DevMatPrec qScore;
	DevMatPrec diffs;
	DevMatPrec sampleWeights;

	void init(Variable* x, int nTarget);

	void forwardQ(Variable* x);

	Precision learnQ(Precision lambda, HostMatPrec& hDiffs, vector<Precision>& sampleWeights);
};

} /* namespace cytonLib */

#endif /* REINLEARNNET_H_ */
