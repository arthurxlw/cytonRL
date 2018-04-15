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

#ifndef _CYTONLIB_WEIGHTFACTORY_H_
#define _CYTONLIB_WEIGHTFACTORY_H_

#include "Weight.h"
#include "NumberProbes.h"

namespace cytonLib {

class WeightFactory
{
public:
	vector<Weight*> weights;

	Weight whole;
	int numUpdates;
	int randNum;

	bool optSgd;

	DevMatPrec momentum;
	DevMatPrec gradientVariance;
	DevMatPrec updateVariance;
	DevMatPrec dWeight;

	bool optRmsprop;
	Precision RMSpropGamma;
	Precision RMSpropEpsilon;


	NumberProbes<Precision> probeNorms;

	WeightFactory()
	{
		numUpdates=0;

		optSgd=false;
		optRmsprop=false;
	}

	void init(const string& method);

	void create(Weight& weight, string tag, int ni, int nj);

	void alloc(Precision clipGradient);

	void clearGrad();

	void update(Precision lambda);

	void save(const string& fileName);

	void load(const string& fileName);

	string statistics();

};

extern WeightFactory weightFactory;

} /* namespace cytonLib */

#endif /* WEIGHTFACTORY_H_ */
