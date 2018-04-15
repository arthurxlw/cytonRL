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

#ifndef _CYTONLIB_LINEARLAYER_H_
#define _CYTONLIB_LINEARLAYER_H_

#include "Layer.h"
#include "Weight.h"
#include "Variable.h"
#include "WeightFactory.h"

namespace cytonLib
{

class LinearLayer: public Layer
{
public:
	int dimInput;
	int dimOutput;

	Weight w;
	Weight b;
	bool biased;

	Variable* init(string tag_, Variable* x_, int dimOutput_, bool biased_=true, WeightFactory* weightFactory_=NULL);

	Variable* init(string tag_, LinearLayer* base, Variable* x_);

	void forward();

	void backward();

	void calculateGradient();

protected:

	LinearLayer* base;
};




} /* namespace cytonLib */

#endif /* LINEARLAYER_H_ */
