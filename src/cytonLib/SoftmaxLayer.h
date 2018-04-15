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

#ifndef _CYTONLIB_SOFTMAXLAYER_H_
#define _CYTONLIB_SOFTMAXLAYER_H_

#include "basicHeads.h"
#include "utils.h"
#include "DeviceMatrix.h"
#include "HostMatrix.h"
#include "Variable.h"
#include "Layer.h"

namespace cytonLib {

class SoftmaxLayer: public Layer
{

public:
	DevMatPrec likehood;
	HostMatPrec likehood_host;
	Precision likehoodSum;
	bool sparse;
	HostMatPrec hY;

public:
	Variable* init(string tag_,Variable* x_);

	void forward();

	void backward();

	Precision backwardSparse(int* trgWords, Precision scale=1.0, bool ignore0=false);

	Precision backwardSmoothPrepare(int* trgWords, Precision scale=1.0, bool ignore0=false,
			Precision epsilon=0.1);


};

} /* namespace cytonLib */

#endif /* SOFTMAXLAYER_H_ */
