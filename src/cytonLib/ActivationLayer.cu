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

#include "ActivationLayer.h"
#include "Global.h"

namespace cytonLib
{

ActivationLayer::ActivationLayer()
{

}

//	typedef enum
//	{
//	    CUDNN_ACTIVATION_SIGMOID      = 0,
//	    CUDNN_ACTIVATION_RELU         = 1,
//	    CUDNN_ACTIVATION_TANH         = 2,
//	    CUDNN_ACTIVATION_CLIPPED_RELU = 3,
//	    CUDNN_ACTIVATION_ELU          = 4
//	} cudnnActivationMode_t;
//

Variable* ActivationLayer::init(string tag_, Variable* x_, cudnnActivationMode_t mode)
{

	this->tag=tag_;
	this->x=x_;

	y.resize(x->n, x->c, x->h, x->w);
	y.enlarge=false;

	checkError( cudnnCreateActivationDescriptor(&activeDesc) );

	checkError( cudnnSetActivationDescriptor(activeDesc, mode, CUDNN_PROPAGATE_NAN, 0.0));

	return &y;
}

void ActivationLayer::forward()
{
	y.resize(*x);

	checkError( cudnnActivationForward(global.cudnnHandle, activeDesc,
			&global.one, x->desc, x->data,
			&global.zero, y.desc, y.data) );
}

void ActivationLayer::backward()
{
	checkError( cudnnActivationBackward(global.cudnnHandle, activeDesc,
			&global.one, y.desc, y.data, y.desc, y.grad.data, x->desc, x->data,
			&global.zero, x->desc, x->grad.data ) );
}



}

