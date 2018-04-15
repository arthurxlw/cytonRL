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

#include "DueLinLayer.h"
#include "Global.h"
#include "cublasWrapper.h"

namespace cytonLib
{

Variable* DueLinLayer::init(string tag_, Variable* x_,
		int dimOutput_, bool biased_, WeightFactory* weightFactory_)
{
	this->tag=tag_;
	this->x=x_;

	WeightFactory* pWF=weightFactory_;
	if(pWF==NULL)
	{
		pWF=&weightFactory;
	}

	dimInput=x->c * x->h * x->w;
	dimOutput=dimOutput_;

	linU.init(tag+".linU", x, dimOutput, true, pWF);
	linBias.init(tag+".linBias", x, 1, true, pWF);
	linBias.addGrad=true;

	y.resize(x->n, dimOutput, 1, 1);
	y.enlarge=false;

	wu.resize(dimOutput, dimOutput);
	wu.setValue(-1.0/(Precision)dimOutput);
	wu.diag().setValue(1.0 - 1.0/(Precision)dimOutput);

	return &y;
}

void DueLinLayer::forward(const string& tag_, Variable& x, Variable& y)
{
	linU.forward();
	linBias.forward();

	int num=x.n;
	assert(x.c*x.h*x.w == dimInput);

	y.resize(x.n, dimOutput, 1, 1);
	checkError(cublasXgemm(global.cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
				dimOutput, num, dimOutput,
				&global.one, wu.data, dimOutput, linU.y.data, dimOutput,
				&global.zero, y.data, dimOutput));

	checkError(cublasXgemm(global.cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
			dimOutput, num, 1,
			&global.one, global.ones(dimOutput), dimOutput, linBias.y.data, 1,
			&global.one, y.data, dimOutput));
}

void DueLinLayer::backward(const string& tag_, Variable& x, Variable& y)
{
	int num=x.n;

	checkError(cublasXgemm(global.cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
				1, num, dimOutput,
				&global.one, global.ones(dimOutput), 1, y.grad.data, dimOutput,
				&global.zero, linBias.y.grad.data, 1));

	checkError(cublasXgemm(global.cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
					dimOutput, num, dimOutput,
					&global.one, wu.data, dimOutput, y.grad.data, dimOutput,
					&global.zero, linU.y.grad.data, dimOutput));

	linU.backward();

	linBias.backward();
}

void DueLinLayer::calculateGradient(const string& tag_, Variable& x, Variable& y)
{
	linU.calculateGradient();
	linBias.calculateGradient();
}

void DueLinLayer::forward()
{
	forward(tag, *x, y);
}

void DueLinLayer::backward()
{
	backward(tag, *x, y);
}


void DueLinLayer::calculateGradient()
{
	calculateGradient(tag, *x, y);
}

} /* namespace cytonLib */
