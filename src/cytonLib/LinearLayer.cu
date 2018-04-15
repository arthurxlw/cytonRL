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

#include "LinearLayer.h"
#include "WeightFactory.h"
#include "Global.h"
#include "cublasWrapper.h"

namespace cytonLib
{

Variable* LinearLayer::init(string tag_, Variable* x_,
		int dimOutput_, bool biased_, WeightFactory* weightFactory_)
{
	this->tag=tag_;
	this->x=x_;
	this->base=NULL;

	WeightFactory* pWF=weightFactory_;
	if(pWF==NULL)
	{
		pWF=&weightFactory;
	}

	dimInput=x->c * x->h * x->w;
	dimOutput=dimOutput_;
	biased=biased_;

	y.resize(x->n, dimOutput, 1, 1);
	y.enlarge=false;

	pWF->create(w, tag+".w", dimInput, dimOutput);
	if(biased)
	{
		pWF->create(b, tag+".b", dimOutput, 1);
	}
	return &y;
}

Variable* LinearLayer::init(string tag_, LinearLayer* base_, Variable* x_)
{
	this->tag=tag_;
	this->x=x_;
	this->base=base_;
	this->addGrad=base->addGrad;

	dimInput=x->c * x->h * x->w;
	assert(dimInput==base->dimInput);
	dimOutput=base->dimOutput;
	biased=base->biased;

	y.resize(x->n, dimOutput, 1, 1);
	y.enlarge=false;

	return &y;

}

void LinearLayer::forward()
{
	if(base!=NULL)
	{
		w.set(base->w);
		if(biased)
		{
			b.set(base->b);
		}
		base=NULL;
	}
	int num=x->n;
	assert(x->c*x->h*x->w == dimInput);

	y.resize(x->n, dimOutput, 1, 1);
	checkError(cublasXgemm(global.cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
			dimOutput, num, dimInput,
			&global.one, w.data, w.ni, x->data, dimInput,
			&global.zero, y.data, dimOutput));

	if(biased)
	{
		checkError(cublasXgemm(global.cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
				dimOutput, num, 1,
				&global.one, b.data, b.ni, global.ones(num), 1,
				&global.one, y.data, dimOutput));
	}
}

void LinearLayer::backward()
{
	int num=x->n;
	Precision* beta=addGrad?&global.one:&global.zero;
	checkError(cublasXgemm(global.cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
			dimInput, num, dimOutput,
			&global.one, w.data, w.ni, y.grad.data, dimOutput,
			beta, x->grad.data, dimInput));
}

void LinearLayer::calculateGradient()
{
	int num=x->n;
	checkError(cublasXgemm(global.cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T,
			dimInput, dimOutput,  num,
			&global.one, x->data, dimInput, y.grad.data, dimOutput,
			&global.one, w.grad.data, w.grad.ni));

	if(biased)
	{
		checkError(cublasXgemm(global.cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
				dimOutput, 1, num,
				&global.one, y.grad.data, dimOutput, global.ones(num), num,
				&global.one, b.grad.data, b.ni));
	}
}


} /* namespace cytonLib */
