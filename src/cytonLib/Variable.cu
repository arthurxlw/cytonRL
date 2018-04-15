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

#include "Variable.h"
#include "HostMatReal.h"

namespace cytonLib
{

Variable::Variable()
{
	checkError(cudnnCreateTensorDescriptor(&desc));
	data=NULL;
	grad.data=NULL;

}

void Variable::setDesc(int n_, int c_, int h_, int w_, int nStride_, int cStride_, int hStride_, int wStride_)
{
	this->n=n_;
	this->c=c_;
	this->h=h_;
	this->w=w_;
	this->nStride=nStride_;
	this->cStride=cStride_;
	this->hStride=hStride_;
	this->wStride=wStride_;

	checkError(cudnnSetTensor4dDescriptorEx(
			desc, cudnnDataType,
			n, c, h, w,
			nStride, cStride, hStride, wStride
			));

}


void Variable::setDesc(int n_, int c_, int h_, int w_)
{
	this->n=n_;
	this->c=c_;
	this->h=h_;
	this->w=w_;
	this->nStride=c*h*w;
	this->cStride=h*w;
	this->hStride=w;
	this->wStride=1;

	checkError(cudnnSetTensor4dDescriptorEx(
			desc, cudnnDataType,
			n, c, h, w,
			nStride, cStride, hStride, wStride
			));

}

void Variable::setDesc(int n_)
{
	this->n=n_;
	nStride=c*h*w;
	cStride=h*w;
	hStride=w;
	wStride=1;

	checkError(cudnnSetTensor4dDescriptorEx(
			desc, cudnnDataType,
			n, c, h, w,
			nStride, cStride, hStride, wStride
			));

}


void Variable::resize(int n_, int c_, int h_, int w_, int matShape)
{
	if(this->n!=n_ || this->c!=c_ || this->h!=h_ || this->w!=w_ )
	{
		assert(!frozen);
		this->n=n_;
		this->c=c_;
		this->h=h_;
		this->w=w_;
		nStride=c*h*w;
		cStride=h*w;
		hStride=w;
		wStride=1;
		checkError(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, cudnnDataType, n, c, h, w));
	}

	int ni_=0;
	int nj_=0;
	if(matShape==0)
	{
		ni_=w*h*c;
		nj_=n;
	}
	else if(matShape==1)
	{
		ni_=w*h;
		nj_=c*n;
	}
	else if(matShape==2)
	{
		ni_=w;
		nj_=h*c*n;
	}
	else
	{
		assert(false);
	}
	if(this->ni !=ni_ || this->nj !=nj_ )
	{
		assert(!frozen);
		MatrixGradient::resize(ni_, nj_);
	}
}

void Variable::resize(const Variable& o)
{
	if(n!=o.n || c!=o.c || h!=o.h || w!=o.w)
	{
		assert(!frozen);
		this->resize(o.n, o.c, o.h, o.w);
	}
}

void Variable::reshape(int n_, int c_, int h_, int w_)
{
	assert(!frozen);
	assert(n*c*h*w==n_*c_*h_*w_);
	assert(n_>=1 && c_ >=1 && h_>=1 && w_>=1);
	this->n=n_;
	this->c=c_;
	this->h=h_;
	this->w=w_;
	nStride=c*h*w;
	cStride=h*w;
	hStride=w;
	wStride=1;

	MatrixGradient::reshapeMatrix(w*h*c, n);
}

void Variable::reshape(int n_, int c_, int h_)
{
	assert(!frozen);
	assert(n_>=1 && c_ >=1 && h_>=1);
	int size=n*c*h*w;
	int t=n_*c_*h_;
	assert(size%t==0);
	int w_=size/t;

	this->n=n_;
	this->c=c_;
	this->h=h_;
	this->w=w_;
	nStride=c*h*w;
	cStride=h*w;
	hStride=w;
	wStride=1;

	MatrixGradient::reshapeMatrix(w*h*c, n);
}

void Variable::reshape(int n_, int c_)
{
	assert(!frozen);
	assert(n_>=1 && c_ >=1);
	int size=n*c*h*w;
	int t=n_*c_;
	assert(size%t==0);
	int h_=size/t;
	int w_=1;

	this->n=n_;
	this->c=c_;
	this->h=h_;
	this->w=w_;

	MatrixGradient::reshapeMatrix(w*h*c, n);
}

void Variable::reshape(int n_)
{
	assert(!frozen);
	assert(n_>=1);
	int size=n*c*h*w;
	int t=n_;
	assert(size%t==0);
	int c_=size/t;
	int h_=1;
	int w_=1;

	this->n=n_;
	this->c=c_;
	this->h=h_;
	this->w=w_;

	MatrixGradient::reshapeMatrix(w*h*c, n);
}

void Variable::set(int n_, int c_, int h_, int w_, Precision* data, Precision *grad)
{
	assert(!frozen);
	this->n=n_;
	this->c=c_;
	this->h=h_;
	this->w=w_;

	checkError(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, cudnnDataType, n, c, h, w));
	MatrixGradient::set(w*h*c, w*h*c, n,  data, grad);
}

void Variable::setWithStrideH(int n_, int c_, int h_, int stride_, Precision* data, Precision *grad)
{
	assert(!frozen);
	this->n=n_;
	this->c=c_;
	this->h=h_;
	this->w=1;

	checkError(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, cudnnDataType, n, c, h, w));

	assert(stride_>=h_);
	MatrixGradient::set(h_, stride_, c_*n_,  data,  grad);
}

void Variable::setForced(int n_, int c_, int h_, int w_, Precision* data, Precision *grad)
{
	this->n=n_;
	this->c=c_;
	this->h=h_;
	this->w=w_;

	checkError(cudnnSetTensor4dDescriptor(desc, CUDNN_TENSOR_NCHW, cudnnDataType, n, c, h, w));
	MatrixGradient::setForced(w*h*c, w*h*c, n,  data, grad);
}

void Variable::set(int ni_, int nj_, Precision* data, Precision *grad)
{
	assert(!frozen);
	this->set(nj_, ni_, 1, 1, data, grad);
}

void Variable::set(Variable& o)
{
	this->set(o.n, o.c, o.h, o.w, o.data, o.grad.data);
}

void Variable::copyFrom(Variable& o)
{
	this->resize(o.n, o.c, o.h, o.w);
	DevMatPrec::copyFrom(o);
}

string Variable::toStringDim()
{
	ostringstream os;
	os<<XLLib::stringFormat("%d*%d*%d*%d", n, c, h, w);
	return os.str();
}

} /* namespace cytonVR */
