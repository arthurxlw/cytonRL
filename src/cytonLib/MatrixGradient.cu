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

#include "MatrixGradient.h"


namespace cytonLib
{
extern bool testMode;

void MatrixGradient::resize(int ni, int nj)
{
	assert(!frozen);
	DevMatPrec::resize(ni, nj);
	if(!testMode)
	{
		if(gradShare)
		{
			grad.set(ni, ni, nj, data);
		}
		else
		{
			grad.resize(ni, nj);
		}
	}
}

void MatrixGradient::set(int ni, int stride, int nj, Precision* data, Precision* gradData)
{
	assert(!frozen);
	DevMatPrec::set(ni, stride, nj, data);
	grad.set(ni, stride, nj, gradData);
}

void MatrixGradient::set(MatrixGradient& other)
{
	set(other.ni, other.stride, other.nj, other.data, other.grad.data);
}

void MatrixGradient::setForced(int ni, int stride, int nj, Precision* data, Precision* gradData)
{
	DevMatPrec::setForced(ni, stride, nj, data);
	if(!testMode)
	{
		grad.setForced(ni, stride, nj, gradData);
	}
}

void MatrixGradient::set(int ni, int stride, int nj, MatrixGradient* v, int offset)
{
	assert(!frozen);
	set(ni, stride, nj, v->data+offset, v->grad.data+offset);
}

void MatrixGradient::setForced(int ni, int stride, int nj, MatrixGradient* v, int offset)
{
	assert(!frozen);
	setForced(ni, stride, nj, v->data+offset, v->grad.data+offset);
}


}
