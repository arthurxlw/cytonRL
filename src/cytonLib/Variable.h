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

#ifndef _CYTONLIB_VARIABLE_H
#define _CYTONLIB_VARIABLE_H

#include "MatrixGradient.h"


namespace cytonLib
{

class Variable: public MatrixGradient
{
public:

	int n, c, h, w;
	int nStride, cStride, hStride, wStride;
	cudnnTensorDescriptor_t desc;
	Variable();

	void setDesc(int n_, int c_, int h_, int w_, int nStride_, int cStride_, int hStride_, int wStride_);

	void setDesc(int n_, int c_, int h_, int w_);

	void setDesc(int n_);

	void resize(int n_, int c_, int h_, int w_, int matShape=0);

	void resize(int ni_, int nj_);

	void resize(const Variable& other);

	void reshape(int n_, int c_, int h_, int w_);

	void reshape(int n_, int c_, int h_);

	void reshape(int n_, int c_);

	void reshape(int n_);

	void set(int n_, int c_, int h_, int w_, Precision* data, Precision *grad);

	void setForced(int n_, int c_, int h_, int w_, Precision* data, Precision *grad);

	void set(int ni_, int nj_, Precision* data, Precision *grad);

	void set(Variable& o);

	void copyFrom(Variable& o);

	void setWithStrideH(int n_, int c_, int h_, int stride_, Precision* data, Precision *grad);

	string toStringDim();
};

} /* namespace cytonVR */

#endif /* CUDNNVARNCHW_H_ */
