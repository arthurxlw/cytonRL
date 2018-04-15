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

#ifndef _CYTONLIB_MATRIXGRADIENT_H_
#define _CYTONLIB_MATRIXGRADIENT_H_

#include "basicHeads.h"
#include "HostMatReal.h"
#include "DevMatReal.h"

namespace cytonLib
{

class MatrixGradient : public DevMatPrec
{
public:
	string tag;
	DevMatPrec grad;
	bool gradShare;

	MatrixGradient()
	{
		gradShare=false;
		enlarge=true;
	}

	void resize(int ni, int nj);

	void set(int ni, int stride, int nj, Precision* data, Precision* gradData);

	void set(MatrixGradient& other);

	void setForced(int ni, int stride, int nj, Precision* data, Precision* gradData);

	void set(int ni, int stride, int nj, MatrixGradient* v, int offset=0);

	void setForced(int ni, int stride, int nj, MatrixGradient* v, int offset=0);

};

}

#endif /* VARIABLE_H_ */

