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

#ifndef _CYTONLIB_HOSTMATREAL_H_
#define _CYTONLIB_HOSTMATREAL_H_

#include "HostMatrix.h"

namespace cytonLib
{

template<typename T>
class HostVecReal;

template<typename T>
class HostMatReal: public HostMatrix<T>
{

public:

	HostMatReal()
	{
	}

	HostMatReal(size_t ni, size_t nj):HostMatrix<T>(ni, nj)
	{
	}

	HostMatReal<T> range(size_t i0, long int i1, size_t j0, long int j1);

	HostVecReal<T> getVec(size_t j);

	void add(T alpha, HostMatReal<T> other);

	void add(T alpha);

	T dotProduct(const HostMatReal<T>& o);


};

typedef HostMatReal<Precision> HostMatPrec;

} /* namespace cytonLib */

#endif /* HOSTMATREAL_H_ */
