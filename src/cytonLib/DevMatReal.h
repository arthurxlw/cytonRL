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

#ifndef _CYTONLIB_DEVMATREAL_H_
#define _CYTONLIB_DEVMATREAL_H_

#include "DeviceMatrix.h"

namespace cytonLib
{

template<typename T>
class HostMatReal;

template<typename T>
class HostVecReal;

template<typename T>
class DevMatReal: public DeviceMatrix<T>
{
public:
	DevMatReal();

	DevMatReal(size_t ni, size_t nj);

	void initRandom();

	void initRandom(T low, T up);

	void add(T* mat, T alpha=1.0);

	void addTo(T* mat, T alpha=1.0);

	void scale(T a);

	void update(T* mat, T a, T b);

	T max() const;

	T min() const;

	T getNorm();

	T clip(T threshold);

	DevMatReal<T> diag();

	DevMatReal<T> range(size_t i0, long int i1, size_t j0, long int j1);
};


typedef DevMatReal<Precision> DevMatPrec;

} /* namespace cytonLib */

#endif /* DEVMATREAL_H_ */
