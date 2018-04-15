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

#ifndef _CYTONLIB_HOSTMATRIX_H_
#define _CYTONLIB_HOSTMATRIX_H_

#include "Matrix.h"

namespace cytonLib
{

template<typename T> class DeviceMatrix;

template<typename T> class HostMatrix;

template<typename T>
class HostMatrix: public Matrix<T>
{
public:
	HostMatrix();

	HostMatrix(size_t ni_, size_t nj_);

	void resize(size_t ni_, size_t nj_);

	~HostMatrix();

	void copyFrom(const DeviceMatrix<T>& other);

	void copyFrom(const HostMatrix<T>& other);

	void copyFrom(const T* data, int ni, int nj);

	HostMatrix<T> range(size_t i0, long int i1, size_t j0, long int j1);

	bool continuous() const;

	void setValue(T value);

	void swap(HostMatrix<T>& other);

	T sum() const;

	void scale(T factor);

	void save(ostream& f);

	void load(istream& f);

	void load(const string& fileName);

	T max() const;

	T min() const;

	void setValue(const HostMatrix<T>& other);

	void add(T c);

	void setZero();

};


typedef HostMatrix<int> HostMatInt;

typedef HostMatrix<uchar> HostMatUchar;

} /* namespace cytonLib */

#endif /* HOSTMATRIXBASE_H_ */
