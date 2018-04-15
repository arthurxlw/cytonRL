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

#ifndef _CYTONLIB_DEVICEMATRIX_H_
#define _CYTONLIB_DEVICEMATRIX_H_

#include "Matrix.h"

namespace cytonLib
{

template<typename T> class HostMatrix;

template<typename T> class DeviceMatrix;

template<typename T>
class DeviceMatrix: public Matrix<T>
{
public:

	DeviceMatrix();

	DeviceMatrix(size_t ni, size_t nj);

	void freeData();

	~DeviceMatrix();

	DeviceMatrix<T> range(size_t i0, long int  i1, size_t j0, long int j1);

	void setForced(size_t ni_, size_t stride_, size_t nj_,  T* data_);

	void resize(size_t ni_, size_t nj_);

	void copyFrom(const HostMatrix<T>& m);

	void copyFrom(const DeviceMatrix<T>& m);

	void copyFrom(const T* data, size_t ni_, size_t nj_);

	void copyFrom(const T* data, size_t len);

	void setValue(T Value);

	void setValue(const DeviceMatrix<T>& m);

	void setValueRelax(const HostMatrix<T>& m);

	double checkCompare(const HostMatrix<T>& m);

	void setZero();

	void save(ostream& f);

	void save(const string& fileName);

	void load(istream& f);

	void load(const string& fileName);

	void swap(DeviceMatrix<T> & mat);

	void initRandom(T initFactor);

	void reshapeMatrix(size_t ni_, size_t nj_);

	T max() const;

	T min() const;

	T sum() const;

	void applyMin(T minValue);

	template<typename D>
	void convertFrom(const DeviceMatrix<D>&m , Precision scale);

};

typedef DeviceMatrix<int> DevMatInt;

typedef DeviceMatrix<uchar> DevMatUchar;

} /* namespace cytonLib */

#endif /* DEVICEMATRIXBASE_H_ */
