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

#ifndef _CYTONLIB_MATRIX_H_
#define _CYTONLIB_MATRIX_H_

#include "basicHeads.h"

namespace cytonLib
{

template<typename T>
struct MatrixElement
{
  size_t i;
  size_t j;
  T value;
};

typedef MatrixElement<Precision> MatElemPrec;

typedef struct MatrixDim_
{
	size_t ni;
	size_t stride;
	size_t nj;
} MatrixDim;

template<typename T>
class Matrix
{
public:
	T* data;
	size_t ni;
	size_t stride;
	size_t nj;
	size_t capacity;
	bool enlarge;
	bool frozen;

	Matrix(): data(NULL), ni(0), stride(0), nj(0), capacity(0), enlarge(true), frozen(false)
	{
	}

	inline size_t length() const
	{
		return ni*nj;
	}

	inline void reshape(size_t ni_, size_t nj_)
	{
		assert(ni*nj==ni_*nj_);
		ni=ni_;
		stride=ni_;
		nj=nj_;
	}

	inline void set(size_t ni_, size_t stride_, size_t nj_,  T* data_)
	{
		assert(capacity==0);

		ni=ni_;
		stride=stride_;
		nj=nj_;
		data=data_;
	}

	inline void set(size_t ni_, size_t nj_,  T* data_)
	{
		assert(capacity==0);

		ni=ni_;
		stride=ni_;
		nj=nj_;
		data=data_;
	}


	inline void set(Matrix<T>& other)
	{
		assert(capacity==0);

		ni=other.ni;
		stride=other.stride;
		nj=other.nj;
		data=other.data;
	}

	inline bool continuous() const
	{
		return stride==ni;
	}

	inline bool empty() const
	{
		return ni==0||nj==0;
	}

	inline T& at(size_t i, size_t j)
	{
		assert(i<ni && j<nj);
		return data[i + stride*j];
	}

	inline const T& at(size_t i, size_t j) const
	{
		assert(i<ni && j<nj);
		return data[i + stride*j];
	}

	inline T& at(size_t k)
	{
		if(k>=length())
		{
			assert(k<length());
		}
		if(continuous())
		{
			return data[k];
		}
		else
		{
			size_t i=k%ni;
			size_t j=k/ni;
			return data[i+stride*j];
		}
	}

	inline const T& at(size_t k) const
	{
		assert(k<length());
		if(continuous())
		{
			return data[k];
		}
		else
		{
			size_t i=k%ni;
			size_t j=k/ni;
			return data[i+stride*j];
		}
	}

	inline MatrixDim dim() const
	{
		MatrixDim res={ni, stride, nj};
		return res;
	}
};


void getBlockSizesForSimpleMatrixOperation(size_t num_rows,
                                           size_t num_cols,
                                           dim3 *dimGrid,
                                           dim3 *dimBlock) ;

} /* namespace cytonLib */

#endif /* MATRIXBASE_H_ */
