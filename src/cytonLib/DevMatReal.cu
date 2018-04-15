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

#include "HostMatReal.h"
#include "DevMatReal.h"
#include "cublasWrapper.h"
#include "Global.h"

namespace cytonLib
{


template<typename T>
DevMatReal<T>::DevMatReal()
{
}

template<typename T>
DevMatReal<T>::DevMatReal(size_t ni, size_t nj)
{
	this->resize(ni, nj);
}

template<typename T>
void DeviceMatrix<T>::setZero()
{
	if(this->continuous())
	{
		checkError(cudaMemset(this->data, 0, this->length()*sizeof(T)));
	}
	else
	{
		for(size_t j=0; j<this->nj; j++)
		{
			checkError(cudaMemset(this->data+this->stride*j, 0, this->ni*sizeof(T)));
		}
	}
}

template<typename T>
void DevMatReal<T>::initRandom()
{
	size_t len=this->length();
    checkError(curandGenerateUniformX(global.curandGenerator, this->data, len));
}

template<typename T>
__global__ void initRandom_kernel(T *data, size_t n, T a, T b) {
    size_t i = blockDim.x*blockIdx.x + threadIdx.x;
    if(i < n)
    {
    	T& t=data[i];
    	t= a*t+b;
    }
}

template<typename T>
void DevMatReal<T>::initRandom(T low, T up)
{
	this->initRandom();
	size_t n=this->length();
	T a=(up-low);
	T b=low;
	initRandom_kernel<<<ceil(n, blockSize), blockSize>>>(this->data, n, a, b);
	checkError(cudaGetLastError());
}

template<typename T>
void DevMatReal<T>::add(T* mat, T alpha)
{
	checkError(cublasXaxpy(global.cublasHandle, this->length(), &alpha,
			mat, 1, this->data, 1));
}

template<typename T>
void DevMatReal<T>::scale(T a)
{
	assert(this->continuous());
	checkError(cublasXscal(global.cublasHandle, this->length(), &a,
			this->data, 1));
}

template<typename T>
void DevMatReal<T>::update(T* mat, T a, T b)
{
	checkError(cublasXscal(global.cublasHandle, this->length(), &a,
			this->data, 1));

	checkError(cublasXaxpy(global.cublasHandle, this->length(), &b,
			mat, 1, this->data, 1));
}

template<typename T>
void DevMatReal<T>::addTo(T* mat, T alpha)
{
	checkError(cublasXaxpy(global.cublasHandle, this->length(), &alpha,
			this->data, 1, mat, 1));
}


template<typename T>
T DevMatReal<T>::getNorm()
{
	assert(this->continuous());
	T res=0;
	checkError(cublasXnrm2(global.cublasHandle, this->length(), this->data, 1, &res));
	return res;
}

template<typename T>
T DevMatReal<T>::clip(T threshold)
{
	T res=this->getNorm();
	if(res>threshold)
	{
		this->scale(threshold/res);
	}
	return res;
}

template<typename T>
T DevMatReal<T>::max() const
{
	HostMatReal<T> mat;

	T ans;
	if(this->length()>4096)
	{
		thrust::device_ptr<T> dev_ptr = thrust::device_pointer_cast(this->data);
		thrust::device_ptr<T> iter =
			thrust::max_element(dev_ptr, dev_ptr+this->length());
		ans=*iter;
	}
	else
	{
		mat.copyFrom(*this);
		ans=mat.max();
	}
	return ans;
}

template<typename T>
__global__
void devMatReal_reduceCols_min(
    T *result, const T *mat, const MatrixDim d)
{
  __shared__ T sdata[CU1DBLOCK];
  const size_t tid = threadIdx.x;
  const size_t i = blockIdx.x;
  const size_t row_start = i * d.stride;

  T tdata = sizeof(T) == sizeof(float) ? CUDART_INF_F : CUDART_INF;
  for (size_t j = tid; j < d.ni; j += CU1DBLOCK) {
    tdata = fmin(tdata, mat[row_start + j]);
  }
  sdata[tid] = tdata;
  __syncthreads();

  // Tree reduce
# pragma unroll
  for (size_t shift = CU1DBLOCK / 2; shift > warpSize; shift >>= 1) {
    if (tid < shift)
      sdata[tid] = fmin(sdata[tid], sdata[tid + shift]);
    __syncthreads();
  }

  // Reduce last warp. Threads implicitly synchronized within a warp.
  if (tid < warpSize) {
    for (size_t shift = warpSize; shift > 0; shift >>= 1)
      sdata[tid] = fmin(sdata[tid], sdata[tid + shift]);
  }

  // Output to vector result.
  if (tid == 0) {
    result[i] = sdata[0];
  }
}

template<typename T>
T DeviceMatrix<T>::min() const
{
	HostMatReal<T> mat;

	if(this->length()>4096)
	{
		DevMatReal<T> col_min(this->nj, 1);
		 devMatReal_reduceCols_min<<<this->nj, CU1DBLOCK>>>(col_min.data, this->data, this->dim());
		T ans = col_min.min();
	}
	else
	{
		mat.copyFrom(*this);
	}
	return mat.min();
}

template<typename T>
DevMatReal<T> DevMatReal<T>::diag()
{
	DevMatReal<T> res;
	size_t n=std::min(this->ni, this->nj);
	res.set(1, this->stride+1, n, this->data);
	return res;
}

template<typename T>
DevMatReal<T> DevMatReal<T>::range(size_t i0, long int i1, size_t j0, long int j1)
{
	DevMatReal<T> res;
	if(i1<0)
	{
		i1=this->ni;
	}
	else
	{
		if(i1>this->ni)
		{
			XLLib::printfln("DevMatReal.range assert.i1 fail");
			std::cout<<i0<<" "<<i1<<" "<<j0<<" "<<j1<<"\n";
			XLLib::printfln("");
		}
		assert(i1<=this->ni);
	}
	if(j1<0)
	{
		j1=this->nj;
	}
	else
	{
		if(j1>this->nj)
		{
			XLLib::printfln("DevMatReal.range assert.j1 fail");
			std::cout<<"ni nj "<<this->ni<<" "<<this->nj<<"\n";
			std::cout<<"i0 i1 j0 j1 "<<i0<<" "<<i1<<" "<<j0<<" "<<j1<<"\n";
			XLLib::printfln("");
		}
		assert(j1<=this->nj);
	}
	res.set(i1-i0, this->stride, j1-j0, this->data+this->stride*j0+i0);
	res.enlarge=false;
	return res;
}

template class DevMatReal<double>;
template class DevMatReal<float>;


} /* namespace cytonLib */
