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

#include "DeviceMatrix.h"
#include "HostMatrix.h"
#include "cublasWrapper.h"
#include "Global.h"

namespace cytonLib {

template<typename T>
DeviceMatrix<T>::DeviceMatrix()
{
}

template<typename T>
DeviceMatrix<T>::DeviceMatrix(size_t ni, size_t nj)
{
	resize(ni, nj);
}

template<typename T>
void DeviceMatrix<T>::freeData()
{
	if(this->capacity>0)
	{
		assert(this->ni==this->stride);

		checkError(cudaFree(this->data));
		this->capacity=0;
	}
}

template<typename T>
DeviceMatrix<T>::~DeviceMatrix()
{
	freeData();
}


template<typename T>
void DeviceMatrix<T>::setForced(size_t ni_, size_t stride_, size_t nj_,  T* data_)
{
	if(this->capacity>0)
	{
		freeData();
	}

	this->set(ni_, stride_, nj_, data_);
}

template<typename T>
void DeviceMatrix<T>::resize(size_t ni_, size_t nj_)
{
	if(this->ni!=ni_ || this->nj!=nj_)
	{
		if(this->ni*this->nj==ni_*nj_)
		{
			this->reshape(ni_, nj_);
		}
		else
		{
			assert(this->continuous());
			assert(this->length()==0 || this->capacity>0);

			this->ni = ni_;
			this->stride = ni_;
			this->nj = nj_;

			if(this->capacity < this->ni*this->nj)
			{
				assert(this->enlarge);
				freeData();

				this->capacity=(size_t)this->ni * this->nj;
				checkError(cudaMalloc(&this->data,this->capacity*sizeof(T)));
				assert(this->data!=NULL);
			}
		}
	}
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
DeviceMatrix<T> DeviceMatrix<T>::range(size_t i0, long int i1, size_t j0, long int j1)
{
	DeviceMatrix<T> res;
	if(i1<0)
	{
		i1=this->ni;
	}
	else
	{
		if(i1>this->ni)
		{
			fprintf(stderr, "DeviceMatrix.range assert.i1 fail\n");
			std::cout<<i0<<" "<<i1<<" "<<j0<<" "<<j1<<"\n";
			fprintf(stderr, "\n");
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
			fprintf(stderr, "DeviceMatrix.range assert.j1 fail\n");
			std::cout<<"ni nj "<<this->ni<<" "<<this->nj<<"\n";
			std::cout<<"i0 i1 j0 j1 "<<i0<<" "<<i1<<" "<<j0<<" "<<j1<<"\n";
			fprintf(stderr, "\n");
		}
		assert(j1<=this->nj);
	}
	res.set(i1-i0, this->stride, j1-j0, this->data+this->stride*j0+i0);
	res.enlarge=false;
	return res;
}

template<class T>
void DeviceMatrix<T>::copyFrom(const HostMatrix<T>& m)
{
	this->resize(m.ni, m.nj);
	if(m.continuous() && this->continuous())
	{
		checkError(cudaMemcpy(this->data, m.data, sizeof(T)*m.length(),cudaMemcpyDefault));
	}
	else
	{
		for(size_t j=0; j<this->nj; j++)
		{
			checkError(cudaMemcpy(this->data+this->stride*j, m.data+m.stride*j,
					sizeof(T)*this->ni,cudaMemcpyDefault));
		}

	}
}

template<typename T>
void DeviceMatrix<T>::copyFrom(const DeviceMatrix<T>& m)
{
	resize(m.ni, m.nj);
	this->setValue(m);
}

template<class T>
void DeviceMatrix<T>::copyFrom(const T* data, size_t ni_, size_t nj_)
{
	resize(ni_, nj_);
	checkError(cudaMemcpy(this->data, data, sizeof(T)*ni_*nj_,cudaMemcpyDefault));
}

template<class T>
void DeviceMatrix<T>::copyFrom(const T* data, size_t len)
{
	resize(len, 1);
	checkError(cudaMemcpy(this->data, data, sizeof(T)*len,cudaMemcpyDefault));
}


template<typename T, typename D>
__global__ void DeviceMatrix_convertFrom_kernel(T *data, size_t len,
		D* other, Precision scale)
{
	size_t i=blockIdx.x * blockDim.x + threadIdx.x;
	if (i < len)
	{
		D to=other[i];
		T tt=to*scale;
		data[i] = tt;
	}
}

template<typename T, typename D>
__global__ void DeviceMatrix_convertFrom_kernel(T *data, size_t ni, size_t stride, size_t nj,
		D* other, size_t oStride, Precision scale)
{
	size_t i=blockIdx.x * blockDim.x + threadIdx.x;
	if (i < ni)
	{
		size_t j=blockIdx.y;
		D to=other[oStride*j+i];
		T tt=to*scale;
		data[stride*j+i] = tt;
	}
}


template<typename T>
__global__ void DeviceMatrix_setValue_kernel(T *data, size_t ni, size_t stride, size_t nj, T value)
{
	size_t i=blockIdx.x * blockDim.x + threadIdx.x;
	if (i < ni)
	{
		size_t j=blockIdx.y;
		data[stride*j+i] = value;
	}
}

template<class T>
void DeviceMatrix<T>::setValue(T value)
{
	dim3 gridDim;
	gridDim.x = ceil(this->ni, blockSize);
	gridDim.y = this->nj;
	DeviceMatrix_setValue_kernel <<< gridDim, blockSize >>> (this->data, this->ni, this->stride, this->nj, value);
}

template<typename T>
__global__
void DeviceMatrix_setValue_mat_kernel(T *data, size_t ni, size_t stride, size_t nj, T* srcData, size_t srcStride)
{
	size_t i=blockIdx.x * blockDim.x + threadIdx.x;
	if (i < ni)
	{
		size_t j=blockIdx.y;
		data[stride*j+i] = srcData[srcStride*j+i];
	}
}

template<typename T>
void DeviceMatrix<T>::setValue(const DeviceMatrix<T>& m)
{
	assert(this->ni==m.ni && this->nj==m.nj);
	if(this->continuous() && m.continuous())
	{
		checkError(cudaMemcpy(this->data, m.data, sizeof(T)*m.length(),cudaMemcpyDefault));
	}
	else
	{
		dim3 gridDim;
		gridDim.x = ceil(this->ni, blockSize);
		gridDim.y = this->nj;
		DeviceMatrix_setValue_mat_kernel <<< gridDim, blockSize >>> (this->data, this->ni, this->stride, this->nj,
				m.data, m.stride);
	}
}


template<typename T>
void DeviceMatrix<T>::setValueRelax(const HostMatrix<T>& m)
{
	assert(this->length()==m.length());
	assert(this->continuous() && m.continuous());
	checkError(cudaMemcpy(this->data, m.data, sizeof(T)*m.length(),cudaMemcpyDefault));
}

template<typename T>
double DeviceMatrix<T>::checkCompare(const HostMatrix<T>& m)
{
	assert(this->length()==m.length());
	assert(this->continuous() && m.continuous());

	HostMatrix<T> h;
	h.copyFrom(*this);

	double diff=0;
	for(int i=0; i<this->length(); i++)
	{
		double tH=h.at(i);
		double tO=m.at(i);
		double tDiff=tH-tO;
		diff += tDiff*tDiff;
	}
	double avgDiff = sqrt(diff/this->length());
	return avgDiff;
}

template<typename T>
void DeviceMatrix<T>::save(ostream& f)
{
	HostMatrix<T> hm;
	hm.copyFrom(*this);
	hm.save(f);
}

template<typename T>
void DeviceMatrix<T>::save(const string& fileName)
{
	XLLib::dirPrepare4file(fileName);
	ofstream f(fileName.c_str());
	assert(f.good());
	this->save(f);
	f.close();
}

template<typename T>
void DeviceMatrix<T>::load(istream& f)
{
	HostMatrix<T> hm;
	hm.load(f);
	this->copyFrom(hm);
}


template<typename T>
void DeviceMatrix<T>::load(const string& fileName)
{
	ifstream f(fileName);
	load(f);
	f.close();
}

template<typename T>
void DeviceMatrix<T>::reshapeMatrix(size_t ni_, size_t nj_)
{
	assert(!this->frozen);
	assert(this->continuous());

	this->ni=ni_;
	this->stride=ni_;
	this->nj=nj_;
}


// Vector reduce.
template<typename T>
__global__
static void _vec_transform_reduce_sum(
		const T* v, T * result, const size_t dim, const size_t inc) {

	__shared__ T sdata[CU1DBLOCK];
	T tdata = 0.0;

	const size_t tid = threadIdx.x;
	const size_t vec_len = dim * inc;
	const size_t grid_stride = gridDim.x * blockDim.x * inc;
	size_t i = (blockIdx.x * blockDim.x + tid) * inc;

	// Grid reduce. Loop over the whole vector v.
	for (; i < vec_len; i += grid_stride)
	{
		tdata += v[i];
	}
	sdata[tid] = tdata;
	__syncthreads();

	// Tree reduce
# pragma unroll
	for (size_t shift = CU1DBLOCK / 2; shift > warpSize; shift >>= 1) {
		if (tid < shift)
		{
			sdata[tid] += sdata[tid + shift];
		}
		__syncthreads();
	}

	// Reduce last warp. Threads implicitly synchronized within a warp.
	if (tid < warpSize) {
		for (size_t shift = warpSize; shift > 0; shift >>= 1)
		{
			sdata[tid] += sdata[tid + shift];
		}
	}

	// Output to vector result.
	if (tid == 0)
	{
		result[blockIdx.x] = sdata[0];
	}
}

template<typename T>
T DeviceMatrix<T>::sum() const {

	size_t len=this->length();
	if (len == 0)
	{
		return 0.0;
	}


	T result;

	// Small vectors are copied to RAM and reduced on CPU.
	// The length is chosen by cu-vector-speed-test
	if (len < 4096)
	{
		HostMatrix<T> m;
		m.copyFrom(*this);
		result = m.sum();
	} else
	{
		// Use no more than 256 blocks (still too many?)
		size_t dimBlock = CU1DBLOCK;
		size_t dimGrid = ceil(len, dimBlock);
		if (dimGrid > 256) {
			dimGrid = 256;
		}
		DeviceMatrix<T> ans(dimGrid, 1);
		_vec_transform_reduce_sum<<<dimGrid, dimBlock>>>(this->data, ans.data, len, 1);
		checkError(cudaGetLastError());
		HostMatrix<T> ans_cpu;
		ans_cpu.copyFrom(ans);
		result = ans_cpu.sum();
	}

	return result;

}


template<typename T>
__global__
void DevMatReal_applyMin_kernel(T* data, size_t len, T minValue)
{
	size_t i=blockDim.x*blockIdx.x+threadIdx.x;
	if(i<len)
	{
		if(data[i]<minValue)
		{
			data[i]=minValue;
		}
	}
}


template<typename T>
void DeviceMatrix<T>::swap(DeviceMatrix<T> & mat)
{
	std::swap(mat.data, this->data);
	std::swap(mat.capacity, this->capacity);
	std::swap(mat.ni, this->ni);
	std::swap(mat.nj, this->nj);
	std::swap(mat.stride, this->stride);
}

template<typename T>
__global__
static void deviceMatrix_add_elements(T *data, MatrixDim dim, T alpha,
		MatrixElement<T>* x, size_t num_elements)
{
	size_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < num_elements)
	{
		MatrixElement<T>* e=x+i;
		data[e->j * dim.stride + e->i] += alpha * e->value;
	}

}

template<typename T>
template<typename D>
void DeviceMatrix<T>::convertFrom(const DeviceMatrix<D>&m , Precision scale)
{
	this->resize(m.ni, m.nj);
	if(this->continuous() && m.continuous())
	{
		DeviceMatrix_convertFrom_kernel<<<ceil(m.length(), blockSize), blockSize>>>(this->data, m.length(), m.data, scale);
	}
	else
	{
		dim3 gridDim;
		gridDim.x = ceil(this->ni, blockSize);
		gridDim.y = this->nj;
		DeviceMatrix_convertFrom_kernel <<< gridDim, blockSize >>> (this->data, this->ni, this->stride, this->nj,
				m.data, m.stride, scale);
	}
}

template void DeviceMatrix<float>::convertFrom(const DeviceMatrix<uchar>& m, Precision scale);
template void DeviceMatrix<double>::convertFrom(const DeviceMatrix<uchar>& m, Precision scale);

template class DeviceMatrix<double>;
template class DeviceMatrix<float>;
template class DeviceMatrix<int>;
template class DeviceMatrix<char>;
template class DeviceMatrix<uchar>;

} /* namespace cytonLib */
