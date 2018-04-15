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

#include "HostMatrix.h"
#include "DeviceMatrix.h"
#include "Global.h"
#include "utils.h"

namespace cytonLib {

template<typename T>
HostMatrix<T>::HostMatrix()
{
	this->capacity=0;
}


template<typename T>
HostMatrix<T>::HostMatrix(size_t ni_, size_t nj_)
{
	resize(ni_, nj_);
}

template<typename T>
void HostMatrix<T>::resize(size_t ni_, size_t nj_)
{
	if(this->ni==ni_ && this->nj==nj_)
	{
		return;
	}

	assert(this->capacity>0||this->empty());

	this->ni = ni_;
	this->stride= ni_;
	this->nj = nj_;

	size_t capacity_=this->ni*this->nj;
	if(this->capacity<capacity_)
	{
		if(this->capacity==0)
		{
			this->data=(T*)malloc(sizeof(T)*capacity_);
		}
		else
		{
			this->data=(T*)realloc(this->data, sizeof(T)*capacity_);
		}
		this->capacity=capacity_;
		assert(this->data!=NULL);
	}

}

template<typename T>
HostMatrix<T>::~HostMatrix()
{
	if(this->capacity>0)
	{
		assert(this->continuous());
		assert(this->data!=NULL);
		free(this->data);
	}
}

template<typename T>
HostMatrix<T> HostMatrix<T>::range(size_t i0, long int i1, size_t j0, long int j1)
{
	if(i1<0)
	{
		i1=this->ni;
	}
	if(j1<0)
	{
		j1=this->nj;
	}
	HostMatrix<T> res;
	res.set(i1-i0, this->stride, j1-j0, this->data+this->stride*j0+i0);
	return res;
}

template<typename T>
bool HostMatrix<T>::continuous() const
{
	return this->ni==this->stride;
}


template<typename T>
void HostMatrix<T>::setValue(T value)
{
	T* td=this->data;
	size_t len=this->length();
	for(size_t i=0;i<len;i++, td++)
	{
		*td=value;
	}
}


template<typename T>
T HostMatrix<T>::sum() const
{
	double res = 0;
	T* td=this->data;
	size_t len=this->length();
	for(size_t i=0; i<len; i++, td++)
	{
		res+=*td;
	}
	return res;
}

template<typename T>
void HostMatrix<T>::scale(T factor)
{
	T* td=this->data;
	size_t len=this->length();
	for(size_t i=0; i<len; i++, td++)
	{
		*td *= factor;
	}
}

template<typename T>
void HostMatrix<T>::copyFrom(const DeviceMatrix<T>& dm)
{
	this->resize(dm.ni, dm.nj);
	if(dm.continuous() && this->continuous())
	{
		cudaMemcpy(this->data, dm.data, dm.length()*sizeof(T),cudaMemcpyDeviceToHost);
	}
	else
	{
		for(size_t j=0; j<dm.nj; j++)
		{
			cudaMemcpy(this->data+this->stride*j, dm.data+dm.stride*j,
					dm.ni*sizeof(T), cudaMemcpyDeviceToHost);
		}
	}
}

template<typename T>
void HostMatrix<T>::copyFrom(const HostMatrix<T>& other)
{
	resize(other.ni, other.nj);
	setValue(other);
}

template<typename T>
void HostMatrix<T>::copyFrom(const T* data, int ni, int nj)
{
	resize(ni, nj);
	cudaMemcpy(this->data, data, ni*nj*sizeof(T),cudaMemcpyDefault);
}

template<typename T>
void HostMatrix<T>::save(ostream& f)
{
	assert(this->continuous());

	f<<"#matrix "<<this->ni<<" "<<this->nj<<"\n";

	T* id=this->data;

	for(size_t j=0; j<this->nj; j++)
	{
		for(size_t i=0; i<this->ni; i++)
		{
			if(i!=0)
			{
				f<<" ";
			}
			f<<*id;
			id++;
		}
		f<<"\n";
	}

	f<<"#end\n\n";
}


template<typename T>
void HostMatrix<T>::load(istream& f)
{
	assert(this->continuous());

	string tag;
	f>>tag;
	assert(tag=="#matrix");

	size_t ni, nj;
	f>>ni;
	f>>nj;

	this->resize(ni, nj);
	T* id=this->data;

	for(size_t j=0; j<nj; j++)
	{
		for(size_t i=0; i<ni; i++)
		{
			f>>*id;
			id++;
		}
	}

	f>>tag;
	assert(tag=="#end");
}

template<typename T>
void HostMatrix<T>::load(const string& fileName)
{
	if(!XLLib::fileExists(fileName))
	{
		fprintf(stderr, "file does not exist: %s. ", fileName.c_str());
		assert(false);
	}
	ifstream f(fileName.c_str());
	this->load(f);
	f.close();
}

template<class T>
T HostMatrix<T>::max() const
{
	T ans = - std::numeric_limits<T>::infinity();
	size_t i;
	size_t dim = this->length();
	for (i = 0; i + 4 <= dim; i += 4) {
		T a1 = this->data[i], a2 = this->data[i+1], a3 = this->data[i+2], a4 = this->data[i+3];
		if (a1 > ans || a2 > ans || a3 > ans || a4 > ans)
		{
			T b1 = (a1 > a2 ? a1 : a2);
			T b2 = (a3 > a4 ? a3 : a4);
			if (b1 > ans) ans = b1;
			if (b2 > ans) ans = b2;
		}
	}
	for (; i < dim; i++)
	{
		if (this->data[i] > ans)
		{
			ans = this->data[i];
		}
	}
	return ans;
}


template<class T>
T HostMatrix<T>::min() const
{
	T ans = std::numeric_limits<T>::infinity();
	T* data=this->data;
	size_t i;
	size_t dim = this->length();
	for (i = 0; i + 4 <= dim; i += 4) {
		T a1 = data[i], a2 = data[i+1], a3 = data[i+2], a4 = data[i+3];
		if (a1 < ans || a2 < ans || a3 < ans || a4 < ans)
		{
			T b1 = (a1 < a2 ? a1 : a2);
			T b2 = (a3 < a4 ? a3 : a4);
			if (b1 < ans) ans = b1;
			if (b2 < ans) ans = b2;
		}
	}
	for (; i < dim; i++)
	{
		if (data[i] < ans)
		{
			ans = data[i];
		}
	}
	return ans;
}


template<class T>
void HostMatrix<T>::setValue(const HostMatrix<T>& other)
{
	assert(this->ni==other.ni && this->nj==other.nj);
	if(this->continuous() && other.continuous())
	{
		size_t len=this->length();
		std::memcpy(this->data, other.data, sizeof(T)*len);
	}
	else
	{
		for(size_t j=0; j<this->nj; j++)
		{
			std::memcpy(this->data+this->stride*j, other.data+other.stride*j, sizeof(T)*this->ni);
		}
	}
}

template<class T>
void HostMatrix<T>::add(T c)
{
	size_t len=this->length();
	T*& data=this->data;
	for (size_t i = 0; i < len; i++)
	{
		data[i] += c;
	}
}


template<class T>
void HostMatrix<T>::setZero()
{
	if(this->continuous())
	{
		memset(this->data, 0, sizeof(T)*this->ni*this->nj);
	}
	else
	{
		for(size_t j=0; j<this->nj; j++)
		{
			memset(this->data+this->stride*j, 0, sizeof(T)*this->ni);
		}
	}
}

template<typename Real>
void HostMatrix<Real>::swap(HostMatrix<Real>& other)
{
  std::swap(this->data, other.data);
  std::swap(this->ni, other.ni);
  std::swap(this->stride, other.stride);
  std::swap(this->nj, other.nj);
  std::swap(this->capacity, other.capacity);
}

template class HostMatrix<float>;
template class HostMatrix<double>;
template class HostMatrix<int>;
template class HostMatrix<char>;
template class HostMatrix<uchar>;

} /* namespace cytonLib */
