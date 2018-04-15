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

namespace cytonLib
{

template<typename T>
HostMatReal<T> HostMatReal<T>::range(size_t i0, long int i1, size_t j0, long int j1)
{
	HostMatReal<T> res;
	if(i1<0)
	{
		i1=this->ni;
	}
	if(j1<0)
	{
		j1=this->nj;
	}
	res.set(i1-i0, this->stride, j1-j0, this->data+this->stride*j0+i0);
	return res;
}


template<typename T>
T HostMatReal<T>::dotProduct(const HostMatReal<T>& o)
{
	size_t len=this->length();
	assert(len==o.length());
	T res=0;
	assert(false);
	return res;
}

template<typename T>
void HostMatReal<T>::add(T alpha, HostMatReal<T> other)
{
	assert(this->ni==other.ni && this->nj==other.nj);
	for(size_t i=0; i<this->ni; i++)
	{
		for(size_t j=0; j<this->nj; j++)
		{
			this->at(i, j) += alpha*other.at(i,j);
		}
	}
}

template<typename T>
void HostMatReal<T>::add(T alpha)
{
	for(size_t j=0; j<this->nj; j++)
	{
		for(size_t i=0; i<this->ni; i++)
		{
			this->at(i, j) += alpha;
		}
	}
}


template class HostMatReal<double>;
template class HostMatReal<float>;

} /* namespace cytonLib */
