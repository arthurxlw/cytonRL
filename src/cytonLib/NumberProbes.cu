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

#include "NumberProbes.h"

namespace cytonLib
{

template<typename T>
void NumberProbes<T>::init(int len_)
{
	vector<NumberProbe<T>>::resize(len_);
}

template<typename T>
void NumberProbes<T>::reset()
{
	for(int i=0; i<vector<NumberProbe<T>>::size(); i++)
	{
		vector<NumberProbe<T>>::at(i).reset();
	}
}

template<typename T>
void NumberProbes<T>::update(T* ts)
{
	for(int i=0; i<vector<NumberProbe<T>>::size(); i++)
	{
		T t=ts[i];
		vector<NumberProbe<T>>::at(i).update(t);
	}

}

template<typename T>
void NumberProbes<T>::update(vector<T>& t)
{
	assert(t.size()==vector<NumberProbe<T>>::size());
	this->update(&t[0]);
}

template<typename T>
string NumberProbes<T>::toString(bool detail)
{
	ostringstream os;
	for(int i=0; i<vector<NumberProbe<T>>::size(); i++)
	{
		if(i!=0)
		{
			os<<" ";
		}

		string str=vector<NumberProbe<T>>::at(i).toString(detail);
		os<<str;
	}
	return os.str();
}

template<typename T>
void NumberProbes<T>::update(int idx, T t)
{
	vector<NumberProbe<T>>::at(idx).update(t);
}

template class NumberProbes<float>;
template class NumberProbes<double>;

} /* namespace cytonLib */
