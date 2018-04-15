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

#include "NumberProbe.h"

namespace cytonLib
{

template<typename T>
void NumberProbe<T>::reset()
{
	num=0;
}

template<typename T>
void NumberProbe<T>::update(T t)
{
	if(num==0)
	{
		sum=t;
		min=t;
		max=t;
	}
	else
	{
		sum+=t;
		min=std::min(t, min);
		max=std::max(t, max);
	}
	num+=1;

}

template<typename T>
T NumberProbe<T>::getAverage()
{
	return sum/num;
}

template<typename T>
string NumberProbe<T>::toString(bool detail)
{
	if(num==0)
	{
		return string("empty");
	}
	else
	{
		if(detail)
		{
			return XLLib::stringFormat("%.2e(%d)/%.2e", sum/num, num, max);
		}
		else
		{
			return XLLib::stringFormat("%.2e(%d)", sum/num, num);
		}
	}
}


template class NumberProbe<float>;
template class NumberProbe<double>;

} /* namespace cytonLib */
