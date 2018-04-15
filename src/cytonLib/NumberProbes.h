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

#ifndef _CYTONLIB_NUMBERPROBES_H_
#define _CYTONLIB_NUMBERPROBES_H_

#include "basicHeads.h"
#include "NumberProbe.h"

namespace cytonLib
{

template<typename T>
class NumberProbes: public vector<NumberProbe<T>>
{
public:
	NumberProbes()
	{
	}

	void init(int len_);

	void reset();

	void update(T* t);

	void update(vector<T>& t);

	void update(int idx, T t);

	string toString(bool detail=false);
};

} /* namespace cytonLib */

#endif /* NUMBERPROBES_H_ */
