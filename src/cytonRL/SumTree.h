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

#ifndef _CYTONRL_SUMTREE_H_
#define _CYTONRL_SUMTREE_H_

#include "basicHeadsRl.h"

namespace cytonRl
{

class SumTree
{
public:
	vector<double> data;
	int n;

	void init(int capacity);

	void set(int dataIndex, double priority);

	int retrieve(double value, int validLen);

	void sample(int batchSize, vector<int>& indexes, int validLen, Precision beta, vector<Precision>& weights);
};

} /* namespace cytonRl */

#endif /* SUMTREE_H_ */
