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

#ifndef _CYTONRL_REPLAYMEMORY_H_
#define _CYTONRL_REPLAYMEMORY_H_

#include "basicHeadsRl.h"
#include "HostMatReal.h"
#include "DevMatReal.h"
#include "Steps.h"

namespace cytonRl
{

class ReplayMemory
{
public:
	DevMatUchar buffer;
	int height;
	int width;
	int inputFrames;
	int capacity;
	int len;

	Steps steps;

	void init(int height_, int width_, int inputFrames_, int capacity_);

	void addImage(int index, HostMatUchar& mat);

	void getInput(int index, DevMatPrec& mat);

	void getImage(int index, DevMatPrec& mat);

	Step& getStep(int index);

};

} /* namespace cytonRl */

#endif /* IMAGEBUFFER_H_ */
