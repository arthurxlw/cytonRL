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
#ifndef _CYTONLIB_UTILS_H
#define _CYTONLIB_UTILS_H

#include "basicHeads.h"
#include "HostMatrix.h"
#include "DeviceMatrix.h"
#include "Global.h"


namespace cytonLib
{

void writeBinaryTag(FILE* file);

void readBinaryTag(FILE* file);

bool probeFileBinary(ifstream& f);

void setFileBinary(ostream& f, bool binary);

void applyMask(int* mask, Precision* mat, int dim0, int dim1, int dim2,
		bool transpose, Precision value);

void applyMask(int* mask, Precision* mat, int len, Precision value);

}

#endif
