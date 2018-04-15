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

#include "Matrix.h"

namespace cytonLib {

template class MatrixElement<double>;
template class MatrixElement<float>;
template class MatrixElement<int>;


template class Matrix<double>;
template class Matrix<float>;
template class Matrix<int>;


void getBlockSizesForSimpleMatrixOperation(size_t num_rows,
                                           size_t num_cols,
                                           dim3 *dimGrid,
                                           dim3 *dimBlock)
{
  assert(num_rows > 0 && num_cols > 0);
  size_t col_blocksize = 64, row_blocksize = 4;
  while (col_blocksize > 1 &&
         (num_cols + (num_cols / 2) <= col_blocksize ||
          num_rows > 65535 * row_blocksize)) {
    col_blocksize /= 2;
    row_blocksize *= 2;
  }

  dimBlock->x = col_blocksize;
  dimBlock->y = row_blocksize;
  dimBlock->z = 1;
  dimGrid->x = ceil(num_cols, col_blocksize);
  dimGrid->y = ceil(num_rows, row_blocksize);
  assert(dimGrid->y <= 65535 &&
               "Matrix has too many rows to process");
  dimGrid->z = 1;
}


} /* namespace cytonLib */
