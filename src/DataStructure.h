// Copyright (C) 2026 Ebrahim Jahanbakhsh & Michel Milinkovitch
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#ifndef LAGRANGIANSOLID_DATASTRUCTURE_H
#define LAGRANGIANSOLID_DATASTRUCTURE_H
#include "thrust/device_vector.h"
#include "thrust/host_vector.h"
#include "Typedefs.h"
#include "Eigen/Core"
#include "Primitives.h"

namespace LagSol {
    typedef thrust::device_vector<Float> ScalarArrayDev;
    typedef thrust::device_vector<Vector> VectorArrayDev;
    typedef thrust::device_vector<Tensor> TensorArrayDev;
    typedef thrust::device_vector<int4> Int4ArrayDev;
    typedef thrust::device_vector<int3> Int3ArrayDev;
    typedef thrust::device_vector<int2> Int2ArrayDev;
    typedef thrust::device_vector<uint2> UInt2ArrayDev;
    typedef thrust::device_vector<char> CharArrayDev;
    typedef thrust::device_vector<int> IntArrayDev;
    typedef thrust::device_vector<unsigned int> UIntArrayDev;
    typedef thrust::device_vector<UIntM> UIntMArrayDev;

    typedef thrust::host_vector<Float> ScalarArrayHost;
    typedef thrust::host_vector<Vector> VectorArrayHost;
    typedef thrust::host_vector<Tensor> TensorArrayHost;
    typedef thrust::host_vector<int4> Int4ArrayHost;
    typedef thrust::host_vector<int3> Int3ArrayHost;
    typedef thrust::host_vector<int2> Int2ArrayHost;
    typedef thrust::host_vector<char> CharArrayHost;
    typedef thrust::host_vector<int> IntArrayHost;
    typedef thrust::host_vector<unsigned int> UIntArrayHost;
    typedef thrust::host_vector<UIntM> UIntMArrayHost;

    typedef Eigen::Array<Float,Eigen::Dynamic,1> ScalarArrayEigen;
    typedef Eigen::Array<Float,Eigen::Dynamic,3,Eigen::RowMajor> VectorArrayEigen;
    typedef Eigen::Array<Float,Eigen::Dynamic,9,Eigen::RowMajor> TensorArrayEigen;



    struct is_not_a_finite_Vector {
        Float limit;
        __host__ __device__ is_not_a_finite_Vector() : limit(INFINITY) {}
        __host__ __device__ is_not_a_finite_Vector(const Float& _lim) : limit(_lim) {}
        __host__ __device__ bool operator()(const Vector& a) const {
            return !(isfinite(a[0]) && isfinite(a.data[1]) && isfinite(a[2]) && a.mag()<limit);
        }
    };

}


#endif //LAGRANGIANSOLID_DATASTRUCTURE_H
