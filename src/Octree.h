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


#ifndef LAGRANGIANSOLID_CUDA_OCTREE_CUH
#define LAGRANGIANSOLID_CUDA_OCTREE_CUH

#include "Typedefs.h"
#include "MortonKey.cuh"
#include "Kernels.h"
#include "Expand.h"
#include <thrust/device_vector.h>

namespace LagSol {

    __global__ void maskParticlesKernel(UIntM* __restrict__ output, const UIntM* input, const UIntM* mask, Int num);
    __global__ void refineMaskKernel(UIntM* __restrict__ mask, Int* __restrict__ particleLevel, const Int* expNPBranch, Int threshold, Int num);
    __global__ void findBranchMinMaxKernel(Vector* __restrict__ bMinPos, Vector* __restrict__ bMaxPos, const Vector* pos, const Float* h, const Int* offset, const Int* nPBranch, Int nBranches);
    inline __host__ __device__ bool getIntersection(const Vector& aPMin, const Vector& aPMax, const Vector& bPMin, const Vector& bPMax) {
        if (aPMax[0] < bPMin[0]) return false; // a is left of b
        if (aPMin[0] > bPMax[0]) return false; // a is right of b
        if (aPMax[1] < bPMin[1]) return false; // a is above b
        if (aPMin[1] > bPMax[1]) return false; // a is below b
        if (aPMax[2] < bPMin[2]) return false; // a is front of b
        if (aPMin[2] > bPMax[2]) return false; // a is behind of b
        return true; // boxes overlap
    }
    __global__ void findBranchNgbKernel(Int* __restrict__ ngbIds,
                                        Int* __restrict__ ngbSize,
                                        const Vector* bMinPos1,
                                        const Vector* bMaxPos1,
                                        const Vector* bMinPos2,
                                        const Vector* bMaxPos2,
                                        UInt maxBranchNgb,
                                        Int nBranches1,
                                        Int nBranches2);
    __global__ void copyToNgbVector(Int* __restrict__ output, const Int* ngbIds, const Int* ngbSize, const Int* ngbOffset, UInt maxBranchNgb, Int nBranches);

    struct Octree {
        Int threshold; // This value indicates the maximum allowed number of particles in each branch.
        Int maxLevel;  // This value indicates the maximum level of resolution of the octree.
        Int currentLevel;

        thrust::device_vector<UIntM> bCode;      // This array contains Morton keys of the branches.
        thrust::device_vector<Int> bLevel;       // This array contains the resolution level of the branches.
        thrust::device_vector<Int> bProc;        // This array contains the branches processor's rank. Each branch is located in a processor.
        thrust::device_vector<Vector> bMinPos;   // This array contains the lowest  (000) corner of the branches' bounding volume. Bounding volume is determined by the radius of particles included in the branch.
        thrust::device_vector<Vector> bMaxPos;   // This array contains the highest (111) corner of the branches' bounding volume. Bounding volume is determined by the radius of particles included in the branch.

        thrust::device_vector<Int> bPNum;        // This array contains the number of particles included in each branch.
        thrust::device_vector<Int> bPOffset;     // This array contains the exclusive summation of the branches particles number.

        Octree(const thrust::device_vector<Vector>& pos,
               const thrust::device_vector<Float>& h,
               const thrust::device_vector<UIntM>& mKey,
               Int _threshold,
               Int _maxLevel);
        void clear();
        void create(const thrust::device_vector<UIntM>& mKey);
        void findBranchsMinMax(const thrust::device_vector<Vector>& pos, const thrust::device_vector<Float>& h);
//        void findBranchsNgb(UInt maxBranchNgb);
    };

    __global__ void copyToNgbList(const Int* ngbSize, const Int* ngbOffset, const Int* localNgbIds,
                                  const Int* particles1, Int  nP1, Int  maxNgb, Int* __restrict__ ngbIds);
    void findBranchsNgb(const Octree& treeN,
                        const Octree& treeF,
                        thrust::device_vector<Int>& bNgbIds,      // This array contains the index of the branches in treeF who intersect with branches in treeN.
                        thrust::device_vector<Int>& bNgbSize,     // This array contains the the number of neighbors for each branch.
                        thrust::device_vector<Int>& bNgbOffset);  // This array contains the exclusive summation of the neighbors number for each branch.
}


#endif //LAGRANGIANSOLID_CUDA_OCTREE_CUH
