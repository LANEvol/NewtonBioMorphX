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


#include <immintrin.h>
#include "thrust/sequence.h"
#include "thrust/extrema.h"
#include <thrust/iterator/constant_iterator.h>
#include <thrust/sort.h>
#include "Octree.h"

namespace LagSol {

    __global__ void maskParticlesKernel(UIntM* __restrict__ output, const UIntM* input, const UIntM* mask, Int num) {
        Int idx = blockIdx.x*blockDim.x + threadIdx.x;
        if (idx < num)
            output[idx] = input[idx] & mask[idx];
    }

    __global__ void refineMaskKernel(UIntM* __restrict__ mask, Int* __restrict__ particleLevel, const Int* expNPBranch, Int threshold, Int num) {
        Int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < num) {
            if (expNPBranch[idx] > threshold) {
                mask[idx] |= (mask[idx]) >> (UIntM)3;
                particleLevel[idx]++;
            }
        }
    }

    __global__ void findBranchMinMaxKernel(Vector* __restrict__ bMinPos, Vector* __restrict__ bMaxPos, const Vector* pos, const Float* h, const Int* offset, const Int* nPBranch, Int nBranches) {
        Int idx = blockIdx.x*blockDim.x + threadIdx.x;
        if (idx < nBranches) {
            bMinPos[idx] = Vector(+MAX_FLOAT);
            bMaxPos[idx] = Vector(-MAX_FLOAT);
            for (Int pId = offset[idx]; pId < offset[idx] + nPBranch[idx]; pId++) {
                Vector pMin = pos[pId] - h[pId];
                Vector pMax = pos[pId] + h[pId];

                bMinPos[idx][0] = min(bMinPos[idx][0],pMin[0]);
                bMinPos[idx][1] = min(bMinPos[idx][1],pMin[1]);
                bMinPos[idx][2] = min(bMinPos[idx][2],pMin[2]);

                bMaxPos[idx][0] = max(bMaxPos[idx][0],pMax[0]);
                bMaxPos[idx][1] = max(bMaxPos[idx][1],pMax[1]);
                bMaxPos[idx][2] = max(bMaxPos[idx][2],pMax[2]);
            }
        }
    }

    Octree::Octree(const thrust::device_vector<Vector>& pos,
                    const thrust::device_vector<Float>& h,
                    const thrust::device_vector<UIntM>& mKey,
                    Int _threshold,
                    Int _maxLevel) {
        threshold = _threshold;
        maxLevel = _maxLevel;
        create(mKey);
        findBranchsMinMax(pos,h);
    };

    void Octree::clear() {
        bCode.clear();
        bLevel.clear();
        bProc.clear();
        bMinPos.clear();
        bMaxPos.clear();

        bPNum.clear();
        bPOffset.clear();
    };

    void Octree::create(const thrust::device_vector<UIntM>& mKey) {
        Int num = mKey.size();
        Int nBranches = 0;

        thrust::device_vector<UIntM> maskedKeys(num,0);
        thrust::device_vector<Int> particleLevel(num,0);
        thrust::device_vector<Int> expNPBranch(num,0);
        thrust::device_vector<UIntM> mask(num, _INITIAL_MASK);

        bCode = thrust::device_vector<UIntM>(num,0);
        bPNum = thrust::device_vector<Int>(num,0);
        int maxParticleInBranch = 0;
        for (Int level = 0; level < maxLevel; level++) {
            currentLevel = level;
            _LAUNCH(num,256,maskParticlesKernel)(thrust::raw_pointer_cast(maskedKeys.data()),thrust::raw_pointer_cast(mKey.data()),thrust::raw_pointer_cast(mask.data()), num);
            cudaDeviceSynchronize();

            auto new_end = thrust::reduce_by_key(maskedKeys.begin(), maskedKeys.end(),thrust::make_constant_iterator(1),bCode.begin(), bPNum.begin());

            nBranches = new_end.first - bCode.begin();

            if (nBranches==0)
                break;

            expand(bPNum.begin(), bPNum.begin() + nBranches, bPNum.begin(), expNPBranch.begin());

            maxParticleInBranch = (*thrust::max_element(bPNum.begin(), bPNum.begin() + nBranches));
            if (maxParticleInBranch <= threshold)
                break;
            _LAUNCH(num,256,refineMaskKernel)(thrust::raw_pointer_cast(mask.data()),thrust::raw_pointer_cast(particleLevel.data()),thrust::raw_pointer_cast(expNPBranch.data()),threshold, num);
            cudaDeviceSynchronize();
        }

//            std::cout << "Octree : (level, particles in branch) = ("<<currentLevel<<"/"<<maxLevel<<", "<< maxParticleInBranch<<"/"<<threshold<<")"<<std::endl;
        bCode.resize(nBranches);
        bPNum.resize(nBranches);
        bPOffset.resize(nBranches,0);
        bLevel.resize(nBranches);

        thrust::exclusive_scan(bPNum.begin(), bPNum.end(), bPOffset.begin());
        thrust::gather(bPOffset.begin(), bPOffset.end(), particleLevel.begin(), bLevel.begin());
    };

    void Octree::findBranchsMinMax(const thrust::device_vector<Vector>& pos, const thrust::device_vector<Float>& h){
        Int nBranches = bCode.size();
        bMinPos.resize(nBranches);
        bMaxPos.resize(nBranches);
        _LAUNCH(nBranches,64, findBranchMinMaxKernel)(
               thrust::raw_pointer_cast(bMinPos.data()),
               thrust::raw_pointer_cast(bMaxPos.data()),
               thrust::raw_pointer_cast(pos.data()),
               thrust::raw_pointer_cast(h.data()),
               thrust::raw_pointer_cast(bPOffset.data()),
               thrust::raw_pointer_cast(bPNum.data()),nBranches);
        cudaDeviceSynchronize();
    };

    __global__ void copyToNgbList(const Int* ngbSize, const Int* ngbOffset, const Int* localNgbIds,
                                  const Int* particles1, Int  nP1, Int  maxNgb, Int* __restrict__ ngbIds) {
        Int idx = blockIdx.x * blockDim.x + threadIdx.x;

        if (idx < nP1) {
            Int k = idx * maxNgb;
            thrust::copy(thrust::seq,localNgbIds + k, localNgbIds + k + ngbSize[idx], ngbIds + ngbOffset[idx]);
        }
    }


    // Pass 1: number of treeF branches whose bounding box intersects each treeN branch
    __global__ void countBranchNgbKernel(Int* __restrict__ ngbSize,
                                         const Vector* bMinPos1, const Vector* bMaxPos1,
                                         const Vector* bMinPos2, const Vector* bMaxPos2,
                                         Int nBranches1, Int nBranches2) {
        Int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < nBranches1) {
            Int n = 0;
            for (Int j = 0; j < nBranches2; j++)
                if (getIntersection(bMinPos1[idx], bMaxPos1[idx], bMinPos2[j], bMaxPos2[j]))
                    n++;
            ngbSize[idx] = n;
        }
    }

    // Pass 2: write the neighbour ids at each branch's offset (exact size, no limit)
    __global__ void fillBranchNgbKernel(Int* __restrict__ ngbIds, const Int* ngbOffset,
                                        const Vector* bMinPos1, const Vector* bMaxPos1,
                                        const Vector* bMinPos2, const Vector* bMaxPos2,
                                        Int nBranches1, Int nBranches2) {
        Int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < nBranches1) {
            Int k = ngbOffset[idx];
            for (Int j = 0; j < nBranches2; j++)
                if (getIntersection(bMinPos1[idx], bMaxPos1[idx], bMinPos2[j], bMaxPos2[j]))
                    ngbIds[k++] = j;
        }
    }

    void findBranchsNgb(const Octree& treeN,
                        const Octree& treeF,
                        thrust::device_vector<Int>& bNgbIds,      // ids of the treeF branches intersecting each treeN branch
                        thrust::device_vector<Int>& bNgbSize,     // number of neighbours of each branch
                        thrust::device_vector<Int>& bNgbOffset) { // exclusive sum of bNgbSize
        Int nBranches1 = treeN.bCode.size();
        Int nBranches2 = treeF.bCode.size();
        bNgbSize.assign(nBranches1, 0);
        bNgbOffset.assign(nBranches1, 0);
        bNgbIds.clear();
        if (nBranches1 == 0 || nBranches2 == 0)
            return;

        _LAUNCH(nBranches1, 512, countBranchNgbKernel)(
               thrust::raw_pointer_cast(bNgbSize.data()),
               thrust::raw_pointer_cast(treeN.bMinPos.data()), thrust::raw_pointer_cast(treeN.bMaxPos.data()),
               thrust::raw_pointer_cast(treeF.bMinPos.data()), thrust::raw_pointer_cast(treeF.bMaxPos.data()),
               nBranches1, nBranches2);
        cudaDeviceSynchronize();

        thrust::exclusive_scan(bNgbSize.begin(), bNgbSize.end(), bNgbOffset.begin());
        bNgbIds.resize(bNgbOffset.back() + bNgbSize.back());

        _LAUNCH(nBranches1, 512, fillBranchNgbKernel)(
               thrust::raw_pointer_cast(bNgbIds.data()),
               thrust::raw_pointer_cast(bNgbOffset.data()),
               thrust::raw_pointer_cast(treeN.bMinPos.data()), thrust::raw_pointer_cast(treeN.bMaxPos.data()),
               thrust::raw_pointer_cast(treeF.bMinPos.data()), thrust::raw_pointer_cast(treeF.bMaxPos.data()),
               nBranches1, nBranches2);
        cudaDeviceSynchronize();
    }
}