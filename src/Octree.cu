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

    __global__ void findBranchNgbKernel(Int* __restrict__ ngbIds,
                                        Int* __restrict__ ngbSize,
                                        const Vector* bMinPos1,
                                        const Vector* bMaxPos1,
                                        const Vector* bMinPos2,
                                        const Vector* bMaxPos2,
                                        UInt maxBranchNgb,
                                        Int nBranches1,
                                        Int nBranches2) {
        Int idx = blockIdx.x*blockDim.x + threadIdx.x;
        if (idx < nBranches1) {
            ngbSize[idx] = 0;
            for (Int j = 0; j < nBranches2; j++)
                if (getIntersection(bMinPos1[idx],bMaxPos1[idx],
                                    bMinPos2[j  ],bMaxPos2[j  ])) {
                    if (ngbSize[idx] < maxBranchNgb) {
                        ngbIds[ngbSize[idx] + idx * maxBranchNgb] = j;
//                  } else {
//                        printf("maxBranchNgb exceeds! %d %d %d %d \n",ngbSize[idx],nBranches1,idx,j);
                    }
                    ngbSize[idx]++;
                }
        }
    }


    __global__ void copyToNgbVector(Int* __restrict__ output, const Int* ngbIds, const Int* ngbSize, const Int* ngbOffset, UInt maxBranchNgb, Int nBranches) {
        Int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < nBranches) {
            Int k = idx * maxBranchNgb;
            thrust::copy(thrust::seq,ngbIds + k, ngbIds + k + ngbSize[idx], output + ngbOffset[idx]);
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

        // std::cout<<"------------------------------------------"<<std::endl;
        // for (int i=0; i<bCode.size(); i++) {
        //     Vector minPosTemp(bMinPos[i]);
        //     Vector maxPosTemp(bMaxPos[i]);
        //
        //     std::cout<<i<<","<<bPNum[i]<<","<<bLevel[i]<< std::endl;
        //     std::cout<<" " << minPosTemp.x<<" "<< minPosTemp.y<<" "<< minPosTemp.z<<" "<<" : "<<std::endl;
        //     std::cout<<" " << maxPosTemp.x<<" "<< maxPosTemp.y<<" "<< maxPosTemp.z<<" "<<" : ";
        //     // for (int j=0; j<bNgbSize[i]; j++)
        //     //     std::cout<<bNgbIds[bNgbOffset[i]+j]<<" ";
        //     std::cout<<std::endl;
        //     std::cout<<std::endl;
        // }
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

    void findBranchsNgb(const Octree& treeN,
                        const Octree& treeF,
                        thrust::device_vector<Int>& bNgbIds,      // This array contains the index of the branches in treeF who intersect with branches in treeN.
                        thrust::device_vector<Int>& bNgbSize,     // This array contains the the number of neighbors for each branch.
                        thrust::device_vector<Int>& bNgbOffset) { // This array contains the exclusive summation of the neighbors number for each branch.

        int maxBranchNgb = 500;
        Int nBranches = treeN.bCode.size();
        bNgbIds.resize(nBranches * maxBranchNgb);
        bNgbSize.resize(nBranches,0);
        if (nBranches==0)
            return;
        _LAUNCH(nBranches,512,findBranchNgbKernel)(
               thrust::raw_pointer_cast(bNgbIds.data()),
               thrust::raw_pointer_cast(bNgbSize.data()),
               thrust::raw_pointer_cast(treeN.bMinPos.data()),
               thrust::raw_pointer_cast(treeN.bMaxPos.data()),
               thrust::raw_pointer_cast(treeF.bMinPos.data()),
               thrust::raw_pointer_cast(treeF.bMaxPos.data()),
               maxBranchNgb,
               treeN.bCode.size(),
               treeF.bCode.size());
        
        cudaDeviceSynchronize();

        bNgbOffset.resize(nBranches);
        thrust::exclusive_scan(bNgbSize.begin(),bNgbSize.end(),bNgbOffset.begin());

        thrust::device_vector<Int> tempIds(bNgbOffset.back() + bNgbSize.back());

        _LAUNCH(nBranches,512,copyToNgbVector)(
                thrust::raw_pointer_cast(tempIds.data()),
                thrust::raw_pointer_cast(bNgbIds.data()),
                thrust::raw_pointer_cast(bNgbSize.data()),
                thrust::raw_pointer_cast(bNgbOffset.data()),
                maxBranchNgb, nBranches);
        cudaDeviceSynchronize();
        bNgbIds = tempIds;
    };

/*
    bool octreeSearch(DeviceData& data, const Vector& _minPos, const Vector& _maxPos, const Float& charLimit) {
//        const Encoder32 encoder(Vector(-12.0), Vector(12.0));
        if (!std::isfinite((_maxPos-_minPos).norm1()) || ((_maxPos-_minPos).norm1()>charLimit)) {
//            printf("Warning, the domains' limits are not finite ! \n");
            return false;
        }

        const Encoder32 encoder(_minPos, _maxPos);

        thrust::device_vector<UIntM> mKeyN(data.boundaryNodePos.size(),0);
        thrust::device_vector<UIntM> mKeyF(data.boundaryFacePos.size(),0);
        thrust::transform(data.boundaryNodePos.begin(), data.boundaryNodePos.end(), mKeyN.begin(),encoder);
        thrust::transform(data.boundaryFacePos.begin(), data.boundaryFacePos.end(), mKeyF.begin(),encoder);

        thrust::device_vector<int> seqIdN(data.boundaryNodePos.size());
        thrust::device_vector<int> seqIdF(data.boundaryFacePos.size());
        thrust::sequence(seqIdN.begin(),seqIdN.end());
        thrust::sequence(seqIdF.begin(),seqIdF.end());
        thrust::sort_by_key(mKeyN.begin(),mKeyN.end(), seqIdN.begin());
        thrust::sort_by_key(mKeyF.begin(),mKeyF.end(), seqIdF.begin());

        _REMAP(seqIdN,data.boundaryNodePos);
        _REMAP(seqIdN,data.boundaryNodeR);
        _REMAP(seqIdN,data.boundaryNodeIds);

        _REMAP(seqIdF,data.boundaryFacePos);
        _REMAP(seqIdF,data.boundaryFaceR);
        _REMAP(seqIdF,data.tri);

        Octree treeF(data.boundaryFacePos,data.boundaryFaceR,mKeyF,1000,15);
        Octree treeN(data.boundaryNodePos,data.boundaryNodeR,mKeyN,1000,15);




        std::cout<<"------------------------------------------"<<std::endl;
        for (int i=0; i<treeN.bCode.size(); i++) {
            Vector minPosTemp(treeN.bMinPos[i]);
            Vector maxPosTemp(treeN.bMaxPos[i]);

            std::cout<<i<<" "<<treeN.bPNum[i]<< std::endl;
            std::cout<<" " << minPosTemp.x<<" "<< minPosTemp.y<<" "<< minPosTemp.z<<" "<<" : "<<std::endl;
            std::cout<<" " << maxPosTemp.x<<" "<< maxPosTemp.y<<" "<< maxPosTemp.z<<" "<<" : ";
            // for (int j=0; j<bNgbSize[i]; j++)
            //     std::cout<<bNgbIds[bNgbOffset[i]+j]<<" ";
            std::cout<<std::endl;
            std::cout<<std::endl;
        }




        thrust::device_vector<Int> bNgbIds;    // This array contains the index of the branches neighbors. Branches are recognized as neighbor if their bounding volume intersects.
        thrust::device_vector<Int> bNgbSize;   // This array contains the the number of neighbors for each branch.
        thrust::device_vector<Int> bNgbOffset; // This array contains the exclusive summation of the neighbors number for each branch.
        findBranchsNgb(treeN, treeF,bNgbIds,bNgbSize,bNgbOffset);
//        std::cout<<treeF.bCode.size()<<" "<<treeN.bCode.size()<<std::endl;

        Int nBranches = treeN.bCode.size();
        Int num = data.boundaryNodePos.size();

        size_t nP1Max = (*thrust::max_element(treeN.bPNum.begin(), treeN.bPNum.end()));
        size_t nP2Max = (*thrust::max_element(treeF.bPNum.begin(), treeF.bPNum.end())) * (*thrust::max_element(bNgbSize.begin(), bNgbSize.end()));
        nP2Max = std::max(nP2Max, (nP1Max*MAX_NGB));

        Float allocMemMB = (4.0*(nP1Max*nP2Max+3.0*nP1Max+nP2Max) + 8.0*(nP1Max+nP2Max))*1e-6*1.25;
        if (allocMemMB > 12000.0){
//            int a = (*thrust::max_element(bNgbSize.begin(), bNgbSize.end()));
//            std::cout<<"Domain dims : "<<(_maxPos-_minPos).norm1()<<" "<<charLimit<<std::endl;
//            std::cout<<"Node tree : "<<treeN.currentLevel<<" "<<treeN.bPNum.size()<<" "<<nP1Max<<std::endl;
//            std::cout<<"Face tree : "<<treeF.currentLevel<<" "<<treeF.bPNum.size()<<" "<<a<<std::endl;
//            printf("Warning, allocating about %.0f MB for octreeSearch! \n",allocMemMB);
            return false;
        }

        thrust::device_vector<Int> localNgbIds(nP1Max*nP2Max,-1);
        thrust::device_vector<Int> particles1(nP1Max);
        thrust::device_vector<Int> particles2(nP2Max);

        thrust::device_vector<Int> localNgbSize(nP1Max);
        thrust::device_vector<Int> localNgbOffset(nP1Max);

        thrust::device_vector<Vector> posP1(nP1Max);
        thrust::device_vector<Vector> posP2(nP2Max);
        thrust::device_vector<Float>  hP1(nP1Max);
        thrust::device_vector<Float>  hP2(nP2Max);

        thrust::host_vector<Int> hBNgbSize(bNgbSize);
        thrust::host_vector<Int> hBNgbIds(bNgbIds);
        thrust::host_vector<Int> hBNgbOffset(bNgbOffset);
//        thrust::host_vector<Int> hBCode(treeN.bCode);
        thrust::host_vector<Int> hBPOffsetN(treeN.bPOffset);
        thrust::host_vector<Int> hBPNumN(treeN.bPNum);
        thrust::host_vector<Int> hBPOffsetF(treeF.bPOffset);
        thrust::host_vector<Int> hBPNumF(treeF.bPNum);

        data.ngb_list.resize(num * MAX_NGB);
        data.ngb_size.resize(num);
        data.ngb_offset.resize(num);

        size_t ngbListSize = 0;

        Int temp1 = 0;
        Int temp2 = 0;

        for (Int branch = 0; branch < nBranches; branch++) {
            Int offset = hBPOffsetN[branch];
            Int nP1 = hBPNumN[branch];
            Int bNgbSize = hBNgbSize[branch];
            Int bNgbOffset = hBNgbOffset[branch];

            thrust::sequence(particles1.begin(),particles1.begin() + nP1, offset);

            Int nP2 = 0;

            for (Int i = 0; i < bNgbSize; i++) {
                Int bId = hBNgbIds[bNgbOffset + i];
                Int offset2 = hBPOffsetF[bId];
                thrust::sequence(particles2.begin() + nP2, particles2.begin() + nP2 + hBPNumF[bId], offset2);
                nP2 += hBPNumF[bId];
            }

            thrust::fill_n(localNgbSize.begin(), nP1, 0);
            thrust::fill_n(localNgbIds.begin(), nP1 * MAX_NGB, num+1);

            thrust::gather(particles1.begin(), particles1.begin() + nP1, data.boundaryNodePos.begin(), posP1.begin());
            thrust::gather(particles2.begin(), particles2.begin() + nP2, data.boundaryFacePos.begin(), posP2.begin());
            thrust::gather(particles1.begin(), particles1.begin() + nP1, data.boundaryNodeR.begin(), hP1.begin());
            thrust::gather(particles2.begin(), particles2.begin() + nP2, data.boundaryFaceR.begin(), hP2.begin());

            _LUNCH(nP2,64,distCheckKernel)(thrust::raw_pointer_cast(posP1.data()),
                    thrust::raw_pointer_cast(posP2.data()),
                    thrust::raw_pointer_cast(hP1.data()),
                    thrust::raw_pointer_cast(hP2.data()),
                    nP1, nP2, MAX_NGB,
                    thrust::raw_pointer_cast(particles2.data()),
                    thrust::raw_pointer_cast(localNgbSize.data()),
                    thrust::raw_pointer_cast(localNgbIds.data()));

            cudaDeviceSynchronize();

            thrust::exclusive_scan(localNgbSize.begin(),localNgbSize.begin() + nP1, localNgbOffset.begin());
            size_t sz = localNgbOffset[nP1-1] + localNgbSize[nP1-1];
            ngbListSize += sz;

            _LUNCH(nP1,64,copyToNgbList)(thrust::raw_pointer_cast(localNgbSize.data()),
                    thrust::raw_pointer_cast(localNgbOffset.data()),
                    thrust::raw_pointer_cast(localNgbIds.data()),
                    thrust::raw_pointer_cast(particles1.data()),
                    nP1,MAX_NGB,
                    thrust::raw_pointer_cast(&data.ngb_list[ngbListSize - sz]));
            cudaDeviceSynchronize();

            thrust::copy(localNgbSize.begin(), localNgbSize.begin() + nP1, data.ngb_size.begin() + temp1);

            temp1+=nP1;
            temp2+=nP2;
        }

        data.ngb_list.resize(ngbListSize);
        thrust::exclusive_scan(data.ngb_size.begin(),data.ngb_size.end(), data.ngb_offset.begin());

        if (temp1 != num)
            _ERROR_MESSAGE("Search error "<<temp1<<" != "<<num);
        int maxNgbTemp = *thrust::max_element(data.ngb_size.begin(),data.ngb_size.end());
        if (maxNgbTemp > MAX_NGB)
            _ERROR_MESSAGE("ngbSize, "<<maxNgbTemp<<", exceeds "<<MAX_NGB);

        cudaDeviceSynchronize();
//	std::cout<<"<<<<<<<<<<<<<<<<<<<<< "<<temp1<<" <<<< "<<temp2<<"    "<<nBranches<<" "<<ngbListSize<<std::endl;
        return true;
    }
*/

}