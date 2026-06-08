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

#ifndef LAGRANGIANSOLID_CUDA_DEVICEDATA_H
#define LAGRANGIANSOLID_CUDA_DEVICEDATA_H

#include "DataStructure.h"
#include "MeshIO.h"
#include "MortonKey.cuh"
#include "DeviceDataPtrManaged.cuh"

namespace LagSol {
    struct DeviceData {

        Encoder32* encoder;

        VectorArrayDev pos;
        VectorArrayDev posRef;
        VectorArrayDev normalTetra;

        ScalarArrayDev dt;

        TensorArrayDev vGrad;
        TensorArrayDev vGradNode;

        ScalarArrayDev grRate1_Ref;
        ScalarArrayDev grRate2_Ref;
        ScalarArrayDev grRate3_Ref;
        TensorArrayDev fiber1_Ref;
        TensorArrayDev fiber2_Ref;
        TensorArrayDev fiber3_Ref;
        TensorArrayDev fiber4_Ref;
        TensorArrayDev actin_Ref;

        TensorArrayDev grRate1;
        TensorArrayDev grRate2;
        TensorArrayDev grRate3;
        TensorArrayDev fiber1;
        TensorArrayDev fiber2;
        TensorArrayDev fiber3;
        TensorArrayDev fiber4;
        TensorArrayDev actin;

        ScalarArrayDev k1;
        ScalarArrayDev k2;

        Int4ArrayDev tet;
        Int3ArrayDev tri;
        Int3ArrayDev fac;

        IntArrayDev isRigid;
        UInt2ArrayDev bcState;
        CharArrayDev bidsXMax;
        CharArrayDev bidsYMax;
        CharArrayDev bidsZMax;
        CharArrayDev bidsXMin;
        CharArrayDev bidsYMin;
        CharArrayDev bidsZMin;

        VectorArrayDev tempVec1;
        VectorArrayDev tempVec2;
        VectorArrayDev tempVec3;
        VectorArrayDev tempVec4;
        TensorArrayDev tempTens1;
        TensorArrayDev tempTens2;
        TensorArrayDev tempTens3;
        TensorArrayDev tempTens4;

        VectorArrayDev vel;
        VectorArrayDev force;
        ScalarArrayDev tetVol;
        ScalarArrayDev E;
        ScalarArrayDev nu;
        ScalarArrayDev plasticity;
        ScalarArrayDev visc;
        TensorArrayDev R;
        TensorArrayDev Fg;
        TensorArrayDev Fp;
        TensorArrayDev stress;
        IntArrayDev layer;
        ScalarArrayDev vol;
        ScalarArrayDev tetQual;
        ScalarArrayDev potEnergy;
        ScalarArrayDev kEnergy;
        ScalarArrayDev vonMises;
        ScalarArrayDev pressure;


        UIntMArrayDev mKey;

        IntArrayDev boundaryNodeIds;
        IntArrayDev boundaryNodeType;
        VectorArrayDev boundaryNodeNormal;
        VectorArrayDev boundaryNodePos;
        VectorArrayDev boundaryFacePos;
        ScalarArrayDev boundaryNodeR;
        ScalarArrayDev boundaryFaceR;

        IntArrayDev ngb_list;
        UIntArrayDev ngb_size;
        UIntArrayDev ngb_offset;

        void init(Mesh& mesh);
        void set(DeviceDataPtrManaged& devicePtr);
        ~DeviceData();
    };
}

#endif //LAGRANGIANSOLID_CUDA_DEVICEDATA_H
