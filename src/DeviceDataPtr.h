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

#ifndef LAGRANGIANSOLID_CUDA_DEVICEDATAPTR_H
#define LAGRANGIANSOLID_CUDA_DEVICEDATAPTR_H

#include "Primitives.h"

//namespace LagSol {
    struct DeviceDataPtr {

        Vector* pos;
        Vector* posRef;
        Vector* normalTetra;
        Float* dt;

        Tensor* vGrad;
        Tensor* vGradNode;

        Float* grRate1_Ref;
        Float* grRate2_Ref;
        Float* grRate3_Ref;
        Tensor* fiber1_Ref;
        Tensor* fiber2_Ref;
        Tensor* fiber3_Ref;
        Tensor* fiber4_Ref;
        Tensor* actin_Ref;


        Tensor* grRate1;
        Tensor* grRate2;
        Tensor* grRate3;
        Tensor* fiber1;
        Tensor* fiber2;
        Tensor* fiber3;
        Tensor* fiber4;
        Tensor* actin;


        Float* k1;
        Float* k2;

        int4* tet;
        int3* tri;
        int3* fac;

        int* isRigid;
        uint2* bcState;
        char* bidsXMin;
        char* bidsXMax;
        char* bidsYMin;
        char* bidsYMax;
        char* bidsZMin;
        char* bidsZMax;

        Vector* tempVec1;
        Vector* tempVec2;
        Vector* tempVec3;
        Vector* tempVec4;
        Tensor* tempTens1;
        Tensor* tempTens2;
        Tensor* tempTens3;
        Tensor* tempTens4;


        Vector* vel;
        Vector* force;
        Float* tetVol;
        Float* E;
        Float* nu;
        Float* plasticity;
        Float* hardening;
        Float* ENodal;
        Float* nuNodal;
        int* bdryFunction;
        Float* scalarFunction;
        Float* scalarFunctionNodal;
        Float* scalarActivityFunction;
        Float* scalarActivityFunctionNodal;
        Float* visc;
        Vector* G;
        Vector* lamInt;
        Tensor* R;
        Tensor* stress;
        Tensor* Fg;
        Tensor* Fp;
        int* layer;
        Float* nodalTag;
        Float* nodalR;
        Float* nodalG;
        Float* nodalB;
        Float* vol;
        Float* tetQual;
        Float* potEnergy;
        Float* kEnergy;
        Float* vonMises;
        Float* pressure;
        Float* plasticDeformation;
        Float* lamIntScalarNodal;

        int* boundaryNodeIds;
        int* boundaryNodeType;
        Vector* boundaryNodeNormal;
        Vector* boundaryNodePos;
        Vector* boundaryFacePos;
        Float* boundaryNodeR;
        Float* boundaryFaceR;

        Int* ngb_list;
        UInt* ngb_size;
        UInt* ngb_offset;
    };

//}

#endif //LAGRANGIANSOLID_CUDA_DEVICEDATAPTR_H