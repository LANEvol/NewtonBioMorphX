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

/* default */#include "../src/Typedefs.h"
/* default */#include "../src/Primitives.h"
/* default */#include "../src/DeviceDataPtr.h"
/* default */#include "../src/SVD3Cuda.h"
/* default */#include "cuda_runtime.h"

extern "C"
__global__ void compute_for_vis_nvrtc(DeviceDataPtr *data, Float t, int ntet) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < ntet) {

        int layer = data->layer[i] - 1;

        /* default */bool isRigid = false;

        if (isRigid)
            return;

        int a = data->tet[i].x;
        int b = data->tet[i].y;
        int c = data->tet[i].z;
        int d = data->tet[i].w;

        Vector Xa = data->posRef[a];
        Vector Xb = data->posRef[b];
        Vector Xc = data->posRef[c];
        Vector Xd = data->posRef[d];

        Vector xa = data->pos[a];
        Vector xb = data->pos[b];
        Vector xc = data->pos[c];
        Vector xd = data->pos[d];

        Vector Xab(Xa-Xb);
        Vector Xac(Xa-Xc);
        Vector Xad(Xa-Xd);

        Vector xab(xa-xb);
        Vector xac(xa-xc);
        Vector xad(xa-xd);
        Vector xbc(xb-xc);
        Vector xbd(xb-xd);

        Tensor dX(-Xab[0], -Xac[0], -Xad[0],
                  -Xab[1], -Xac[1], -Xad[1],
                  -Xab[2], -Xac[2], -Xad[2]);
        Tensor dx(-xab[0], -xac[0], -xad[0],
                  -xab[1], -xac[1], -xad[1],
                  -xab[2], -xac[2], -xad[2]);

        Tensor F = dx.dot(dX.inv());
        Tensor Rmat = data->R[i]; // Rotation matrix

/////////////////////////////////////////////////
        Vector tetCenterRef = (Xa+Xb+Xc+Xd) * 0.25;

        Float X = tetCenterRef[0];
        Float Y = tetCenterRef[1];
        Float Z = tetCenterRef[2];

        /* default *///Float R = 0.0;

        /* default *///Float Phi = 0.0;

        /* default *///Float Theta = 0.0;
        //////////////////////////////////////////////
        /* default */Float E = 1.0;
        /* default */Float nu = 0.4;
        /* default */Float visc = 0.0;
        /* default */Float yMin = 0.0;
        /* default */Float grRate1_Ref = 0.0;
        /* default */Float grRate2_Ref = 0.0;
        /* default */Float grRate3_Ref = 0.0;
        //////////////////////////////////////////////
        /* default */Tensor fiber1_Ref = Tensor(0.0);
        /* default */Tensor fiber2_Ref = Tensor(0.0);
        /* default */Tensor fiber3_Ref = Tensor(0.0);
        /* default */Tensor fiber4_Ref = Tensor(0.0);
        /* default */Tensor actin_Ref = Tensor(0.0);
        //////////////////////////////////////////////

        Float J = F.det();
        Tensor grRateTens1(grRate1_Ref*grRate1_Ref,0,0,0,0,0,0,0,0);
        Tensor grRateTens2(0,0,0,0,grRate2_Ref*grRate2_Ref,0,0,0,0);
        Tensor grRateTens3(0,0,0,0,0,0,0,0,grRate3_Ref*grRate3_Ref);

        data->grRate1[i] = F.dot(Rmat.dot(grRateTens1).dot(Rmat.trans())).dot(F.trans())*pow(J,Float(-2.0/3.0));
        data->grRate2[i] = F.dot(Rmat.dot(grRateTens2).dot(Rmat.trans())).dot(F.trans())*pow(J,Float(-2.0/3.0));
        data->grRate3[i] = F.dot(Rmat.dot(grRateTens3).dot(Rmat.trans())).dot(F.trans())*pow(J,Float(-2.0/3.0));
        data->fiber1[i] = F.dot(Rmat.dot(fiber1_Ref).dot(Rmat.trans())).dot(F.trans());
        data->fiber2[i] = F.dot(Rmat.dot(fiber2_Ref).dot(Rmat.trans())).dot(F.trans());
        data->fiber3[i] = F.dot(Rmat.dot(fiber3_Ref).dot(Rmat.trans())).dot(F.trans());
        data->fiber4[i] = F.dot(Rmat.dot(fiber4_Ref).dot(Rmat.trans())).dot(F.trans());
        data->actin[i] = F.dot(Rmat.dot(actin_Ref).dot(Rmat.trans())).dot(F.trans());
    }
}