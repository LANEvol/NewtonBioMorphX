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

__device__ Tensor get_fiber_stress(const Tensor &aaT, const Tensor &aaTRef, const Float& J, const Float &k1, const Float &k2) {
    Float Ifm1 = max(aaT.trace() - aaTRef.trace(), Float(0.0));
    Float expTerm = exp(k2 * Ifm1 * Ifm1);
    return Float(2.0) / J * k1 * Ifm1 * expTerm * aaT;
}

extern "C"
__global__ void compute_force_nvrtc(DeviceDataPtr *data, Float dt, Float t, int ntet) {
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

        Vector vab(data->vel[a]-data->vel[b]);
        Vector vac(data->vel[a]-data->vel[c]);
        Vector vad(data->vel[a]-data->vel[d]);

        Vector areaa(xbd.cross(xbc) * 0.5f);
        Vector areab(xac.cross(xad) * 0.5f);
        Vector areac(xad.cross(xab) * 0.5f);
        Vector aread(xab.cross(xac) * 0.5f);

        Tensor dX(-Xab[0], -Xac[0], -Xad[0],
                  -Xab[1], -Xac[1], -Xad[1],
                  -Xab[2], -Xac[2], -Xad[2]);
        Tensor dx(-xab[0], -xac[0], -xad[0],
                  -xab[1], -xac[1], -xad[1],
                  -xab[2], -xac[2], -xad[2]);
        Tensor dv(-vab[0], -vac[0], -vad[0],
                  -vab[1], -vac[1], -vad[1],
                  -vab[2], -vac[2], -vad[2]);

        Tensor F = dx.dot(dX.inv());

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

        Tensor Rmat = data->R[i]; // Rotation matrix
        Vector growthRate(grRate1_Ref,grRate2_Ref,grRate3_Ref);
        data->Fg[i] = data->Fg[i] + Rmat.dot(Tensor::diag(growthRate)).dot(Rmat.trans()) * dt;

        Tensor Fg = data->Fg[i];
        Tensor Fe = F.dot(Fg.inv());

        if (yMin>0.0) {
            Tensor Fep = Fe;
            Tensor Fp = data->Fp[i];
            Tensor F_trial = Fep.dot(Fp.inv());
            Float yMax = 1.0/(yMin + 1e-10*(yMin==0.0));

            Tensor U,sigma,V;
            svd(F_trial,U,sigma,V);

            Tensor sigma_e(sigma), sigma_p(0.0);
            for (int j=0;j<1;j++) {
                sigma_e[0] = min(max(sigma_e[0], yMin), yMax);
                sigma_e[4] = min(max(sigma_e[4], yMin), yMax);
                sigma_e[8] = min(max(sigma_e[8], yMin), yMax);
                sigma_e = sigma_e * pow((sigma[0]*sigma[4]*sigma[8]) / (sigma_e[0]*sigma_e[4]*sigma_e[8]), Float(1.0/3.0));
            }
            sigma_p[0] = sigma[0] / sigma_e[0];
            sigma_p[4] = sigma[4] / sigma_e[4];
            sigma_p[8] = sigma[8] / sigma_e[8];

            Fe = U.dot(sigma_e).dot(V.trans());
            Tensor dFp = V.dot(sigma_p).dot(V.trans());
            data->Fp[i] = dFp.dot(data->Fp[i]);
        }

        Float Je = Fe.det();
        Float Jg = Fg.det();

        Tensor velGrad = dv.dot(dx.inv());
        Tensor Be = Fe.dot(Fe.trans());
        Float I1 = Be.trace();
        Float K = E / (3.0 - 6.0 * nu); // Bulk modulus
        Float mu = E / (2.0 + 2.0 * nu); // Shear modulus

        Float pressure = -K * (1.0 - 1.0/Je) / Jg;
        Tensor S = mu * Be.dev() * pow(Je, Float(-2.0 / 3.0)) / (Je * Jg) - Tensor::eye() * pressure;

        data->vonMises[i] = sqrt(3.0/2.0*S.dev().norm2());
        data->pressure[i] = pressure;
        data->potEnergy[i] = (mu * (Be.trace() * pow(Je, Float(-2.0 / 3.0)) - 3.0) + K * (Je - log(Je)- 1.0)) * data->tetVol[i] +
                            0.5 * (data->vel[a]+data->vel[b]+data->vel[c]+data->vel[d]).mag2()/16.0 * data->tetVol[i];

        //////////////////////////////////////////////
        /* default */Float k1 = 0.0; // fiber stiffness
        /* default */Float k2 = 0.0; // fiber parameter
        /* default */Tensor fiber1_Ref = Tensor(0.0);
        /* default */Tensor fiber2_Ref = Tensor(0.0);
        /* default */Tensor fiber3_Ref = Tensor(0.0);
        /* default */Tensor fiber4_Ref = Tensor(0.0);
        /* default */Tensor actin_Ref = Tensor(0.0);
        //////////////////////////////////////////////

        fiber1_Ref = Rmat.dot(fiber1_Ref).dot(Rmat.trans());
        fiber2_Ref = Rmat.dot(fiber2_Ref).dot(Rmat.trans());
        fiber3_Ref = Rmat.dot(fiber3_Ref).dot(Rmat.trans());
        fiber4_Ref = Rmat.dot(fiber4_Ref).dot(Rmat.trans());
        actin_Ref = Rmat.dot(actin_Ref).dot(Rmat.trans());

        Float J = F.det();

        if (k1>0 && fiber1_Ref.norm2()>0.0) S = S + get_fiber_stress(F.dot(fiber1_Ref).dot(F.trans()), fiber1_Ref, J, k1, k2);
        if (k1>0 && fiber2_Ref.norm2()>0.0) S = S + get_fiber_stress(F.dot(fiber2_Ref).dot(F.trans()), fiber2_Ref, J, k1, k2);
        if (k1>0 && fiber3_Ref.norm2()>0.0) S = S + get_fiber_stress(F.dot(fiber3_Ref).dot(F.trans()), fiber3_Ref, J, k1, k2);
        if (k1>0 && fiber4_Ref.norm2()>0.0) S = S + get_fiber_stress(F.dot(fiber4_Ref).dot(F.trans()), fiber4_Ref, J, k1, k2);

        if (actin_Ref.norm2()>0.0) {
            S = S + F.dot(actin_Ref).dot(F.trans());
        }

        S = S + (velGrad+velGrad.trans()) * 0.5 * visc + Tensor::eye()*velGrad.trace() * 0.5 * visc;

        Vector forcea = S.dot(areaa)*(-1.0f/3.0);
        Vector forceb = S.dot(areab)*(-1.0f/3.0);
        Vector forcec = S.dot(areac)*(-1.0f/3.0);
        Vector forced = S.dot(aread)*(-1.0f/3.0);

#pragma unroll
        for (int j=0; j<3; j++) {
            atomicAdd(&(data->force[a][j]), forcea[j]);
            atomicAdd(&(data->force[b][j]), forceb[j]);
            atomicAdd(&(data->force[c][j]), forcec[j]);
            atomicAdd(&(data->force[d][j]), forced[j]);
        }
    }
}
