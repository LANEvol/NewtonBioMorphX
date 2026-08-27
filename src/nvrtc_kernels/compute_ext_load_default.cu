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
/* default */

#include "cuda_runtime.h"

extern "C"
__global__ void compute_ext_load_nvrtc(DeviceDataPtr *data, Float t, Float tol, int ntri) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < ntri) {
        int t1 = data->tri[i].x;
        int t2 = data->tri[i].y;
        int t3 = data->tri[i].z;

        Float Xa = data->posRef[t1][0];
        Float Ya = data->posRef[t1][1];
        Float Za = data->posRef[t1][2];

        Float Xb = data->posRef[t2][0];
        Float Yb = data->posRef[t2][1];
        Float Zb = data->posRef[t2][2];

        Float Xc = data->posRef[t3][0];
        Float Yc = data->posRef[t3][1];
        Float Zc = data->posRef[t3][2];

        /* default *///Float hapex;

        /* default *///Float Ra;
        /* default *///Float Rb;
        /* default *///Float Rc;

        /* default */Float xMin = gP.posRefMin[0];
        /* default */Float xMax = gP.posRefMax[0];
        /* default */Float yMin = gP.posRefMin[1];
        /* default */Float yMax = gP.posRefMax[1];
        /* default */Float zMin = gP.posRefMin[2];
        /* default */Float zMax = gP.posRefMax[2];

        Float X = (Xa + Xb + Xc) * 1.0/3.0;
        Float Y = (Ya + Yb + Yc) * 1.0/3.0;
        Float Z = (Za + Zb + Zc) * 1.0/3.0;


        /* default *///Float rMin;
        /* default *///Float rMina;
        /* default *///Float rMinb;
        /* default *///Float rMinc;
        /* default *///Float rMax;
        /* default *///Float rMaxa;
        /* default *///Float rMaxb;
        /* default *///Float rMaxc;

        /* default *///Float Phi;
        /* default *///Float Theta;
        /* default *///Float R;


        const unsigned int xMinState = (fabs(Xa - xMin) < tol) && (fabs(Xb - xMin) < tol) && (fabs(Xc - xMin) < tol);
        const unsigned int xMaxState = (fabs(Xa - xMax) < tol) && (fabs(Xb - xMax) < tol) && (fabs(Xc - xMax) < tol);
        const unsigned int yMinState = (fabs(Ya - yMin) < tol) && (fabs(Yb - yMin) < tol) && (fabs(Yc - yMin) < tol);
        const unsigned int yMaxState = (fabs(Ya - yMax) < tol) && (fabs(Yb - yMax) < tol) && (fabs(Yc - yMax) < tol);
        const unsigned int zMinState = (fabs(Za - zMin) < tol) && (fabs(Zb - zMin) < tol) && (fabs(Zc - zMin) < tol);
        const unsigned int zMaxState = (fabs(Za - zMax) < tol) && (fabs(Zb - zMax) < tol) && (fabs(Zc - zMax) < tol);
        /* default */const unsigned int rMinStateT = 0;
        /* default */const unsigned int rMaxStateT = 0;

        data->extLoad[i] = Vector(0.0);
        if (xMinState || xMaxState || yMinState || yMaxState || zMinState || zMaxState || rMinStateT || rMaxStateT) {
            /* default */// data->extLoad[i] += xMinState * Vector(0.0);
            /* default */// data->extLoad[i] += xMaxState * Vector(0.0);
            /* default */// data->extLoad[i] += yMinState * Vector(0.0);
            /* default */// data->extLoad[i] += yMaxState * Vector(0.0);
            /* default */// data->extLoad[i] += zMinState * Vector(0.0);
            /* default */// data->extLoad[i] += zMaxState * Vector(0.0);
            /* default */// data->extLoad[i] += rMinStateT * Vector(0.0);
            /* default */// data->extLoad[i] += rMinStateT * Vector(0.0);
        }

        if (data->extLoad[i].mag2() > 0.0) {
            Vector area = (data->posRef[t2] - data->posRef[t1]).cross(data->posRef[t3] - data->posRef[t1]) * 0.5;
            Vector fn = data->extLoad[i] * area.mag() * (1.0 / 3.0);
            atomicAdd(&(data->force[t1][0]),fn[0]);
            atomicAdd(&(data->force[t1][1]),fn[1]);
            atomicAdd(&(data->force[t1][2]),fn[2]);
            atomicAdd(&(data->force[t2][0]),fn[0]);
            atomicAdd(&(data->force[t2][1]),fn[1]);
            atomicAdd(&(data->force[t2][2]),fn[2]);
            atomicAdd(&(data->force[t3][0]),fn[0]);
            atomicAdd(&(data->force[t3][1]),fn[1]);
            atomicAdd(&(data->force[t3][2]),fn[2]);
        }
    }
}
