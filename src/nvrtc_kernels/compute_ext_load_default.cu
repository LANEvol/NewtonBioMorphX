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

        Float X = (Xa + Xb + Xc) * 1.0/3.0;
        Float Y = (Ya + Yb + Yc) * 1.0/3.0;
        Float Z = (Za + Zb + Zc) * 1.0/3.0;
        
        /* default *///Float hapex;

        /* default *///Float Ra;
        /* default *///Float Rb;
        /* default *///Float Rc;
        /* default *///Float Na;
        /* default *///Float Nb;
        /* default *///Float Nc;
        

        /* default *///Float XMin;
        /* default *///Float XMax;
        /* default *///Float YMin;
        /* default *///Float YMax;
        /* default *///Float ZMin;
        /* default *///Float ZMax;
        /* default *///Float RMin;
        /* default *///Float RMax;
        /* default *///Float NMin;
        /* default *///Float NMax;


        /* default *///Float RMina;
        /* default *///Float RMinb;
        /* default *///Float RMinc;
        /* default *///Float RMaxa;
        /* default *///Float RMaxb;
        /* default *///Float RMaxc;


        /* default *///Float Phi;
        /* default *///Float Theta;
        /* default *///Float R;
        /* default *///Float N;
        /* default *///Float S;

        /* default */const unsigned int XMinStateT = 0;
	    /* default */const unsigned int XMaxStateT = 0;
        /* default */const unsigned int YMinStateT = 0;
        /* default */const unsigned int YMaxStateT = 0;
        /* default */const unsigned int ZMinStateT = 0;
        /* default */const unsigned int ZMaxStateT = 0;
        /* default */const unsigned int RMinStateT = 0;
        /* default */const unsigned int RMaxStateT = 0;
        /* default */const unsigned int NMinStateT = 0;
        /* default */const unsigned int NMaxStateT = 0;

        data->extLoad[i] = Vector(0.0);
        
        if (XMinStateT || XMaxStateT || YMinStateT || YMaxStateT || ZMinStateT || ZMaxStateT || RMinStateT || RMaxStateT || NMinStateT || NMaxStateT) {
            /* default *///data->extLoad[i] += XMinStateT * Vector(0.0);
            /* default *///data->extLoad[i] += XMaxStateT * Vector(0.0);
            /* default *///data->extLoad[i] += YMinStateT * Vector(0.0);
            /* default *///data->extLoad[i] += YMaxStateT * Vector(0.0);
            /* default *///data->extLoad[i] += ZMinStateT * Vector(0.0);
            /* default *///data->extLoad[i] += ZMaxStateT * Vector(0.0);
            /* default *///data->extLoad[i] += RMinStateT * Vector(0.0);
            /* default *///data->extLoad[i] += RMaxStateT * Vector(0.0);
            /* default *///data->extLoad[i] += NMinStateT * Vector(0.0);
            /* default *///data->extLoad[i] += NMaxStateT * Vector(0.0);
        }

        if (data->extLoad[i].mag2() > 0.0) {
            Vector area = (data->posRef[t2] - data->posRef[t1]).cross(data->posRef[t3] - data->posRef[t1]) * 0.5;
            /* default *///Tensor RmatTri;
            Vector fn = RmatTri.dot(data->extLoad[i]) * area.mag() * (1.0 / 3.0);
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
