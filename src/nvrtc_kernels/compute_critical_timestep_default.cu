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
/* default */#include "cuda_runtime.h"

extern "C"
__global__ void compute_critical_timestep_nvrtc(DeviceDataPtr *data, Float t, int ntet) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < ntet) {
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

        int layer = data->layer[i] - 1;

        /////////////////////////////////////////////////
        Vector Xcenter = (Xa+Xb+Xc+Xd) * 0.25;

        Float X = Xcenter[0];
        Float Y = Xcenter[1];
        Float Z = Xcenter[2];

        /* default *///Float R = 0.0;

        /* default *///Float Phi = 0.0;

        /* default *///Float Theta = 0.0;

        //////////////////////////////////////////////
        /* default */Float E = 1.0;
        /* default */Float nu = 0.4;
        /* default */Float visc = 0.0;
        //////////////////////////////////////////////

        Vector xab(xa-xb);
        Vector xac(xa-xc);
        Vector xad(xa-xd);

        Tensor dx(-xab[0], -xac[0], -xad[0],
          -xab[1], -xac[1], -xad[1],
          -xab[2], -xac[2], -xad[2]);

        Float rho = 1.0;
        Float tetVol = dx.det()/6.0f;
        Float spacing = pow(max(tetVol,Float(0)), Float(1.0/3.0));
        Float K = E / (3.0f - 6.0f * nu); // Bulk modulus
        Float mu = E / (2.0f + 2.0f * nu); // Shear modulus

        Float dt_press = 0.5 * spacing * sqrt(rho / K);
        Float dt_shear = 0.5 * spacing * sqrt(rho / mu);
        Float dt_visc = 0.125 * spacing * spacing * rho / (visc + 1e-10);
        data->dt[i] = min(dt_press, min(dt_shear, dt_visc));
    }
}
