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
__global__ void compute_orthogonal_basis_nvrtc(DeviceDataPtr *data, int ntet) {
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
        Vector tetCenter = (Xa+Xb+Xc+Xd)*0.25;

        Float X = tetCenter[0];
        Float Y = tetCenter[1];
        Float Z = tetCenter[2];

        /* default *///Vector n;
        /* default *///Vector t1;
        /* default *///Vector t2;
        /* default *///Vector t;
        /* default *///Vector b;

        /* default *///Float Theta;
        /* default *///Float Phi;
        /* default *///Float hapex = gP.apex*0.5;

        /* default */data->R[i][0] = Float(1.0);
        /* default */data->R[i][1] = Float(0.0);
        /* default */data->R[i][2] = Float(0.0);
        /* default */data->R[i][3] = Float(0.0);
        /* default */data->R[i][4] = Float(1.0);
        /* default */data->R[i][5] = Float(0.0);
        /* default */data->R[i][6] = Float(0.0);
        /* default */data->R[i][7] = Float(0.0);
        /* default */data->R[i][8] = Float(1.0);
    }
}
