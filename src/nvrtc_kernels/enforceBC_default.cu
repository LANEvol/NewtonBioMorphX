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
__global__ void enforceBC_nvrtc(DeviceDataPtr *data, int nver) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < nver) {

        Float x = data->pos[i][0];
        Float y = data->pos[i][1];
        Float z = data->pos[i][2];

        uint2 bcState = data->bcState[i];

        Tensor xxt(1.0,0.0,0.0);
        Tensor yyt(0.0,1.0,0.0);
        Tensor zzt(0.0,0.0,1.0);

        /* default *///Float hapex;
        /* default *///Float theta;

        /* default *///Vector r;
        /* default *///Float phi;

        /* default *///Tensor rrt;
        /* default *///Tensor nnt;
        /* default *///Tensor aat;
        /* default *///Tensor ppt;

        /* default */Tensor proj;
        
        data->vel[i] = data->vel[i] - proj.dot(data->vel[i]);
        // data->vGradNode[i] = data->vGradNode[i] - proj.dot(data->vGradNode[i]).dot(proj.trans());
        if (data->isRigid[i] != 0)
            data->vel[i] = Vector(0.0);

    }
}
