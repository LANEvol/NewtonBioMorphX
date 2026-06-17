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
__global__ void compute_bids_nvrtc(DeviceDataPtr *data, Float tol, int nver) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < nver) {

        const unsigned int ax0Constraint = 1u;
        const unsigned int ax1Constraint = 2u;
        const unsigned int ax2Constraint = 4u;

        Float X = data->posRef[i][0];
        Float Y = data->posRef[i][1];
        Float Z = data->posRef[i][2];

        /* default *///Float R;
        /* default *///Float hapex;

        /* default */Float xMin = gP.posRefMin[0];
        /* default */Float xMax = gP.posRefMax[0];
        /* default */Float yMin = gP.posRefMin[1];
        /* default */Float yMax = gP.posRefMax[1];
        /* default */Float zMin = gP.posRefMin[2];
        /* default */Float zMax = gP.posRefMax[2];

        /* default *///Float rMin;
        /* default *///Float rMax;

        /* default */int bcTypeMinAxis0;
        /* default */int bcTypeMaxAxis0;
        /* default */int bcTypeMinAxis1;
        /* default */int bcTypeMaxAxis1;
        /* default */int bcTypeMinAxis2;
        /* default */int bcTypeMaxAxis2;

        const unsigned int xMinState = fabs(data->posRef[i][0] - xMin) < tol;
        const unsigned int xMaxState = fabs(data->posRef[i][0] - xMax) < tol;
        const unsigned int yMinState = fabs(data->posRef[i][1] - yMin) < tol;
        const unsigned int yMaxState = fabs(data->posRef[i][1] - yMax) < tol;
        const unsigned int zMinState = fabs(data->posRef[i][2] - zMin) < tol;
        const unsigned int zMaxState = fabs(data->posRef[i][2] - zMax) < tol;
        /* default *///const unsigned int rMinState = fabs(R - rMin) < tol;
        /* default *///const unsigned int rMaxState = fabs(R - rMax) < tol;

        /* default *///unsigned int bcState;

        data->bcState[i].x = bcState;
    }
}
