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
        /* default *///Float N;

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

        /* default *///int bcTypeMinAxis0;
        /* default *///int bcTypeMaxAxis0;
        /* default *///int bcTypeMinAxis1;
        /* default *///int bcTypeMaxAxis1;
        /* default *///int bcTypeMinAxis2;
        /* default *///int bcTypeMaxAxis2;

        /* default *///const unsigned int XMinState;
        /* default *///const unsigned int XMaxState;
        /* default *///const unsigned int YMinState;
        /* default *///const unsigned int YMaxState;
        /* default *///const unsigned int ZMinState;
        /* default *///const unsigned int ZMaxState;
        /* default *///const unsigned int RMinState;
        /* default *///const unsigned int RMaxState;
        /* default *///const unsigned int NMinState;
        /* default *///const unsigned int NMaxState;

        /* default *///unsigned int bcState;

        data->bcState[i].x = bcState;
    }
}
