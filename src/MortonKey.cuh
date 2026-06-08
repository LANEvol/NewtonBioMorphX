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

#ifndef LAGRANGIANSOLID_CUDA_MORTON_KEY_CUH
#define LAGRANGIANSOLID_CUDA_MORTON_KEY_CUH

#include "DataStructure.h"
#include "Managed.h"

namespace LagSol {
    struct Encoder32 : Managed {
        Vector minPos;
        Vector maxPos;
        Vector delta;
        __host__ __device__ Encoder32(const Vector& _minPos,const Vector& _maxPos) : minPos(_minPos), maxPos(_maxPos), delta(_maxPos - _minPos) {};
        inline __host__ __device__ UInt32 operator() (const Vector& pos) {
            UInt32 xx, yy, zz;

            Float xScaled = (pos[0] - minPos[0]) / delta[0];
            Float yScaled = (pos[1] - minPos[1]) / delta[1];
            Float zScaled = (pos[2] - minPos[2]) / delta[2];

            xScaled *= 1024.0;
            yScaled *= 1024.0;
            zScaled *= 1024.0;

            if ((xScaled < 0.0   ) || (yScaled < 0.0   ) || (zScaled < 0.0   ) ||
                (xScaled > 1024.0) || (yScaled > 1024.0) || (zScaled > 1024.0))
                return	0xffffffffu;

            xx = ((UInt32)xScaled * 0x00010001u) & 0xFF0000FFu;
            xx = (xx * 0x00000101u) & 0x0F00F00Fu;
            xx = (xx * 0x00000011u) & 0xC30C30C3u;
            xx = (xx * 0x00000005u) & 0x49249249u;

            yy = ((UInt32)yScaled * 0x00010001u) & 0xFF0000FFu;
            yy = (yy * 0x00000101u) & 0x0F00F00Fu;
            yy = (yy * 0x00000011u) & 0xC30C30C3u;
            yy = (yy * 0x00000005u) & 0x49249249u;

            zz = ((UInt32)zScaled * 0x00010001u) & 0xFF0000FFu;
            zz = (zz * 0x00000101u) & 0x0F00F00Fu;
            zz = (zz * 0x00000011u) & 0xC30C30C3u;
            zz = (zz * 0x00000005u) & 0x49249249u;

            return (xx * 4 + yy * 2 + zz);
        };

    };

    struct Encoder64 : Managed {
        Vector minPos;
        Vector maxPos;
        Vector delta;

        __host__ __device__ Encoder64(const Vector& _minPos,const Vector& _maxPos) : minPos(_minPos), maxPos(_maxPos), delta(_maxPos - _minPos) {};
        inline __host__ __device__ UInt64 operator() (const Vector& pos) {
            Float xScaled = (pos[0] - minPos[0]) / delta[0];
            Float yScaled = (pos[1] - minPos[1]) / delta[1];
            Float zScaled = (pos[2] - minPos[2]) / delta[2];

            xScaled *= 2097152.0;
            yScaled *= 2097152.0;
            zScaled *= 2097152.0;

            if ((xScaled < 0.0      ) || (yScaled < 0.0      ) || (zScaled < 0.0      ) ||
                (xScaled > 2097152.0) || (yScaled > 2097152.0) || (zScaled > 2097152.0))
                return	0xffffffffffffffff;

            UInt64 xx,yy,zz;

            xx = (UInt64)xScaled;
            xx = (xx | (xx << (UInt64)32)) & 0x7fff00000000ffff; // 0b0111111111111111000000000000000000000000000000001111111111111111
            xx = (xx | (xx << (UInt64)16)) & 0x00ff0000ff0000ff; // 0b0000000011111111000000000000000011111111000000000000000011111111
            xx = (xx | (xx << (UInt64)8 )) & 0x700f00f00f00f00f; // 0b0111000000001111000000001111000000001111000000001111000000001111
            xx = (xx | (xx << (UInt64)4 )) & 0x30c30c30c30c30c3; // 0b0011000011000011000011000011000011000011000011000011000011000011
            xx = (xx | (xx << (UInt64)2 )) & 0x1249249249249249; // 0b0001001001001001001001001001001001001001001001001001001001001001

            yy = (UInt64)yScaled;
            yy = (yy | (yy << (UInt64)32)) & 0x7fff00000000ffff; // 0b0111111111111111000000000000000000000000000000001111111111111111
            yy = (yy | (yy << (UInt64)16)) & 0x00ff0000ff0000ff; // 0b0000000011111111000000000000000011111111000000000000000011111111
            yy = (yy | (yy << (UInt64)8 )) & 0x700f00f00f00f00f; // 0b0111000000001111000000001111000000001111000000001111000000001111
            yy = (yy | (yy << (UInt64)4 )) & 0x30c30c30c30c30c3; // 0b0011000011000011000011000011000011000011000011000011000011000011
            yy = (yy | (yy << (UInt64)2 )) & 0x1249249249249249; // 0b0001001001001001001001001001001001001001001001001001001001001001

            zz = (UInt64)zScaled;
            zz = (zz | (zz << (UInt64)32)) & 0x7fff00000000ffff; // 0b0111111111111111000000000000000000000000000000001111111111111111
            zz = (zz | (zz << (UInt64)16)) & 0x00ff0000ff0000ff; // 0b0000000011111111000000000000000011111111000000000000000011111111
            zz = (zz | (zz << (UInt64)8 )) & 0x700f00f00f00f00f; // 0b0111000000001111000000001111000000001111000000001111000000001111
            zz = (zz | (zz << (UInt64)4 )) & 0x30c30c30c30c30c3; // 0b0011000011000011000011000011000011000011000011000011000011000011
            zz = (zz | (zz << (UInt64)2 )) & 0x1249249249249249; // 0b0001001001001001001001001001001001001001001001001001001001001001

            return (zz | (yy << (UInt64)1) | (xx << (UInt64)2));
//		return  (xx * (UIntM)4 + yy * (UIntM)2 + zz);
        };
    };


}

#endif //LAGRANGIANSOLID_CUDA_MORTON_KEY_CUH
