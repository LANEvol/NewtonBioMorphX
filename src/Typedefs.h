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

#ifndef LAGRANGIANSOLID_CUDA_TYPEDEFS_H
#define LAGRANGIANSOLID_CUDA_TYPEDEFS_H

typedef int Int;
typedef unsigned int UInt;
typedef unsigned int UInt32;
typedef unsigned long UInt64;

#define EXPONENTIAL_GROWTH_RATE_FUNCTION
// #define DBL_PRECISION

#ifdef DBL_PRECISION
typedef double Float;
#else
typedef float Float;
#endif


#define _MORTON_KEY_BITS 32
#if _MORTON_KEY_BITS == 32
typedef UInt32 UIntM;
#define _MAX_UINTM    (UIntM)0xffffffff
#define _MAX_MORTON   (UIntM)0x3fffffff
#define _INITIAL_MASK (UIntM)0x38000000
#define _MAX_LEVEL_TREE 9
#elif _MORTON_KEY_BITS == 64
typedef UInt64 UIntM;
#define _MAX_UINTM    (UIntM)0xffffffffffffffff
#define _MAX_MORTON   (UIntM)0x7fffffffffffffff
#define _INITIAL_MASK (UIntM)0x7000000000000000
#define _MAX_LEVEL_TREE 20
#endif

#define MAX_FLOAT FLT_MAX

const size_t MAX_NGB = 1000;

//#define _LUNCH(n,tb,kernel) if (n>0) (kernel)<<<(n + tb - 1) / tb,tb>>>
#define _LAUNCH(n,tb,kernel) (kernel)<<<(n + tb - 1) / tb,tb>>>

#define _LAUNCH_2D(n,m,tb,kernel) (kernel)<<<dim3((n + tb - 1) / tb, (m + tb - 1) / tb),dim3(tb,tb)>>>


#define _LOAD_NVRTC(NVRTCKernel, kernelId, pathToKernel) {\
    if (NVRTCKernel.module != nullptr) cuModuleUnload(NVRTCKernel.module);\
    NVRTCKernel.kernel = loadKernel(compileKernel(loadTextSource(pathToKernel + NVRTCKernel.name + std::string(".cu")), std::string("temp") + std::to_string(kernelId) + std::string(".cu")), NVRTCKernel.module, NVRTCKernel.name);\
    }

#define _LAUNCH_NVRTC(n,tb,kernel,...) { \
    void* args[] = __VA_ARGS__;\
    cuLaunchKernel(kernel,\
               (n + tb - 1) / tb, 1, 1, \
               tb, 1, 1, \
               0, 0, args, 0);\
    }

#define _COPY4(a,a0,a1,a2,a3) {a[0]=a0;a[1]=a1;a[2]=a2;a[3]=a3;}

#define _REMAP(_map,_array) { \
	decltype(_array) _dummyVect(_array); \
	thrust::gather(_map.begin(), _map.end(), _dummyVect.begin(), _array.begin());\
}

#define _ERROR_MESSAGE(message) {\
	std::cerr << message << std::endl<<std::flush;\
}

#define MAX_NLAYERS 6

namespace LagSol {
    enum BasicGeometry {
        Arbitrary = 0, Box, Disk, Tube, Cone, Sphere, Torus,
    };

    enum CoordinateSystem {
        NormalTangent = 0, Cartesian, CylindricalZ, CylindricalY, Spherical, Toroidal, ConeAdapted,
    };
}
#endif //LAGRANGIANSOLID_CUDA_TYPEDEFS_H
