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

#ifndef LAGRANGIANSOLID_CUDA_KERNELS_H
#define LAGRANGIANSOLID_CUDA_KERNELS_H

#include "Typedefs.h"
#include "DeviceData.h"
#include "DeviceDataPtrManaged.cuh"
#include<curand.h>
#include<curand_kernel.h>
#include "Parameters.h"

namespace LagSol {

    class NVRTCKernal {
        public:
        std::string name;
        CUfunction kernel;
        CUmodule module;
        NVRTCKernal(std::string _name) : name(_name) {
            module = nullptr;
            kernel = nullptr;
        };
    };

    extern __managed__ Parameters gP;
    extern NVRTCKernal compute_force_nvrtc;
    extern NVRTCKernal compute_critical_timestep_nvrtc;
    extern NVRTCKernal compute_for_vis_nvrtc;

    __global__ void copy_pos_xyz(const Vector* pos, Float* xArray, Float* yArray, Float* zArray, int nver);
    __global__ void compute_volume(const DeviceDataPtrManaged* data, int ntet);
    __global__ void compute_mesh_quality(const DeviceDataPtrManaged* data, int ntet);

    __global__ void read_boundary_nodes(DeviceDataPtrManaged *data, Float radii, int nBoundary);
    __global__ void read_boundary_faces(DeviceDataPtrManaged *data, Float radii, int ntri);
    __global__ void compute_normal_from_dist(DeviceDataPtrManaged *data, Float *distance, int ntet);
    __global__ void compute_orthogonal_basis(DeviceDataPtrManaged *data, int ntet);
    __global__ void mark_rigid_nodes(DeviceDataPtrManaged *data, int ntet);

    __global__ void compute_vgrad_node(DeviceDataPtrManaged *data, int ntet);

    __global__ void compute_contact_force(DeviceDataPtrManaged *data, Float spacing, Float kappa, Float kappa_vel, Float rep_thickness, int nnbd);

    __global__ void compute_bids(DeviceDataPtrManaged *data, Float tol, int nver);
    __global__ void enforceBC(DeviceDataPtrManaged *data, int nver);

    __global__ void compute_kenergy(DeviceDataPtrManaged *data, int nver);
    __global__ void update_vel(DeviceDataPtrManaged *data, Float dt, int nver);
    __global__ void enforce_anchored(DeviceDataPtrManaged *data, Float dt, int nver);
    __global__ void update_pos(DeviceDataPtrManaged *data, Float dt, int nver);
    __global__ void distCheckKernel(
        const Vector* __restrict__ pos1,
        const Vector* __restrict__ pos2,
        const Float* __restrict__ h1,
        const Float* __restrict__ h2,
        Int nP1,Int nP2, Int maxNgb,
        const Int* p2,
        Int* __restrict__ ngbSize,
        Int* __restrict__ ngbIds);

    __host__ __device__ Vector closestPointTriangle(const Vector &p, const Vector &a, const Vector &b, const Vector &c, Float &u, Float &v, Float &w);

}
#endif //LAGRANGIANSOLID_CUDA_KERNELS_H
