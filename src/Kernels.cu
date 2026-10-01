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

#include "Kernels.h"
#include "MyGUI.h"

namespace LagSol {

    NVRTCKernal compute_force_nvrtc("compute_force_nvrtc");
    NVRTCKernal compute_orthogonal_basis_nvrtc("compute_orthogonal_basis_nvrtc");
    NVRTCKernal compute_bids_nvrtc("compute_bids_nvrtc");
    NVRTCKernal mark_rigid_nodes_nvrtc("mark_rigid_nodes_nvrtc");
    NVRTCKernal enforceBC_nvrtc("enforceBC_nvrtc");
    NVRTCKernal compute_critical_timestep_nvrtc("compute_critical_timestep_nvrtc");
    NVRTCKernal compute_for_vis_nvrtc("compute_for_vis_nvrtc");
    NVRTCKernal compute_ext_load_nvrtc("compute_ext_load_nvrtc");

    __global__ void copy_pos_xyz(const Vector* pos, Float* xArray, Float* yArray, Float* zArray, int nver) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < nver) {
            xArray[i] = pos[i][0];
            yArray[i] = pos[i][1];
            zArray[i] = pos[i][2];
        }
    }

    __global__ void compute_mesh_quality(const DeviceDataPtrManaged* data, int ntet) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < ntet) {
            int a = data->tet[i].x;
            int b = data->tet[i].y;
            int c = data->tet[i].z;
            int d = data->tet[i].w;

            Vector xa = data->pos[a];
            Vector xb = data->pos[b];
            Vector xc = data->pos[c];
            Vector xd = data->pos[d];

            Vector xab(xa-xb);
            Vector xac(xa-xc);
            Vector xad(xa-xd);
            Vector xbc(xb-xc);
            Vector xbd(xb-xd);
            Vector xcd(xc-xd);

            data->tetQual[i] = 6.0 * pow(sqrt(2.0) * fabs(xab.cross(xac).dot(xad)), Float(2.0/3.0)) / (xab.mag2() + xac.mag2() + xad.mag2() + xbc.mag2() + xbd.mag2() + xcd.mag2());
        }
    }

    __global__ void compute_volume(const DeviceDataPtrManaged* data, int ntet) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < ntet) {
            int a = data->tet[i].x;
            int b = data->tet[i].y;
            int c = data->tet[i].z;
            int d = data->tet[i].w;

            Vector xa = data->pos[a];
            Vector xb = data->pos[b];
            Vector xc = data->pos[c];
            Vector xd = data->pos[d];

            Vector xab(xa-xb);
            Vector xac(xa-xc);
            Vector xad(xa-xd);

            data->tetVol[i] = xac.cross(xab).dot(xad) * (1.0f / 6.0f);
            atomicAdd(&(data->vol[data->tet[i].x]), data->tetVol[i]*0.25f);
            atomicAdd(&(data->vol[data->tet[i].y]), data->tetVol[i]*0.25f);
            atomicAdd(&(data->vol[data->tet[i].z]), data->tetVol[i]*0.25f);
            atomicAdd(&(data->vol[data->tet[i].w]), data->tetVol[i]*0.25f);

        }
    }

    __global__ void read_boundary_nodes(DeviceDataPtrManaged *data, Float radii, int nBoundary) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < nBoundary) {
            data->boundaryNodePos[i] = data->pos[data->boundaryNodeIds[i]];
            data->boundaryNodeR[i] = radii;
        }
    }

    __global__ void read_boundary_faces(DeviceDataPtrManaged *data, Float radii, int ntri) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < ntri) {
            Vector t1(data->pos[data->tri_only_contact[i].x]);
            Vector t2(data->pos[data->tri_only_contact[i].y]);
            Vector t3(data->pos[data->tri_only_contact[i].z]);
            data->boundaryFacePos[i] = (t1 + t2 + t3) / 3.0;
            data->boundaryFaceR[i] = max((t1 - data->boundaryFacePos[i]).mag2(),(t2 - data->boundaryFacePos[i]).mag2());
            data->boundaryFaceR[i] = max((t3 - data->boundaryFacePos[i]).mag2(), data->boundaryFaceR[i]);
            data->boundaryFaceR[i] = sqrt(data->boundaryFaceR[i]) + radii;
        }
    }


    __global__ void compute_normal_from_dist(DeviceDataPtrManaged *data, Float *distance, int ntet) {
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

            Vector Xab(Xa-Xb);
            Vector Xac(Xa-Xc);
            Vector Xad(Xa-Xd);

            Float u_ab(distance[a] - distance[b]);
            Float u_ac(distance[a] - distance[c]);
            Float u_ad(distance[a] - distance[d]);

            Tensor dX(-Xab[0], -Xac[0], -Xad[0],
                      -Xab[1], -Xac[1], -Xad[1],
                      -Xab[2], -Xac[2], -Xad[2]);
            Vector du(-u_ab, -u_ac, -u_ad);

            Vector uGrad = dot(du, dX.inv());
            data->normalTetra[i] = uGrad / uGrad.mag();
            if (!isfinite(data->normalTetra[i].mag2()))
                data->normalTetra[i] = Vector(1.0,0.0,0.0);
        }
    }

    __global__ void compute_vgrad_node(DeviceDataPtrManaged *data, int ntet) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < ntet) {
            int a = data->tet[i].x;
            int b = data->tet[i].y;
            int c = data->tet[i].z;
            int d = data->tet[i].w;

            Vector p0 = data->pos[a];
            Vector p1 = data->pos[b];
            Vector p2 = data->pos[c];
            Vector p3 = data->pos[d];

            Vector vab(p0-p1);
            Vector vac(p0-p2);
            Vector vad(p0-p3);
            Vector vbc(p1-p2);
            Vector vbd(p1-p3);

            Vector velab(data->vel[a]-data->vel[b]);
            Vector velac(data->vel[a]-data->vel[c]);
            Vector velad(data->vel[a]-data->vel[d]);

            Tensor dx(-vab[0], -vac[0], -vad[0],
                      -vab[1], -vac[1], -vad[1],
                      -vab[2], -vac[2], -vad[2]);
            Tensor dvel(-velab[0], -velac[0], -velad[0],
                        -velab[1], -velac[1], -velad[1],
                        -velab[2], -velac[2], -velad[2]);

            Tensor velGrad = dvel.dot(dx.inv());
            data->vGrad[i] = velGrad;

            for (int j = 0; j < 9; j++) {
                atomicAdd(&(data->vGradNode[a][j]), velGrad[j] * 0.25f * (data->tetVol[i] / data->vol[a]));
                atomicAdd(&(data->vGradNode[b][j]), velGrad[j] * 0.25f * (data->tetVol[i] / data->vol[b]));
                atomicAdd(&(data->vGradNode[c][j]), velGrad[j] * 0.25f * (data->tetVol[i] / data->vol[c]));
                atomicAdd(&(data->vGradNode[d][j]), velGrad[j] * 0.25f * (data->tetVol[i] / data->vol[d]));
            }
        }
    }

    __global__ void compute_contact_force(DeviceDataPtrManaged *data, Float spacing, Float kappa, Float kappaVel, Float repThickness, int nnbd) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        Float ub, vb, wb;
        Float hc = repThickness * spacing; // Thickness of repulsive skin
        if (i < nnbd) {
            int pp = data->boundaryNodeIds[i];
            // Float Epp = data->ENodal[pp]/(1.f-data->nuNodal[pp]*data->nuNodal[pp]);
            Vector point = Vector(data->pos[pp]);
            for (int j = 0; j < data->ngb_size[i]; j++) {
                int tid = data->ngb_list[data->ngb_offset[i] + j];
                int t0 = data->tri_only_contact[tid].x;
                int t1 = data->tri_only_contact[tid].y;
                int t2 = data->tri_only_contact[tid].z;
                if ((pp != t0) && (pp != t1) && (pp != t2)) {
                    // Float E0 = data->ENodal[t0]/(1.f-data->nuNodal[t0]*data->nuNodal[t0]);
                    // Float E1 = data->ENodal[t1]/(1.f-data->nuNodal[t1]*data->nuNodal[t1]);
                    // Float E2 = data->ENodal[t2]/(1.f-data->nuNodal[t2]*data->nuNodal[t2]);
                    // Float Et = (E0+E1+E2)/3.0f;

                    Vector cc = closestPointTriangle(point, data->pos[t0], data->pos[t1], data->pos[t2], ub, vb, wb) - point;
                    Float rc = cc.mag();
                    if (rc < hc) {
                        cc = cc / rc;
                        Float delta = hc - rc;
                        // Vector nodeVel(data->vel[pp]);
                        // Vector aVel(data->vel[t0]);
                        // Vector bVel(data->vel[t1]);
                        // Vector cVel(data->vel[t2]);
                        // Vector velRel = (aVel * ub + bVel * vb + cVel * wb) - nodeVel;
                        // Float K = kappa * (Et*Epp)/(Et+Epp) * std::sqrt(hc * delta);
                        // Vector fn = cc * (K * delta - kappaVel * std::sqrt(spacing * delta) * velRel.dot(cc));
                        Float K = kappa * std::sqrt(hc * delta);
                        Vector fn = cc * K * delta;
                        atomicAdd(&(data->force[t0][0]), fn[0] * ub);
                        atomicAdd(&(data->force[t0][1]), fn[1] * ub);
                        atomicAdd(&(data->force[t0][2]), fn[2] * ub);
                        atomicAdd(&(data->force[t1][0]), fn[0] * vb);
                        atomicAdd(&(data->force[t1][1]), fn[1] * vb);
                        atomicAdd(&(data->force[t1][2]), fn[2] * vb);
                        atomicAdd(&(data->force[t2][0]), fn[0] * wb);
                        atomicAdd(&(data->force[t2][1]), fn[1] * wb);
                        atomicAdd(&(data->force[t2][2]), fn[2] * wb);
                        atomicAdd(&(data->force[pp][0]),-fn[0]);
                        atomicAdd(&(data->force[pp][1]),-fn[1]);
                        atomicAdd(&(data->force[pp][2]),-fn[2]);
                    }
                }
            }
        }
    }

    __global__ void compute_kenergy(DeviceDataPtrManaged *data, int nver) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < nver) {
            Vector vel = data->vel[i];
            data->kEnergy[i] = vel.dot(vel)*data->vol[i]*0.5f;
        }
    }

    __global__ void update_vel(DeviceDataPtrManaged *data, Float damping, Float dt, int nver) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < nver) {
            Float density = 1.0f;
            Float mass = data->vol[i]*density;// + static_cast<Float>(data->vol[i]==0.f);
            // Vector accel = (data->force[i] - (data->vel[i]*data->vol[i]*damping))/mass;
            Vector accel = data->force[i]/mass;
            // data->vel[i] = data->vel[i] + accel * dt - damping * data->vel[i] * dt;
            data->vel[i] = (data->vel[i] + accel * dt) / (1.0 + damping * dt);
        }
    }

    __global__ void enforce_anchored(DeviceDataPtrManaged *data, Float dt, int nver) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < nver) {
            data->vel[i] = data->vel[i] - data->vel[0];
        }
    }

    __global__ void update_pos(DeviceDataPtrManaged *data, Float dt, int nver) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < nver) {
            data->pos[i] = data->pos[i] + data->vel[i] * dt;
        }
    }

    __global__ void distCheckKernel(
            const Vector* __restrict__ pos1,
            const Vector* __restrict__ pos2,
            const Float* __restrict__ h1,
            const Float* __restrict__ h2,
            Int nP1, Int nP2, Int maxNgb,
            const Int* p2,
            Int* __restrict__ ngbSize,
            Int* __restrict__ ngbIds) {

        const Int BLOCK_SIZE = 64;
        Int idx = blockDim.x * blockIdx.x + threadIdx.x;

        __shared__ Vector sPos1[BLOCK_SIZE];
        __shared__ double sH1[BLOCK_SIZE];

        Vector aPos2(MAX_FLOAT);
        Float aH2(0.0);
        Int aP2(-1);

        if (idx < nP2) {
            aPos2 = pos2[idx];
            aH2 = h2[idx];
            aP2 = p2[idx];
        }

        Int i = 0;
        Int k = 0;
        Int rem = nP1%BLOCK_SIZE;
        while (i < (nP1 - rem)) {
            __syncthreads();
            sPos1[threadIdx.x] = pos1[k*BLOCK_SIZE + threadIdx.x];
            sH1  [threadIdx.x] = h1  [k*BLOCK_SIZE + threadIdx.x];
            __syncthreads();
            for (Int j = 0; j<BLOCK_SIZE; j++) {
                Vector dist = sPos1[j] - aPos2;
                Float  h12  = sH1[j]   + aH2;
                if (dist.mag2() <  h12 *  h12) {
                    ngbIds[atomicAdd(&ngbSize[i],1) + maxNgb * i] = aP2;
                }
                i++;
            }
            k++;
        }
        while (i < nP1) {
            Vector dist = pos1[i] - aPos2;
            Float h12 = h1[i] + aH2;
            if (dist.mag2() < h12 * h12)
                ngbIds[atomicAdd(&ngbSize[i],1) + maxNgb * i] = aP2;
            i++;
        }
    }

    __host__ __device__ Vector closestPointTriangle(const Vector &p, const Vector &a, const Vector &b, const Vector &c, Float &u, Float &v, Float &w) {
        Vector ab = b - a;
        Vector ac = c - a;
        Vector ap = p - a;
        Float d1 = ab.dot(ap);
        Float d2 = ac.dot(ap);
        if (d1 <= 0.0 && d2 <= 0.0) {
            u = 1.0;
            v = 0.0;
            w = 0.0;
            return a;
        }
        Vector bp = p - b;
        Float d3 = ab.dot(bp);
        Float d4 = ac.dot(bp);
        if (d3 >= 0.0 && d4 <= d3) {
            u = 0.0;
            v = 1.0;
            w = 0.0;
            return b;
        }
        Float vc = d1 * d4 - d3 * d2;
        if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
            v = d1 / (d1 - d3);
            u = 1.0 - v;
            w = 0.0;
            return a + ab * v;
        }
        Vector cp = p - c;
        Float d5 = ab.dot(cp);
        Float d6 = ac.dot(cp);
        if (d6 >= 0.0 && d5 <= d6) {
            u = 0.0;
            v = 0.0;
            w = 1.0;
            return c;
        }
        Float vb = d5 * d2 - d1 * d6;
        if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
            w = d2 / (d2 - d6);
            u = 1.0 - w;
            v = 0.0;
            return a + ac * w;
        }
        Float va = d3 * d6 - d5 * d4;
        if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
            w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
            u = 0.0;
            v = 1.0 - w;
            return b + (c - b) * w;
        }
        Float denom = 1.0 / (va + vb + vc);
        v = vb * denom;
        w = vc * denom;
        u = 1.0 - v - w;
        return a + ab * v + ac * w;
    }
}
