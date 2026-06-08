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

#ifndef LAGRANGIANSOLID_CUDA_PARAMETERS_H
#define LAGRANGIANSOLID_CUDA_PARAMETERS_H

//namespace LagSol {
    struct Parameters2 {
        std::string E[MAX_NLAYERS];
        std::string nu[MAX_NLAYERS];
        std::string visc[MAX_NLAYERS];
        std::string grRate1_Ref[MAX_NLAYERS];
        std::string grRate2_Ref[MAX_NLAYERS];
        std::string grRate3_Ref[MAX_NLAYERS];
        std::string plasticity[MAX_NLAYERS];

        std::string k1[MAX_NLAYERS];
        std::string k2[MAX_NLAYERS];
        std::string fiber1_Ref[MAX_NLAYERS];
        std::string fiber2_Ref[MAX_NLAYERS];
        std::string fiber3_Ref[MAX_NLAYERS];
        std::string fiber4_Ref[MAX_NLAYERS];
        std::string actin_Ref[MAX_NLAYERS];


        Parameters2() {
            init();
        }

        void init() {
            for (int i = 0; i < MAX_NLAYERS; i++) {
                E[i] = "1.0";
                nu[i] = "0.35";
                visc[i] = "0.1";
                plasticity[i]    = "0.0";
                grRate1_Ref[i] = "0.0";
                grRate2_Ref[i] = "0.0";
                grRate3_Ref[i] = "0.0";

                k1[i] = "0.0";
                k2[i] = "0.0";
                fiber1_Ref[i] = "Tensor(0, 0, 0)";
                fiber2_Ref[i] = "Tensor(0, 0, 0)";
                fiber3_Ref[i] = "Tensor(0, 0, 0)";
                fiber4_Ref[i] = "Tensor(0, 0, 0)";
                actin_Ref[i] = "Tensor(0, 0, 0)";
            }
        }
    };



    struct Parameters {
        bool isRigidLayer[MAX_NLAYERS];
        Tensor grRateGlobal;
        Float damping;

        bool useMeshDef_E;
        bool useMeshDef_nu;
        bool useMeshDef_viscosity;
        bool useMeshDef_plasticity;
        bool useMeshDef_k1;
        bool useMeshDef_k2;
        bool useMeshDef_grRate1_Ref;
        bool useMeshDef_grRate2_Ref;
        bool useMeshDef_grRate3_Ref;
        bool useMeshDef_fiber1_Ref;
        bool useMeshDef_fiber2_Ref;
        bool useMeshDef_fiber3_Ref;
        bool useMeshDef_fiber4_Ref;
        bool useMeshDef_actin_Ref;

        //Float growthRate;
        int bcTypeMinAxis0; // 0 : free surface, 1 : no-slip, 2 : free-slip
        int bcTypeMaxAxis0;
        int bcTypeMinAxis1;
        int bcTypeMaxAxis1;
        int bcTypeMinAxis2;
        int bcTypeMaxAxis2;
        bool isAnchored() {
            bool isAnchored = false;
            for (int i=0; i<MAX_NLAYERS; i++)
                isAnchored |= isRigidLayer[i];
            if (isAnchored) {
                return isAnchored;
            }
            if (grCoordType == LagSol::CoordinateSystem::Spherical || grCoordType == LagSol::CoordinateSystem::Toroidal) {
                isAnchored = (bcTypeMinAxis0>0) || (bcTypeMaxAxis0>0);
            } else if (grCoordType == LagSol::CoordinateSystem::CylindricalZ) {
                isAnchored = (bcTypeMaxAxis0>0) || (bcTypeMinAxis1>0) || (bcTypeMaxAxis1>0);
            } else if (grCoordType == LagSol::CoordinateSystem::CylindricalY  || grCoordType == LagSol::CoordinateSystem::ConeAdapted) {
                isAnchored = (bcTypeMinAxis0>0) || (bcTypeMaxAxis0>0) || (bcTypeMinAxis1>0) || (bcTypeMaxAxis1>0);
            } else {
                isAnchored = (bcTypeMinAxis0>0) || (bcTypeMaxAxis0>0) || (bcTypeMinAxis1>0) || (bcTypeMaxAxis1>0) || (bcTypeMinAxis2>0) || (bcTypeMaxAxis2>0);
            }
            return isAnchored;
        };

        LagSol::CoordinateSystem grCoordType;

        Vector posRefMax;
        Vector posRefMin;
        Float RTorus;
        Float apex;
        Float rRefMin;
        Float rRefMax;

        __host__ __device__ void init() {
            for (int i = 0; i < MAX_NLAYERS; i++)
                isRigidLayer[i] = false;
            grRateGlobal = Tensor(0.0);
            damping = 0.0;

            useMeshDef_E = false;
            useMeshDef_nu = false;
            useMeshDef_viscosity = false;
            useMeshDef_plasticity = false;
            useMeshDef_grRate1_Ref = false;
            useMeshDef_grRate2_Ref = false;
            useMeshDef_grRate3_Ref = false;
            useMeshDef_k1 = false;
            useMeshDef_k2 = false;
            useMeshDef_fiber1_Ref = false;
            useMeshDef_fiber2_Ref = false;
            useMeshDef_fiber3_Ref = false;
            useMeshDef_fiber4_Ref = false;
            useMeshDef_actin_Ref = false;

            bcTypeMinAxis0 = 0; // 0 : free, 1 : fixed, 2 : tangent-slide
            bcTypeMaxAxis0 = 0;
            bcTypeMinAxis1 = 0;
            bcTypeMaxAxis1 = 0;
            bcTypeMinAxis2 = 0;
            bcTypeMaxAxis2 = 0;

            grCoordType = LagSol::Cartesian;

            rRefMin = 0.0;
            rRefMax = 0.0;
            RTorus = 0.0;
            apex = 0.0;
            posRefMax = Vector(0.0);
            posRefMin = Vector(0.0);
        }
    };
//}
#endif //LAGRANGIANSOLID_CUDA_PARAMETERS_H