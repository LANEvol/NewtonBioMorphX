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

#include "DeviceData.h"

namespace LagSol {
    void DeviceData::init(Mesh& mesh) {
        Eigen::Array<char,Eigen::Dynamic,1> bidsXMaxh,bidsYMaxh,bidsZMaxh;
        Eigen::Array<char,Eigen::Dynamic,1> bidsXMinh,bidsYMinh,bidsZMinh;

        Vector posMin(mesh.pos.col(0).minCoeff(),mesh.pos.col(1).minCoeff(),mesh.pos.col(2).minCoeff());
        Vector posMax(mesh.pos.col(0).maxCoeff(),mesh.pos.col(1).maxCoeff(),mesh.pos.col(2).maxCoeff());
        encoder = new Encoder32(posMin,posMax);

        Vector minCoord, maxCoord;
        minCoord[0] = mesh.pos.col(0).minCoeff();
        minCoord[1] = mesh.pos.col(1).minCoeff();
        minCoord[2] = mesh.pos.col(2).minCoeff();
        maxCoord[0] = mesh.pos.col(0).maxCoeff();
        maxCoord[1] = mesh.pos.col(1).maxCoeff();
        maxCoord[2] = mesh.pos.col(2).maxCoeff();
        bidsXMinh = ((mesh.pos.col(0) - minCoord[0]).abs() < 1e-6).cast<char>();
        bidsXMaxh = ((mesh.pos.col(0) - maxCoord[0]).abs() < 1e-6).cast<char>();
        bidsYMinh = ((mesh.pos.col(1) - minCoord[1]).abs() < 1e-6).cast<char>();
        bidsYMaxh = ((mesh.pos.col(1) - maxCoord[1]).abs() < 1e-6).cast<char>();
        bidsZMinh = ((mesh.pos.col(2) - minCoord[2]).abs() < 1e-6).cast<char>();
        bidsZMaxh = ((mesh.pos.col(2) - maxCoord[2]).abs() < 1e-6).cast<char>();

        isRigid.resize(mesh.nver,0);
        bcState.resize(mesh.nver,{0,0});
        bidsXMin = CharArrayDev(std::vector<char>(reinterpret_cast<char *>(bidsXMinh.data()),reinterpret_cast<char *>(bidsXMinh.data())+mesh.nver));
        bidsXMax = CharArrayDev(std::vector<char>(reinterpret_cast<char *>(bidsXMaxh.data()),reinterpret_cast<char *>(bidsXMaxh.data())+mesh.nver));
        bidsYMin = CharArrayDev(std::vector<char>(reinterpret_cast<char *>(bidsYMinh.data()),reinterpret_cast<char *>(bidsYMinh.data())+mesh.nver));
        bidsYMax = CharArrayDev(std::vector<char>(reinterpret_cast<char *>(bidsYMaxh.data()),reinterpret_cast<char *>(bidsYMaxh.data())+mesh.nver));
        bidsZMin = CharArrayDev(std::vector<char>(reinterpret_cast<char *>(bidsZMinh.data()),reinterpret_cast<char *>(bidsZMinh.data())+mesh.nver));
        bidsZMax = CharArrayDev(std::vector<char>(reinterpret_cast<char *>(bidsZMaxh.data()),reinterpret_cast<char *>(bidsZMaxh.data())+mesh.nver));

        pos = VectorArrayDev(std::vector<Vector>(reinterpret_cast<Vector*>(mesh.pos.data()),reinterpret_cast<Vector*>(mesh.pos.data())+mesh.nver));
        posRef = VectorArrayDev(std::vector<Vector>(reinterpret_cast<Vector*>(mesh.pos.data()),reinterpret_cast<Vector*>(mesh.pos.data())+mesh.nver));

        normalTetra.resize(mesh.ntet,Vector(0.0));

        dt.resize(mesh.ntet,0.0);

        vGrad.resize(mesh.ntet,Tensor(0.0));
        vGradNode.resize(mesh.nver,Tensor(0.0));

        tet = Int4ArrayDev(std::vector<int4>(reinterpret_cast<int4*>(mesh.tet.data()),reinterpret_cast<int4*>(mesh.tet.data())+mesh.ntet));
        tri = Int3ArrayDev(std::vector<int3>(reinterpret_cast<int3*>(mesh.tri.data()),reinterpret_cast<int3*>(mesh.tri.data())+mesh.ntri));
        fac = Int3ArrayDev(std::vector<int3>(reinterpret_cast<int3*>(mesh.fac.data()),reinterpret_cast<int3*>(mesh.fac.data())+mesh.nfac));
        layer = IntArrayDev(std::vector<int>(reinterpret_cast<int*>(mesh.layer.data()),reinterpret_cast<int*>(mesh.layer.data())+mesh.ntet));

        boundaryNodeIds = IntArrayDev(std::vector<int>(reinterpret_cast<int*>(mesh.boundaryNodeIds.data()),reinterpret_cast<int*>(mesh.boundaryNodeIds.data())+mesh.nnbd));

        mKey.resize(mesh.ntet);

        tempVec1.resize(mesh.nver,Vector(0.0));
        tempVec2.resize(mesh.nver,Vector(0.0));
        tempVec3.resize(mesh.nver,Vector(0.0));
        tempVec4.resize(mesh.nver,Vector(0.0));
        tempTens1.resize(mesh.nver,Tensor(0.0));
        tempTens2.resize(mesh.nver,Tensor(0.0));
        tempTens3.resize(mesh.nver,Tensor(0.0));
        tempTens4.resize(mesh.nver,Tensor(0.0));


        vel.resize(mesh.nver,Vector(0.0));
        force.resize(mesh.nver,Vector(0.0));
        tetVol.resize(mesh.ntet,0.0);

        E.resize(mesh.ntet,0.0);
        nu.resize(mesh.ntet,0.0);
        visc.resize(mesh.ntet,0.0);
        plasticity.resize(mesh.ntet,0.0);
        k1.resize(mesh.ntet,0.0);
        k2.resize(mesh.ntet,0.0);
        grRate1_Ref.resize(mesh.ntet,0.0);
        grRate2_Ref.resize(mesh.ntet,0.0);
        grRate3_Ref.resize(mesh.ntet,0.0);

        fiber1_Ref.resize(mesh.ntet,Tensor(0.0));
        fiber2_Ref.resize(mesh.ntet,Tensor(0.0));
        fiber3_Ref.resize(mesh.ntet,Tensor(0.0));
        fiber4_Ref.resize(mesh.ntet,Tensor(0.0));
        actin_Ref.resize(mesh.ntet,Tensor(0.0));


        if (mesh.E.rows() == mesh.ntet)
            E = ScalarArrayDev(std::vector<Float>(reinterpret_cast<Float *>(mesh.E.data()),reinterpret_cast<Float *>(mesh.E.data()) + mesh.ntet));
        if (mesh.nu.rows() == mesh.ntet)
            nu = ScalarArrayDev(std::vector<Float>(reinterpret_cast<Float *>(mesh.nu.data()),reinterpret_cast<Float *>(mesh.nu.data()) + mesh.ntet));
        if (mesh.visc.rows() == mesh.ntet)
            visc = ScalarArrayDev(std::vector<Float>(reinterpret_cast<Float *>(mesh.visc.data()),reinterpret_cast<Float *>(mesh.visc.data()) + mesh.ntet));
        if (mesh.plasticity.rows() == mesh.ntet)
            plasticity = ScalarArrayDev(std::vector<Float>(reinterpret_cast<Float *>(mesh.plasticity.data()),reinterpret_cast<Float *>(mesh.plasticity.data()) + mesh.ntet));
        if (mesh.k1.rows() == mesh.ntet)
            k1 = ScalarArrayDev(std::vector<Float>(reinterpret_cast<Float *>(mesh.k1.data()),reinterpret_cast<Float *>(mesh.k1.data()) + mesh.ntet));
        if (mesh.k2.rows() == mesh.ntet)
            k2 = ScalarArrayDev(std::vector<Float>(reinterpret_cast<Float *>(mesh.k2.data()),reinterpret_cast<Float *>(mesh.k2.data()) + mesh.ntet));
        if (mesh.grRate1.rows() == mesh.ntet)
            grRate1_Ref = ScalarArrayDev(std::vector<Float>(reinterpret_cast<Float *>(mesh.grRate1.data()),reinterpret_cast<Float *>(mesh.grRate1.data()) + mesh.ntet));
        if (mesh.grRate2.rows() == mesh.ntet)
            grRate2_Ref = ScalarArrayDev(std::vector<Float>(reinterpret_cast<Float *>(mesh.grRate2.data()),reinterpret_cast<Float *>(mesh.grRate2.data()) + mesh.ntet));
        if (mesh.grRate3.rows() == mesh.ntet)
            grRate3_Ref = ScalarArrayDev(std::vector<Float>(reinterpret_cast<Float *>(mesh.grRate3.data()),reinterpret_cast<Float *>(mesh.grRate3.data()) + mesh.ntet));
        if (mesh.actinTetra.rows() == mesh.ntet)
            actin_Ref = TensorArrayDev(std::vector<Tensor>(reinterpret_cast<Tensor*>(mesh.actinTetra.data()),reinterpret_cast<Tensor*>(mesh.actinTetra.data())+mesh.ntet));
        if (mesh.fiberTetra1.rows() == mesh.ntet)
            fiber1_Ref = TensorArrayDev(std::vector<Tensor>(reinterpret_cast<Tensor*>(mesh.fiberTetra1.data()),reinterpret_cast<Tensor*>(mesh.fiberTetra1.data())+mesh.ntet));
        if (mesh.fiberTetra2.rows() == mesh.ntet)
            fiber2_Ref = TensorArrayDev(std::vector<Tensor>(reinterpret_cast<Tensor*>(mesh.fiberTetra2.data()),reinterpret_cast<Tensor*>(mesh.fiberTetra2.data())+mesh.ntet));
        if (mesh.fiberTetra3.rows() == mesh.ntet)
            fiber3_Ref = TensorArrayDev(std::vector<Tensor>(reinterpret_cast<Tensor*>(mesh.fiberTetra3.data()),reinterpret_cast<Tensor*>(mesh.fiberTetra3.data())+mesh.ntet));
        if (mesh.fiberTetra4.rows() == mesh.ntet)
            fiber4_Ref = TensorArrayDev(std::vector<Tensor>(reinterpret_cast<Tensor*>(mesh.fiberTetra4.data()),reinterpret_cast<Tensor*>(mesh.fiberTetra4.data())+mesh.ntet));

        actin = actin_Ref;
        fiber1 = fiber1_Ref;
        fiber2 = fiber2_Ref;
        fiber3 = fiber3_Ref;
        fiber4 = fiber4_Ref;
        grRate1 = grRate1_Ref;
        grRate2 = grRate2_Ref;
        grRate3 = grRate3_Ref;


        
        R.resize(mesh.ntet, Tensor(1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0));
        stress.resize(mesh.ntet, Tensor(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0));
        Fp.resize(mesh.ntet, Tensor(1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0));
        Fg.resize(mesh.ntet, Tensor(1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0));
        vol.resize(mesh.nver,0.0);
        tetQual.resize(mesh.ntet,0.0);
        potEnergy.resize(mesh.ntet,0.0);
        kEnergy.resize(mesh.nver,0.0);
        vonMises.resize(mesh.ntet,0.0);
        pressure.resize(mesh.ntet,0.0);

        boundaryNodeType.resize(mesh.nnbd,0);
        boundaryNodeNormal.resize(mesh.nnbd,Vector(0.0));
        boundaryNodePos.resize(mesh.nnbd,Vector(0.0));
        boundaryFacePos.resize(mesh.ntri,Vector(0.0));
        boundaryNodeR.resize(mesh.nnbd,0.0);
        boundaryFaceR.resize(mesh.ntri,0.0);

        ngb_list.resize(mesh.nnbd*MAX_NGB);
        ngb_size.resize(mesh.nnbd);
        ngb_offset.resize(mesh.nnbd);

    }

    DeviceData::~DeviceData() {
        delete encoder;
    }

    #define _SET_POINTER(var) devicePtr.var = thrust::raw_pointer_cast(var.data());

    void DeviceData::set(DeviceDataPtrManaged& devicePtr) {
        cudaDeviceSynchronize();
        _SET_POINTER(pos);
        _SET_POINTER(posRef);
        _SET_POINTER(normalTetra);

        _SET_POINTER(dt);

        _SET_POINTER(vGrad);
        _SET_POINTER(vGradNode);

        _SET_POINTER(grRate1_Ref);
        _SET_POINTER(grRate2_Ref);
        _SET_POINTER(grRate3_Ref);
        _SET_POINTER(fiber1_Ref);
        _SET_POINTER(fiber2_Ref);
        _SET_POINTER(fiber3_Ref);
        _SET_POINTER(fiber4_Ref);
        _SET_POINTER(actin_Ref);

        _SET_POINTER(grRate1);
        _SET_POINTER(grRate2);
        _SET_POINTER(grRate3);
        _SET_POINTER(fiber1);
        _SET_POINTER(fiber2);
        _SET_POINTER(fiber3);
        _SET_POINTER(fiber4);
        _SET_POINTER(actin);

        _SET_POINTER(tet);
        _SET_POINTER(tri);
        _SET_POINTER(fac);
        _SET_POINTER(isRigid);
        _SET_POINTER(bcState);
        _SET_POINTER(bidsXMin);
        _SET_POINTER(bidsXMax);
        _SET_POINTER(bidsYMin);
        _SET_POINTER(bidsYMax);
        _SET_POINTER(bidsZMin);
        _SET_POINTER(bidsZMax);

        _SET_POINTER(tempVec1);
        _SET_POINTER(tempVec2);
        _SET_POINTER(tempVec3);
        _SET_POINTER(tempVec4);
        _SET_POINTER(tempTens1);
        _SET_POINTER(tempTens2);
        _SET_POINTER(tempTens3);
        _SET_POINTER(tempTens4);



        _SET_POINTER(k1);
        _SET_POINTER(k2);
        _SET_POINTER(vel);
        _SET_POINTER(force);
        _SET_POINTER(tetVol);
        _SET_POINTER(E);
        _SET_POINTER(nu);
        _SET_POINTER(plasticity);
        _SET_POINTER(visc);
        _SET_POINTER(R);
        _SET_POINTER(stress);
        _SET_POINTER(Fg);
        _SET_POINTER(Fp);
        _SET_POINTER(layer);
        _SET_POINTER(vol);
        _SET_POINTER(tetQual);
        _SET_POINTER(potEnergy);
        _SET_POINTER(kEnergy);
        _SET_POINTER(vonMises);
        _SET_POINTER(pressure);

        _SET_POINTER(boundaryNodeIds);
        _SET_POINTER(boundaryNodeType);
        _SET_POINTER(boundaryNodeNormal);
        _SET_POINTER(boundaryNodePos);
        _SET_POINTER(boundaryFacePos);
        _SET_POINTER(boundaryNodeR);
        _SET_POINTER(boundaryFaceR);

        _SET_POINTER(ngb_list);
        _SET_POINTER(ngb_size);
        _SET_POINTER(ngb_offset);

        cudaDeviceSynchronize();
    }
}
