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


#include <algorithm>
#include <thrust/sort.h>
#include "MeshIO.h"
#include "igl/boundary_facets.h"


#include <vtkSmartPointer.h>
#include <vtkGenericDataObjectReader.h>
#include <vtkGenericDataObjectWriter.h>
#include <vtkPoints.h>
#include <vtkTetra.h>
#include <vtkUnstructuredGrid.h>
#include <vtkDataSet.h>
#include <vtkIdList.h>
#include <vtkDataArray.h>
#include <vtkPointData.h>
#include <vtkCellData.h>
#include <vtkFloatArray.h>
#include <vtkFieldData.h>

#include <iostream>

#include "Kernels.h"
#include "MyGUI.h"

#include <set>
#include <unordered_set>
#include <cstdio>
#include <gmsh.h>

namespace LagSol {

    struct MeshData {
        Eigen::MatrixXd nodes;        // (numNodes × 3)
        Eigen::MatrixXi elements;     // (numElements × nodesPerElem)
        Eigen::VectorXi layerId;      // (numElements)
    };


    bool iequals(const std::string& a, const std::string& b) {
        return std::equal(a.begin(), a.end(), b.begin(), b.end(), [](char _a, char _b) { return tolower(_a) == tolower(_b); });
    }

    bool isNumber(const std::string& s) {
        return s.find_first_not_of("0123456789") == std::string::npos;
    }

    // Build vertex -> incident tets adjacency
    // T: #T x 4
    // nV: number of vertices
    std::vector<std::vector<int>> build_vertex_to_tets_adjacency(
        const Eigen::MatrixXi& T,
        int nV) {
        std::vector<std::vector<int>> v2t(nV);

        for (int ti = 0; ti < T.rows(); ++ti) {
            for (int lv = 0; lv < 4; ++lv) {
                int v = T(ti, lv);
                v2t[v].push_back(ti);
            }
        }

        // Sort and unique (just in case)
        for (int v = 0; v < nV; ++v) {
            auto& adj = v2t[v];
            std::sort(adj.begin(), adj.end());
            adj.erase(std::unique(adj.begin(), adj.end()), adj.end());
        }

        return v2t;
    }

    // Intersect two sorted vectors of ints into result (also sorted)
    inline void intersect_two_sorted(
        const std::vector<int>& a,
        const std::vector<int>& b,
        std::vector<int>& result) {
        result.clear();
        size_t i = 0, j = 0;
        while (i < a.size() && j < b.size()) {
            if (a[i] < b[j]) {
                ++i;
            } else if (b[j] < a[i]) {
                ++j;
            } else { // a[i] == b[j]
                result.push_back(a[i]);
                ++i;
                ++j;
            }
        }
    }

    // Given T (tets), F (boundary faces), and number of vertices,
    // return FT: tetra index for each boundary face (F.row(i)).
    Eigen::VectorXi tetra_for_boundary_faces_v2t(
        const Eigen::MatrixXi& T,
        const Eigen::MatrixXi& F,
        int nV) {
        // 1. Build vertex -> incident tets adjacency
        std::vector<std::vector<int>> v2t = build_vertex_to_tets_adjacency(T, nV);

        Eigen::VectorXi FT(F.rows());
        std::vector<int> tmp01;
        std::vector<int> tmp012;

        // 2. For each boundary face, intersect adjacency lists
        for (int fi = 0; fi < F.rows(); ++fi) {
            int v0 = F(fi, 0);
            int v1 = F(fi, 1);
            int v2 = F(fi, 2);

            const auto& A0 = v2t[v0];
            const auto& A1 = v2t[v1];
            const auto& A2 = v2t[v2];

            // Intersect A0 ∩ A1
            intersect_two_sorted(A0, A1, tmp01);
            // Intersect (A0 ∩ A1) ∩ A2
            intersect_two_sorted(tmp01, A2, tmp012);

            if (tmp012.size() != 1) {
                // Degenerate or non-manifold case; handle as needed
                FT[fi] = (tmp012.empty() ? -1 : tmp012[0]);
            } else {
                FT[fi] = tmp012[0];
            }
        }

        return FT;
    }


    Mesh::Mesh() {
        nver = 0;
        ntet = 0;
        ntri = 0;
        nfac = 0;
    }

    void Mesh::clear() {
        nver = 0;
        ntet = 0;
        ntri = 0;
        nfac = 0;

        pos.resize(0, Eigen::NoChange);
        posRef.resize(0, Eigen::NoChange);
        vel.resize(0, Eigen::NoChange);

        tet.resize(0, Eigen::NoChange);
        tri.resize(0, Eigen::NoChange);
        tri_mapped.resize(0, Eigen::NoChange);
        fac.resize(0, Eigen::NoChange);

        layer.resize(0, Eigen::NoChange);
        bdryFun.resize(0, Eigen::NoChange);
        scalarFun.resize(0, Eigen::NoChange);
        scalarActivityFun.resize(0, Eigen::NoChange);

        fiber1.resize(0, Eigen::NoChange);
        fiber2.resize(0, Eigen::NoChange);
        fiber3.resize(0, Eigen::NoChange);
        fiber4.resize(0, Eigen::NoChange);

        Fp.resize(0, Eigen::NoChange);
        Fg.resize(0, Eigen::NoChange);
        lamInt.resize(0, Eigen::NoChange);

        actin1.resize(0, Eigen::NoChange);
        actin1Prod.resize(0, Eigen::NoChange);
        actin1Diff.resize(0, Eigen::NoChange);
        actin1Deca.resize(0, Eigen::NoChange);
        actin2.resize(0, Eigen::NoChange);
        actin2Prod.resize(0, Eigen::NoChange);
        actin2Diff.resize(0, Eigen::NoChange);
        actin2Deca.resize(0, Eigen::NoChange);

        boundaryPos.resize(0, Eigen::NoChange);

        boundaryTetIds.resize(0);
        for (int i=0; i<MAX_NLAYERS; i++)
            boundaryTetIdsLayers[i].clear();
        boundaryNodeIds.clear();
    }

    template<typename T, int N>
    void copy_from_vtk_to_eigen(vtkDataArray* arr, Eigen::Array<T,-1,N,Eigen::RowMajor> &eigen_vector) {
        int nTuples = arr->GetNumberOfTuples();
        int nComp = arr->GetNumberOfComponents();
        int nC = std::min(arr->GetNumberOfComponents(),N);
        eigen_vector.resize(nTuples, N);
        eigen_vector.setZero();
        double tuple[nComp];
        for (int i = 0; i < nTuples; ++i) {
            arr->GetTuple(i, tuple);
            for (int j = 0; j < nC; ++j)
                eigen_vector(i,j) = T(tuple[j]);
        }
    }

    template<typename T>
    void copy_from_vtk_to_eigen(vtkDataArray* arr, Eigen::Array<T,-1,1> &eigen_vector) {
        int nTuples = arr->GetNumberOfTuples();
        if (const int nComp = arr->GetNumberOfComponents(); nComp==1) {
            eigen_vector.resize(nTuples);
            double tuple;
            for (int i = 0; i < nTuples; ++i) {
                arr->GetTuple(i, &tuple);
                eigen_vector(i) = T(tuple);
            }
        }
    }

    template<typename T>
    void copy_from_device_array_to_vtk_array(const thrust::device_vector<T> devData, vtkSmartPointer<vtkFloatArray>& vtkArray) {
        thrust::host_vector<T> hostData(devData);
        int nComp = sizeof(T)/sizeof(Float);
        vtkArray->SetNumberOfComponents(nComp); // 3x3 tensor
        vtkArray->SetNumberOfTuples(hostData.size());
        float tuple[nComp];
        for (vtkIdType i = 0; i < hostData.size(); ++i) {
            for (int j = 0; j < nComp; ++j)
                tuple[j] = hostData[i][j];
            vtkArray->SetTuple(i, tuple);
        }
    }

    template<>
    void copy_from_device_array_to_vtk_array<Float>(const thrust::device_vector<Float> devData, vtkSmartPointer<vtkFloatArray>& vtkArray) {
        thrust::host_vector<Float> hostData(devData);
        int nComp = 1;
        vtkArray->SetNumberOfComponents(nComp); // 3x3 tensor
        vtkArray->SetNumberOfTuples(hostData.size());
        float tuple[nComp];
        for (vtkIdType i = 0; i < hostData.size(); ++i) {
            tuple[0] = hostData[i];
            vtkArray->SetTuple(i, tuple);
        }
    }

    template<>
    void copy_from_device_array_to_vtk_array<int>(const thrust::device_vector<int> devData, vtkSmartPointer<vtkFloatArray>& vtkArray) {
        thrust::host_vector<Float> hostData(devData);
        int nComp = 1;
        vtkArray->SetNumberOfComponents(nComp); // 3x3 tensor
        vtkArray->SetNumberOfTuples(hostData.size());
        float tuple[nComp];
        for (vtkIdType i = 0; i < hostData.size(); ++i) {
            tuple[0] = hostData[i];
            vtkArray->SetTuple(i, tuple);
        }
    }

    void Mesh::print_info() {
        std::cout << "    # nodes : " << nver << std::endl;
        std::cout << "    # facets : " << nfac << std::endl;
        std::cout << "    # triangles : " << ntri << std::endl;
        std::cout << "    # tetrahedrons : " << ntet << std::endl;
        std::cout << "    # boundary nodes : " << nnbd << std::endl;
        std::cout << "    # layers : " << nlay << std::endl;
        if (Fg.size() > 0) std::cout << "    # growth deformation tensor is available." << std::endl;
        if (Fp.size() > 0) std::cout << "    # plastic deformation  tensor is available." << std::endl;
        if (fiber1.size() > 0) std::cout << "    # fiber1 is available." << std::endl;
        if (fiber2.size() > 0) std::cout << "    # fiber2 is available." << std::endl;
        if (fiber3.size() > 0) std::cout << "    # fiber3 is available." << std::endl;
        if (fiber4.size() > 0) std::cout << "    # fiber4 is available." << std::endl;
        if (actin1.size() > 0) std::cout << "    # actin is available." << std::endl;
    }

    void Mesh::extract_boundary_and_face_cell_pairs() {
        if (layer.rows()==0) {
            layer = Eigen::VectorXi(ntet);
            layer.setZero();
        }

        std::vector<int> uniqueTags(layer.data(),layer.data()+layer.size());
        std::sort(uniqueTags.begin(), uniqueTags.end());
        auto last1 = std::unique(uniqueTags.begin(), uniqueTags.end());
        uniqueTags.erase(last1,uniqueTags.end());
        nlay = uniqueTags.size();
        std::map<int,int> tagConv;
        for (int i=0; i<nlay; i++)
            tagConv[uniqueTags[i]] = i + 1;
        for (int i=0; i<ntet; i++) {
            layer(i) = tagConv[layer(i)];
        }

        //////////////////////////////////////////////
        Eigen::Matrix<int, Eigen::Dynamic, 3> otri;
        igl::boundary_facets(tet.matrix(),otri);
        tri = otri.array();
        ntri = tri.rows();
        std::cout<<ntri<<std::endl;
        /////////////////////////////////////////////

        boundaryNodeIds = std::vector<int>(tri.data(), tri.data() + tri.size());
        std::sort(boundaryNodeIds.begin(), boundaryNodeIds.end());
        auto last = std::unique(boundaryNodeIds.begin(), boundaryNodeIds.end());
        boundaryNodeIds.erase(last, boundaryNodeIds.end());
        nnbd = boundaryNodeIds.size();
        std::map<int,int> bdryIdsMap;
        for (int i=0;i<nnbd; i++)
            bdryIdsMap.insert(std::make_pair(boundaryNodeIds[i],i));
        tri_mapped.resize(ntri,3);
        for (int i=0;i<ntri; i++) {
            tri_mapped(i,0) = bdryIdsMap.find(tri(i,0))->second;
            tri_mapped(i,1) = bdryIdsMap.find(tri(i,1))->second;
            tri_mapped(i,2) = bdryIdsMap.find(tri(i,2))->second;
        }
        boundaryPos = pos(boundaryNodeIds, Eigen::all);

        std::vector<std::vector<int>> temp(nver);
        boundaryTetIds.resize(ntri);
//            for (int i=0; i<nver; i++)
//                temp[i].reserve(50);
        for (int i=0; i<ntet; i++) {
            temp[tet(i,0)].push_back(i);
            temp[tet(i,1)].push_back(i);
            temp[tet(i,2)].push_back(i);
            temp[tet(i,3)].push_back(i);
        }
        for (int i=0; i<ntri; i++) {
            int tri0 = tri(i,0);
            int tri1 = tri(i,1);
            int tri2 = tri(i,2);
            std::vector<int> v_intersection1,v_intersection2;
            std::set_intersection(temp[tri0].begin(),temp[tri0].end(),
                                  temp[tri1].begin(),temp[tri1].end(),
                                  std::back_inserter(v_intersection1));
            std::set_intersection(v_intersection1.begin(),v_intersection1.end(),
                                  temp[tri2].begin(),temp[tri2].end(),
                                  std::back_inserter(v_intersection2));
            if (v_intersection2.size()==1) {
                boundaryTetIds[i] = v_intersection2[0];
            } else if (v_intersection2.size()>1) {
                boundaryTetIds[i] = v_intersection2[0];
            } else
                std::cout<<" triangle " << i << " is not connected to any tetrahedron !"<<std::endl;
        }
        for (int i=0; i<ntri; i++) {
            int l = layer(boundaryTetIds[i]) - 1;
            boundaryTetIdsLayers[l].push_back(i);
        }

        std::map<std::set<int>, std::tuple<int,int,int>> face_cell_pairs;
        for (int i=0; i<ntet; i++) {
            int a = tet(i,0);
            int b = tet(i,1);
            int c = tet(i,2);
            int d = tet(i,3);
            auto f0 = std::set({a,b,c});
            auto f1 = std::set({a,c,d});
            auto f2 = std::set({a,b,d});
            auto f3 = std::set({b,c,d});

            if (auto iter = face_cell_pairs.find(f0); iter == face_cell_pairs.end())
                face_cell_pairs.insert(std::make_pair(f0,std::make_tuple(0,i,-1)));
            else
                std::get<2>(iter->second) = i;

            if (auto iter = face_cell_pairs.find(f1); iter == face_cell_pairs.end())
                face_cell_pairs.insert(std::make_pair(f1,std::make_tuple(1,i,-1)));
            else
                std::get<2>(iter->second) = i;

            if (auto iter = face_cell_pairs.find(f2); iter == face_cell_pairs.end())
                face_cell_pairs.insert(std::make_pair(f2,std::make_tuple(2,i,-1)));
            else
                std::get<2>(iter->second) = i;

            if (auto iter = face_cell_pairs.find(f3); iter == face_cell_pairs.end())
                face_cell_pairs.insert(std::make_pair(f3,std::make_tuple(3,i,-1)));
            else
                std::get<2>(iter->second) = i;
        }

        nfac = face_cell_pairs.size();
        fac.resize(nfac,3);
        int i = 0;
        for (auto iter=face_cell_pairs.begin(); iter != face_cell_pairs.end(); ++iter, ++i) {
            fac(i,0) = std::get<0>(iter->second);
            fac(i,1) = std::get<1>(iter->second);
            fac(i,2) = std::get<2>(iter->second);
        }
    }

    enum class Origin { Object, Tool, Both };

    Origin classifyOrigin(
        const std::vector<std::pair<int,int>> &object,
        const std::vector<std::pair<int,int>> &tool,
        const std::vector<std::pair<int,int>> &sources)
    {
        bool fromObject = false;
        bool fromTool   = false;

        for(const auto &src : sources) {
            if(std::find(object.begin(), object.end(), src) != object.end())
                fromObject = true;
            if(std::find(tool.begin(), tool.end(), src) != tool.end())
                fromTool = true;
        }

        if(fromObject && fromTool) return Origin::Both;
        if(fromObject) return Origin::Object;
        return Origin::Tool;
    }


    int Mesh::init_from_multi_layer_torus(const double &R, const std::vector<double> &radii, const std::vector<double> &meshSizes) {
        clear();
        Eigen::MatrixXd compactNodes;
        Eigen::MatrixXi compactElements;
        Eigen::VectorXi layerId;

        // -----------------------
        // 1. Validate input
        // -----------------------
        if (radii.size() < 2) {
            throw std::invalid_argument("radii must contain at least 2 values.");
        }
        if (meshSizes.size() + 1 != radii.size()) {
            throw std::invalid_argument("meshSizes.size() must be radii.size() - 1.");
        }
        for (std::size_t i = 1; i < radii.size(); ++i) {
            if (!(radii[i] > radii[i - 1])) {
                throw std::invalid_argument("radii must be strictly increasing.");
            }
        }

        // -----------------------
        // 2. Initialize Gmsh
        // -----------------------
        gmsh::initialize();
        gmsh::option::setNumber("General.Terminal", 1);
        gmsh::model::add("layered_torus");

        std::vector<gmsh::vectorpair> outDimTagsMap;
        std::vector<std::pair<int,int>> torus;
        if (radii[0] > 0.0) {
            for (std::size_t i = 1; i < radii.size(); ++i) {
                gmsh::vectorpair innerTorus;
                innerTorus.emplace_back(3, gmsh::model::occ::addTorus(0.0, 0.0, 0.0,  R, radii[0]));
                gmsh::vectorpair outerTorus;
                outerTorus.emplace_back(3,gmsh::model::occ::addTorus(0.0, 0.0, 0.0,  R, radii[i]));
                gmsh::vectorpair outDimTags;
                gmsh::model::occ::cut(outerTorus,innerTorus, outDimTags, outDimTagsMap);
                torus.push_back(outDimTags.back());
            }
        } else {
            for (std::size_t i = 1; i < radii.size(); ++i)
                torus.emplace_back(3,gmsh::model::occ::addTorus(0.0, 0.0, 0.0,  R, radii[i]));
        }

        gmsh::model::occ::synchronize();

        // -----------------------------
        // 2. Fragment all spheres at once
        // -----------------------------
        std::vector<std::pair<int,int>> outDimTags(torus);
        std::vector<std::vector<std::pair<int,int>>> outMap;

        if (torus.size()>1) {
            gmsh::model::occ::fragment(torus, {}, outDimTags, outMap);
            gmsh::model::occ::synchronize();
        }

        // -----------------------
        // 8. Define radius-based mesh-size fields per shell
        // -----------------------

        std::vector<int> thresholdFields(meshSizes.size());
        for (std::size_t i = 0; i < meshSizes.size(); ++i) {
            double dx_i = meshSizes[i];

            int fTh = gmsh::model::mesh::field::add("Constant");
            gmsh::model::mesh::field::setNumber(fTh, "VIn", dx_i);
            gmsh::model::mesh::field::setNumbers(fTh, "VolumesList", {double(outDimTags[i].second)});
            thresholdFields[i] = fTh;
        }

        int fMin = gmsh::model::mesh::field::add("Min");
        {
            std::vector<double> list;
            list.reserve(thresholdFields.size());
            for (int id : thresholdFields) list.push_back(static_cast<double>(id));
            gmsh::model::mesh::field::setNumbers(fMin, "FieldsList", list);
        }
        gmsh::model::mesh::field::setAsBackgroundMesh(fMin);

        // -----------------------
        // 9. Generate final 3D mesh
        // -----------------------
        gmsh::model::mesh::generate(3);
        gmsh::model::mesh::optimize("Netgen");

        // -----------------------
        // 10. Extract nodes and build compact index
        // -----------------------
        std::vector<std::size_t> nodeTags;
        std::vector<double> nodeCoords, nodeParams;
        gmsh::model::mesh::getNodes(nodeTags, nodeCoords, nodeParams);

        const std::size_t nnodes = nodeTags.size();
        if (nodeCoords.size() != 3 * nnodes) {
            throw std::runtime_error("Inconsistent node coordinate size.");
        }

        compactNodes.resize(static_cast<int>(nnodes), 3);
        std::unordered_map<std::size_t, int> tagToCompact;
        tagToCompact.reserve(nnodes);
        for (std::size_t i = 0; i < nnodes; ++i) {
            tagToCompact[nodeTags[i]] = static_cast<int>(i);
            compactNodes(static_cast<int>(i), 0) = nodeCoords[3*i + 0];
            compactNodes(static_cast<int>(i), 1) = nodeCoords[3*i + 1];
            compactNodes(static_cast<int>(i), 2) = nodeCoords[3*i + 2];
        }

        // -----------------------
        // 11. Extract tetrahedra (element type 4)
        // -----------------------
        std::vector<int> elemTypes;
        std::vector<std::vector<std::size_t>> elemTags;
        std::vector<std::vector<std::size_t>> elemNodeTags;
        gmsh::model::mesh::getElements(elemTypes, elemTags, elemNodeTags);

        const int tetType = 4;
        std::vector<std::size_t> tetTags;
        std::vector<std::size_t> tetNodeTags;
        for (std::size_t i = 0; i < elemTypes.size(); ++i) {
            if (elemTypes[i] == tetType) {
                tetTags = elemTags[i];
                tetNodeTags = elemNodeTags[i];
                break;
            }
        }

        const std::size_t nelems = tetTags.size();
        if (nelems == 0) throw std::runtime_error("No tetrahedral elements found.");
        if (tetNodeTags.size() != 4 * nelems) throw std::runtime_error("Unexpected tetrahedral connectivity size.");

        compactElements.resize(static_cast<int>(nelems), 4);
        for (std::size_t e = 0; e < nelems; ++e) {
            for (int k = 0; k < 4; ++k) {
                std::size_t nt = tetNodeTags[4*e + k];
                auto it = tagToCompact.find(nt);
                if (it == tagToCompact.end()) throw std::runtime_error("Node tag not found in compact map.");
                compactElements(static_cast<int>(e), k) = it->second;
            }
        }

        // -----------------------
        // 12. Compute per-element layer ID by average radius
        // -----------------------
        layerId.resize(static_cast<int>(nelems));
        const double tol = 1e-8 * radii.back();

        for (std::size_t e = 0; e < nelems; ++e) {
            double rAvg = 0.0;
            for (int k = 0; k < 4; ++k) {
                int nid = compactElements(static_cast<int>(e), k);
                double x = compactNodes(nid, 0);
                double y = compactNodes(nid, 1);
                double z = compactNodes(nid, 2);

                rAvg += std::sqrt((sqrt(x*x + y*y)-R)*(sqrt(x*x + y*y)-R)+z*z);
            }
            rAvg /= 4.0;

            int lid = 0;
            int nShells = static_cast<int>(radii.size());
            for (std::size_t i = 0; i < nShells; ++i) {
                double rIn = radii[i];
                double rOut = radii[i+1];
                bool inLayer = (rAvg >= rIn - tol) && ((i < nShells - 1 && rAvg < rOut + tol) || (i == nShells - 1 && rAvg <= rOut + tol));
                if (inLayer) { lid = static_cast<int>(i); break; }
            }
            // if (lid < 0) throw std::runtime_error("Could not assign layer ID to an element.");
            layerId(static_cast<int>(e)) = lid;
            // layerId(static_cast<int>(e)) = 0;
        }
        // -----------------------
        // 12. Finalize Gmsh
        // -----------------------
        gmsh::finalize();

        pos = compactNodes.cast<Float>();
        tet = compactElements.cast<int>();
        layer = layerId.cast<int>();

        nver = pos.rows();
        ntet = tet.rows();

        extract_boundary_and_face_cell_pairs();
        print_info();

        return 1;
    }

    int Mesh::init_from_multi_layer_tube(const double &L, const std::vector<double> &radii, const std::vector<double> &meshSizes) {
        clear();
        Eigen::MatrixXd compactNodes;
        Eigen::MatrixXi compactElements;
        Eigen::VectorXi layerId;

        // -----------------------
        // 1. Validate input
        // -----------------------
        if (radii.size() < 2) {
            throw std::invalid_argument("radii must contain at least 2 values.");
        }
        if (meshSizes.size() + 1 != radii.size()) {
            throw std::invalid_argument("meshSizes.size() must be radii.size() - 1.");
        }
        for (std::size_t i = 1; i < radii.size(); ++i) {
            if (!(radii[i] > radii[i - 1])) {
                throw std::invalid_argument("radii must be strictly increasing.");
            }
        }

        // -----------------------
        // 2. Initialize Gmsh
        // -----------------------
        gmsh::initialize();
        gmsh::option::setNumber("General.Terminal", 1);
        gmsh::model::add("layered_tube");

        std::vector<gmsh::vectorpair> outDimTagsMap;
        std::vector<std::pair<int,int>> cylinders;
        if (radii[0] > 0.0) {
            for (std::size_t i = 1; i < radii.size(); ++i) {
                gmsh::vectorpair innerCylinder;
                innerCylinder.emplace_back(3, gmsh::model::occ::addCylinder(0.0, -L/2.0, 0.0, 0, L, 0, radii[0]));
                gmsh::vectorpair outerCylinder;
                outerCylinder.emplace_back(3,gmsh::model::occ::addCylinder(0.0, -L/2.0, 0.0, 0, L, 0, radii[i]));
                gmsh::vectorpair outDimTags;
                gmsh::model::occ::cut(
                    outerCylinder,
                    innerCylinder, outDimTags, outDimTagsMap);
                cylinders.push_back(outDimTags.back());
            }
        } else {
            for (std::size_t i = 1; i < radii.size(); ++i)
                cylinders.emplace_back(3,gmsh::model::occ::addCylinder(0.0, -L/2.0, 0.0, 0, L, 0, radii[i]));
        }

        gmsh::model::occ::synchronize();

        // -----------------------------
        // 2. Fragment all spheres at once
        // -----------------------------
        std::vector<std::pair<int,int>> outDimTags(cylinders);
        std::vector<std::vector<std::pair<int,int>>> outMap;

        if (cylinders.size()>1) {
            gmsh::model::occ::fragment(cylinders, {}, outDimTags, outMap);
            gmsh::model::occ::synchronize();
        }

        // -----------------------
        // 8. Define radius-based mesh-size fields per shell
        // -----------------------

        std::vector<int> thresholdFields(meshSizes.size());
        for (std::size_t i = 0; i < meshSizes.size(); ++i) {
            double dx_i = meshSizes[i];

            int fTh = gmsh::model::mesh::field::add("Constant");
            gmsh::model::mesh::field::setNumber(fTh, "VIn", dx_i);
            gmsh::model::mesh::field::setNumbers(fTh, "VolumesList", {double(outDimTags[i].second)});
            thresholdFields[i] = fTh;
        }

        int fMin = gmsh::model::mesh::field::add("Min");
        {
            std::vector<double> list;
            list.reserve(thresholdFields.size());
            for (int id : thresholdFields) list.push_back(static_cast<double>(id));
            gmsh::model::mesh::field::setNumbers(fMin, "FieldsList", list);
        }
        gmsh::model::mesh::field::setAsBackgroundMesh(fMin);

        // -----------------------
        // 9. Generate final 3D mesh
        // -----------------------
        gmsh::model::mesh::generate(3);
        gmsh::model::mesh::optimize("Netgen");

        // -----------------------
        // 10. Extract nodes and build compact index
        // -----------------------
        std::vector<std::size_t> nodeTags;
        std::vector<double> nodeCoords, nodeParams;
        gmsh::model::mesh::getNodes(nodeTags, nodeCoords, nodeParams);

        const std::size_t nnodes = nodeTags.size();
        if (nodeCoords.size() != 3 * nnodes) {
            throw std::runtime_error("Inconsistent node coordinate size.");
        }

        compactNodes.resize(static_cast<int>(nnodes), 3);
        std::unordered_map<std::size_t, int> tagToCompact;
        tagToCompact.reserve(nnodes);
        for (std::size_t i = 0; i < nnodes; ++i) {
            tagToCompact[nodeTags[i]] = static_cast<int>(i);
            compactNodes(static_cast<int>(i), 0) = nodeCoords[3*i + 0];
            compactNodes(static_cast<int>(i), 1) = nodeCoords[3*i + 1];
            compactNodes(static_cast<int>(i), 2) = nodeCoords[3*i + 2];
        }

        // -----------------------
        // 11. Extract tetrahedra (element type 4)
        // -----------------------
        std::vector<int> elemTypes;
        std::vector<std::vector<std::size_t>> elemTags;
        std::vector<std::vector<std::size_t>> elemNodeTags;
        gmsh::model::mesh::getElements(elemTypes, elemTags, elemNodeTags);

        const int tetType = 4;
        std::vector<std::size_t> tetTags;
        std::vector<std::size_t> tetNodeTags;
        for (std::size_t i = 0; i < elemTypes.size(); ++i) {
            if (elemTypes[i] == tetType) {
                tetTags = elemTags[i];
                tetNodeTags = elemNodeTags[i];
                break;
            }
        }

        const std::size_t nelems = tetTags.size();
        if (nelems == 0) throw std::runtime_error("No tetrahedral elements found.");
        if (tetNodeTags.size() != 4 * nelems) throw std::runtime_error("Unexpected tetrahedral connectivity size.");

        compactElements.resize(static_cast<int>(nelems), 4);
        for (std::size_t e = 0; e < nelems; ++e) {
            for (int k = 0; k < 4; ++k) {
                std::size_t nt = tetNodeTags[4*e + k];
                auto it = tagToCompact.find(nt);
                if (it == tagToCompact.end()) throw std::runtime_error("Node tag not found in compact map.");
                compactElements(static_cast<int>(e), k) = it->second;
            }
        }

        // -----------------------
        // 12. Compute per-element layer ID by average radius
        // -----------------------
        layerId.resize(static_cast<int>(nelems));
        const double tol = 1e-8 * radii.back();

        for (std::size_t e = 0; e < nelems; ++e) {
            double rAvg = 0.0;
            for (int k = 0; k < 4; ++k) {
                int nid = compactElements(static_cast<int>(e), k);
                double x = compactNodes(nid, 0);
                double y = compactNodes(nid, 1);
                double z = compactNodes(nid, 2);
                rAvg += std::sqrt(x*x + z*z);
            }
            rAvg /= 4.0;

            int lid = 0;
            int nShells = static_cast<int>(radii.size());
            for (std::size_t i = 0; i < nShells; ++i) {
                double rIn = radii[i];
                double rOut = radii[i+1];
                bool inLayer = (rAvg >= rIn - tol) && ((i < nShells - 1 && rAvg < rOut + tol) || (i == nShells - 1 && rAvg <= rOut + tol));
                if (inLayer) { lid = static_cast<int>(i); break; }
            }
            // if (lid < 0) throw std::runtime_error("Could not assign layer ID to an element.");
            layerId(static_cast<int>(e)) = lid;
            // layerId(static_cast<int>(e)) = 0;
        }
        // -----------------------
        // 12. Finalize Gmsh
        // -----------------------
        gmsh::finalize();

        pos = compactNodes.cast<Float>();
        tet = compactElements.cast<int>();
        layer = layerId.cast<int>();

        nver = pos.rows();
        ntet = tet.rows();

        extract_boundary_and_face_cell_pairs();
        print_info();

        return 1;
    }

    int Mesh::init_from_multi_layer_cone(const double &L, const double &apexAng, const std::vector<double> &radii, const std::vector<double> &meshSizes) {
        clear();
        Eigen::MatrixXd compactNodes;
        Eigen::MatrixXi compactElements;
        Eigen::VectorXi layerId;

        // -----------------------
        // 1. Validate input
        // -----------------------
        if (radii.size() < 2) {
            throw std::invalid_argument("radii must contain at least 2 values.");
        }
        if (meshSizes.size() + 1 != radii.size()) {
            throw std::invalid_argument("meshSizes.size() must be radii.size() - 1.");
        }
        for (std::size_t i = 1; i < radii.size(); ++i) {
            if (!(radii[i] > radii[i - 1])) {
                throw std::invalid_argument("radii must be strictly increasing.");
            }
        }

        // -----------------------
        // 2. Initialize Gmsh
        // -----------------------
        gmsh::initialize();
        gmsh::option::setNumber("General.Terminal", 1);
        gmsh::model::add("layered_cones");

        std::vector<gmsh::vectorpair> outDimTagsMap;
        std::vector<std::pair<int,int>> cones;

        if (radii[0] > 0.0) {
            double  R0 = radii[0];
            double  L0 = R0/std::tan(apexAng/2.0);
            double  r0 = std::max(R0 - L * std::tan(apexAng/2.0), 0.0 * radii[0]);
            for (std::size_t i = 1; i < radii.size(); ++i) {

                double  Ri = radii[i];
                double  Li = Ri/std::tan(apexAng/2.0);
                double  ri = std::max(Ri - L * std::tan(apexAng/2.0), 0.0 * radii[0]);

                gmsh::vectorpair innerCone;
                innerCone.emplace_back(3,gmsh::model::occ::addCone(0.0, 0.0, 0.0, 0, std::min(L, L0), 0, R0, r0));
                gmsh::vectorpair outerCone;
                outerCone.emplace_back(3,gmsh::model::occ::addCone(0.0, 0.0, 0.0, 0, std::min(L, Li), 0, Ri, ri));
                gmsh::vectorpair outDimTags;
                gmsh::model::occ::cut(
                    outerCone,
                    innerCone, outDimTags, outDimTagsMap);
                cones.push_back(outDimTags.back());
            }
        } else {
            for (std::size_t i = 1; i < radii.size(); ++i) {
                double  Ri = radii[i];
                double  Li = Ri/std::tan(apexAng/2.0);
                double  ri = std::max(Ri - L * std::tan(apexAng/2.0), 0.0 * radii[0]);
                cones.emplace_back(3,gmsh::model::occ::addCone(0.0, 0.0, 0.0, 0, std::min(L, Li), 0, Ri, ri));
            }
        }

        gmsh::model::occ::synchronize();

        // -----------------------------
        // 2. Fragment all spheres at once
        // -----------------------------
        std::vector<std::pair<int,int>> outDimTags(cones);
        std::vector<std::vector<std::pair<int,int>>> outMap;

        if (cones.size()>1) {
            gmsh::model::occ::fragment(cones, {}, outDimTags, outMap);
            gmsh::model::occ::synchronize();
        }

        // -----------------------
        // 8. Define radius-based mesh-size fields per shell
        // -----------------------

        std::vector<int> thresholdFields(meshSizes.size());
        for (std::size_t i = 0; i < meshSizes.size(); ++i) {
            double dx_i = meshSizes[i];

            int fTh = gmsh::model::mesh::field::add("Constant");
            gmsh::model::mesh::field::setNumber(fTh, "VIn", dx_i);
            gmsh::model::mesh::field::setNumbers(fTh, "VolumesList", {double(outDimTags[i].second)});
            thresholdFields[i] = fTh;
        }

        int fMin = gmsh::model::mesh::field::add("Min");
        {
            std::vector<double> list;
            list.reserve(thresholdFields.size());
            for (int id : thresholdFields) list.push_back(static_cast<double>(id));
            gmsh::model::mesh::field::setNumbers(fMin, "FieldsList", list);
        }
        gmsh::model::mesh::field::setAsBackgroundMesh(fMin);

        // -----------------------
        // 9. Generate final 3D mesh
        // -----------------------
        gmsh::model::mesh::generate(3);
        gmsh::model::mesh::optimize("Netgen");

        // -----------------------
        // 10. Extract nodes and build compact index
        // -----------------------
        std::vector<std::size_t> nodeTags;
        std::vector<double> nodeCoords, nodeParams;
        gmsh::model::mesh::getNodes(nodeTags, nodeCoords, nodeParams);

        const std::size_t nnodes = nodeTags.size();
        if (nodeCoords.size() != 3 * nnodes) {
            throw std::runtime_error("Inconsistent node coordinate size.");
        }

        compactNodes.resize(static_cast<int>(nnodes), 3);
        std::unordered_map<std::size_t, int> tagToCompact;
        tagToCompact.reserve(nnodes);
        for (std::size_t i = 0; i < nnodes; ++i) {
            tagToCompact[nodeTags[i]] = static_cast<int>(i);
            compactNodes(static_cast<int>(i), 0) = nodeCoords[3*i + 0];
            compactNodes(static_cast<int>(i), 1) = nodeCoords[3*i + 1];
            compactNodes(static_cast<int>(i), 2) = nodeCoords[3*i + 2];
        }

        // -----------------------
        // 11. Extract tetrahedra (element type 4)
        // -----------------------
        std::vector<int> elemTypes;
        std::vector<std::vector<std::size_t>> elemTags;
        std::vector<std::vector<std::size_t>> elemNodeTags;
        gmsh::model::mesh::getElements(elemTypes, elemTags, elemNodeTags);

        const int tetType = 4;
        std::vector<std::size_t> tetTags;
        std::vector<std::size_t> tetNodeTags;
        for (std::size_t i = 0; i < elemTypes.size(); ++i) {
            if (elemTypes[i] == tetType) {
                tetTags = elemTags[i];
                tetNodeTags = elemNodeTags[i];
                break;
            }
        }

        const std::size_t nelems = tetTags.size();
        if (nelems == 0) throw std::runtime_error("No tetrahedral elements found.");
        if (tetNodeTags.size() != 4 * nelems) throw std::runtime_error("Unexpected tetrahedral connectivity size.");

        compactElements.resize(static_cast<int>(nelems), 4);
        for (std::size_t e = 0; e < nelems; ++e) {
            for (int k = 0; k < 4; ++k) {
                std::size_t nt = tetNodeTags[4*e + k];
                auto it = tagToCompact.find(nt);
                if (it == tagToCompact.end()) throw std::runtime_error("Node tag not found in compact map.");
                compactElements(static_cast<int>(e), k) = it->second;
            }
        }

        // -----------------------
        // 12. Compute per-element layer ID by average radius
        // -----------------------
        layerId.resize(static_cast<int>(nelems));
        const double tol = 1e-8 * radii.back();

        for (std::size_t e = 0; e < nelems; ++e) {
            double rAvg = 0.0;
            for (int k = 0; k < 4; ++k) {
                int nid = compactElements(static_cast<int>(e), k);
                double x = compactNodes(nid, 0);
                double y = compactNodes(nid, 1);
                double z = compactNodes(nid, 2);

                // double rho = std::sqrt(x*x + z*z) / std::sin(apexAng/2.0);
                // double theta = atan2(z, x);
                double rMax = std::max(radii.back()  - y * std::tan(apexAng/2.0),0.0);
                double n = std::sqrt(x*x + z*z);
                rAvg += n  + radii.back() - rMax;

            }
            rAvg /= 4.0;

            int lid = 0;
            int nShells = static_cast<int>(radii.size());
            for (std::size_t i = 0; i < nShells; ++i) {
                double rIn = radii[i];
                double rOut = radii[i+1];
                bool inLayer = (rAvg >= rIn - tol) && ((i < nShells - 1 && rAvg < rOut + tol) || (i == nShells - 1 && rAvg <= rOut + tol));
                if (inLayer) { lid = static_cast<int>(i); break; }
            }
            // if (lid < 0) throw std::runtime_error("Could not assign layer ID to an element.");
            layerId(static_cast<int>(e)) = lid;
            // layerId(static_cast<int>(e)) = 0;
        }
        // -----------------------
        // 12. Finalize Gmsh
        // -----------------------
        gmsh::finalize();

        pos = compactNodes.cast<Float>();
        tet = compactElements.cast<int>();
        layer = layerId.cast<int>();

        nver = pos.rows();
        ntet = tet.rows();

        extract_boundary_and_face_cell_pairs();
        print_info();

        return 1;
    }


    int Mesh::init_from_multi_layer_sphere(const std::vector<double> &radii, const std::vector<double> &meshSizes) {
        clear();
        Eigen::MatrixXd compactNodes;
        Eigen::MatrixXi compactElements;
        Eigen::VectorXi layerId;

        // -----------------------
        // 1. Validate input
        // -----------------------
        if (radii.size() < 2) {
            throw std::invalid_argument("radii must contain at least 2 values.");
        }
        if (meshSizes.size() + 1 != radii.size()) {
            throw std::invalid_argument("meshSizes.size() must be radii.size() - 1.");
        }
        for (std::size_t i = 1; i < radii.size(); ++i) {
            if (!(radii[i] > radii[i - 1])) {
                throw std::invalid_argument("radii must be strictly increasing.");
            }
        }

        // -----------------------
        // 2. Initialize Gmsh
        // -----------------------
        gmsh::initialize();
        gmsh::option::setNumber("General.Terminal", 1);
        gmsh::model::add("layered_sphere");

        std::vector<gmsh::vectorpair> outDimTagsMap;
        std::vector<std::pair<int,int>> spheres;
        if (radii[0] > 0.0) {
            for (std::size_t i = 1; i < radii.size(); ++i) {
                gmsh::vectorpair innerSphere;
                innerSphere.emplace_back(3, gmsh::model::occ::addSphere(0.0, 0.0, 0.0, radii[0]));
                gmsh::vectorpair outerSphere;
                outerSphere.emplace_back(3,gmsh::model::occ::addSphere(0.0, 0.0, 0.0, radii[i]));
                gmsh::vectorpair outDimTags;
                gmsh::model::occ::cut(
                    outerSphere,
                    innerSphere, outDimTags, outDimTagsMap);
                spheres.push_back(outDimTags.back());
            }
        } else {
            for (std::size_t i = 1; i < radii.size(); ++i)
                spheres.emplace_back(3,gmsh::model::occ::addSphere(0.0, 0.0, 0.0, radii[i]));
        }

        gmsh::model::occ::synchronize();

        // -----------------------------
        // 2. Fragment all spheres at once
        // -----------------------------
        std::vector<std::pair<int,int>> outDimTags(spheres);
        std::vector<std::vector<std::pair<int,int>>> outMap;

        if (spheres.size()>1) {
            gmsh::model::occ::fragment(spheres, {}, outDimTags, outMap);
            gmsh::model::occ::synchronize();
        }


        // -----------------------
        // 8. Define radius-based mesh-size fields per shell
        // -----------------------
        // Distance from origin
        int fDist = gmsh::model::mesh::field::add("MathEval");
        gmsh::model::mesh::field::setString(fDist, "F", "sqrt(x*x + y*y + z*z)");

        std::vector<int> thresholdFields(meshSizes.size());
        for (std::size_t i = 0; i < meshSizes.size(); ++i) {
            double dx_i = meshSizes[i];

            int fTh = gmsh::model::mesh::field::add("Constant");
            gmsh::model::mesh::field::setNumber(fTh, "VIn", dx_i);
            gmsh::model::mesh::field::setNumbers(fTh, "VolumesList", {double(outDimTags[i].second)});
            thresholdFields[i] = fTh;
        }
        int fMin = gmsh::model::mesh::field::add("Min");
        {
            std::vector<double> list;
            list.reserve(thresholdFields.size());
            for (int id : thresholdFields) list.push_back(static_cast<double>(id));
            gmsh::model::mesh::field::setNumbers(fMin, "FieldsList", list);
        }
        gmsh::model::mesh::field::setAsBackgroundMesh(fMin);

        // -----------------------
        // 9. Generate final 3D mesh
        // -----------------------
        gmsh::model::mesh::generate(3);
        gmsh::model::mesh::optimize("Netgen");

        // -----------------------
        // 10. Extract nodes and build compact index
        // -----------------------
        std::vector<std::size_t> nodeTags;
        std::vector<double> nodeCoords, nodeParams;
        gmsh::model::mesh::getNodes(nodeTags, nodeCoords, nodeParams);

        const std::size_t nnodes = nodeTags.size();
        if (nodeCoords.size() != 3 * nnodes) {
            throw std::runtime_error("Inconsistent node coordinate size.");
        }

        compactNodes.resize(static_cast<int>(nnodes), 3);
        std::unordered_map<std::size_t, int> tagToCompact;
        tagToCompact.reserve(nnodes);
        for (std::size_t i = 0; i < nnodes; ++i) {
            tagToCompact[nodeTags[i]] = static_cast<int>(i);
            compactNodes(static_cast<int>(i), 0) = nodeCoords[3*i + 0];
            compactNodes(static_cast<int>(i), 1) = nodeCoords[3*i + 1];
            compactNodes(static_cast<int>(i), 2) = nodeCoords[3*i + 2];
        }

        // -----------------------
        // 11. Extract tetrahedra (element type 4)
        // -----------------------
        std::vector<int> elemTypes;
        std::vector<std::vector<std::size_t>> elemTags;
        std::vector<std::vector<std::size_t>> elemNodeTags;
        gmsh::model::mesh::getElements(elemTypes, elemTags, elemNodeTags);

        const int tetType = 4;
        std::vector<std::size_t> tetTags;
        std::vector<std::size_t> tetNodeTags;
        for (std::size_t i = 0; i < elemTypes.size(); ++i) {
            if (elemTypes[i] == tetType) {
                tetTags = elemTags[i];
                tetNodeTags = elemNodeTags[i];
                break;
            }
        }

        const std::size_t nelems = tetTags.size();
        if (nelems == 0) throw std::runtime_error("No tetrahedral elements found.");
        if (tetNodeTags.size() != 4 * nelems) throw std::runtime_error("Unexpected tetrahedral connectivity size.");

        compactElements.resize(static_cast<int>(nelems), 4);
        for (std::size_t e = 0; e < nelems; ++e) {
            for (int k = 0; k < 4; ++k) {
                std::size_t nt = tetNodeTags[4*e + k];
                auto it = tagToCompact.find(nt);
                if (it == tagToCompact.end()) throw std::runtime_error("Node tag not found in compact map.");
                compactElements(static_cast<int>(e), k) = it->second;
            }
        }

        // -----------------------
        // 12. Compute per-element layer ID by average radius
        // -----------------------
        layerId.resize(static_cast<int>(nelems));
        const double tol = 1e-8 * radii.back();

        for (std::size_t e = 0; e < nelems; ++e) {
            double rAvg = 0.0;
            for (int k = 0; k < 4; ++k) {
                int nid = compactElements(static_cast<int>(e), k);
                double x = compactNodes(nid, 0);
                double y = compactNodes(nid, 1);
                double z = compactNodes(nid, 2);
                rAvg += std::sqrt(x*x + y*y + z*z);
            }
            rAvg /= 4.0;

            int lid = 0;
            for (std::size_t i = 0; i < meshSizes.size(); ++i) {
                double rIn = radii[i];
                double rOut = radii[i+1];
                bool inLayer = (rAvg >= rIn - tol) && ((i < meshSizes.size() - 1 && rAvg < rOut + tol) || (i == meshSizes.size() - 1 && rAvg <= rOut + tol));
                if (inLayer) { lid = static_cast<int>(i); break; }
            }
            // if (lid < 0) throw std::runtime_error("Could not assign layer ID to an element.");
            layerId(static_cast<int>(e)) = lid;
            // layerId(static_cast<int>(e)) = 0;
        }
        // -----------------------
        // 12. Finalize Gmsh
        // -----------------------
        gmsh::finalize();

        pos = compactNodes.cast<Float>();
        tet = compactElements.cast<int>();
        layer = layerId.cast<int>();

        nver = pos.rows();
        ntet = tet.rows();

        extract_boundary_and_face_cell_pairs();
        print_info();

        return 1;
    }


    int Mesh::init_from_multi_layer_disk(double R, const std::vector<double> &thicknesses, const std::vector<double> &meshSizes) {
        clear();
        Eigen::MatrixXd compactNodes;
        Eigen::MatrixXi compactElements;
        Eigen::VectorXi layerId;

        // -----------------------
        // 1. Validate input
        // -----------------------

        if (thicknesses.empty())
            throw std::runtime_error("thicknesses must not be empty.");
        if (meshSizes.size() != thicknesses.size())
            throw std::runtime_error("meshSizes and thicknesses must have same size.");

        // -----------------------
        // 2. Initialize Gmsh
        // -----------------------
        gmsh::initialize();
        gmsh::option::setNumber("General.Terminal", 1);
        gmsh::model::add("multi_layer_tet_fragmented");


        std::vector<gmsh::vectorpair> outDimTagsMap;
        std::vector<std::pair<int,int>> disks;

        auto heights = thicknesses;
        std::inclusive_scan(thicknesses.begin(), thicknesses.end(), heights.begin());
        for (std::size_t i = 0; i < heights.size(); ++i)
            disks.emplace_back(3,gmsh::model::occ::addCylinder(0, 0, 0, 0, 0, heights[i], R));

        gmsh::model::occ::synchronize();

        // -----------------------------
        // 2. Fragment all boxes at once
        // -----------------------------
        std::vector<std::pair<int,int>> outDimTags(disks);
        std::vector<std::vector<std::pair<int,int>>> outMap;

        if (disks.size()>1) {
            gmsh::model::occ::fragment(disks, {}, outDimTags, outMap);
            gmsh::model::occ::synchronize();
        }

        // -----------------------
        // 8. Define radius-based mesh-size fields per shell
        // -----------------------

        std::vector<int> thresholdFields(meshSizes.size());
        for (std::size_t i = 0; i < meshSizes.size(); ++i) {
            double dx_i = meshSizes[i];

            int fTh = gmsh::model::mesh::field::add("Constant");
            gmsh::model::mesh::field::setNumber(fTh, "VIn", dx_i);
            gmsh::model::mesh::field::setNumbers(fTh, "VolumesList", {double(outDimTags[i].second)});
            thresholdFields[i] = fTh;
        }

        int fMin = gmsh::model::mesh::field::add("Min");
        {
            std::vector<double> list;
            list.reserve(thresholdFields.size());
            for (int id : thresholdFields) list.push_back(static_cast<double>(id));
            gmsh::model::mesh::field::setNumbers(fMin, "FieldsList", list);
        }
        gmsh::model::mesh::field::setAsBackgroundMesh(fMin);

        // -----------------------
        // 9. Generate final 3D mesh
        // -----------------------
        gmsh::model::mesh::generate(3);
        gmsh::model::mesh::optimize("Netgen");

        // -----------------------
        // 10. Extract nodes and build compact index
        // -----------------------
        std::vector<std::size_t> nodeTags;
        std::vector<double> nodeCoords, nodeParams;
        gmsh::model::mesh::getNodes(nodeTags, nodeCoords, nodeParams);

        const std::size_t nnodes = nodeTags.size();
        if (nodeCoords.size() != 3 * nnodes) {
            throw std::runtime_error("Inconsistent node coordinate size.");
        }

        compactNodes.resize(static_cast<int>(nnodes), 3);
        std::unordered_map<std::size_t, int> tagToCompact;
        tagToCompact.reserve(nnodes);
        for (std::size_t i = 0; i < nnodes; ++i) {
            tagToCompact[nodeTags[i]] = static_cast<int>(i);
            compactNodes(static_cast<int>(i), 0) = nodeCoords[3*i + 0];
            compactNodes(static_cast<int>(i), 1) = nodeCoords[3*i + 1];
            compactNodes(static_cast<int>(i), 2) = nodeCoords[3*i + 2];
        }

        // -----------------------
        // 11. Extract tetrahedra (element type 4)
        // -----------------------
        std::vector<int> elemTypes;
        std::vector<std::vector<std::size_t>> elemTags;
        std::vector<std::vector<std::size_t>> elemNodeTags;
        gmsh::model::mesh::getElements(elemTypes, elemTags, elemNodeTags);

        const int tetType = 4;
        std::vector<std::size_t> tetTags;
        std::vector<std::size_t> tetNodeTags;
        for (std::size_t i = 0; i < elemTypes.size(); ++i) {
            if (elemTypes[i] == tetType) {
                tetTags = elemTags[i];
                tetNodeTags = elemNodeTags[i];
                break;
            }
        }

        const std::size_t nelems = tetTags.size();
        if (nelems == 0) throw std::runtime_error("No tetrahedral elements found.");
        if (tetNodeTags.size() != 4 * nelems) throw std::runtime_error("Unexpected tetrahedral connectivity size.");

        compactElements.resize(static_cast<int>(nelems), 4);
        for (std::size_t e = 0; e < nelems; ++e) {
            for (int k = 0; k < 4; ++k) {
                std::size_t nt = tetNodeTags[4*e + k];
                auto it = tagToCompact.find(nt);
                if (it == tagToCompact.end()) throw std::runtime_error("Node tag not found in compact map.");
                compactElements(static_cast<int>(e), k) = it->second;
            }
        }

        // -----------------------
        // 12. Compute per-element layer ID by average radius
        // -----------------------
        layerId.resize(static_cast<int>(nelems));
        const double tol = 1e-8 * heights.back();

        for (std::size_t e = 0; e < nelems; ++e) {
            double rAvg = 0.0;
            for (int k = 0; k < 4; ++k) {
                int nid = compactElements(static_cast<int>(e), k);
                double x = compactNodes(nid, 0);
                double y = compactNodes(nid, 1);
                double z = compactNodes(nid, 2);

                rAvg += z;
            }
            rAvg /= 4.0;

            int lid = 0;
            int nShells = static_cast<int>(heights.size());
            for (std::size_t i = 0; i < nShells; ++i) {
                double zIn = 0.0;
                if (i>0)
                    zIn = heights[i-1];
                double zOut = heights[i];
                bool inLayer = (rAvg >= zIn - tol) && ((i < nShells - 1 && rAvg < zOut + tol) || (i == nShells - 1 && rAvg <= zOut + tol));
                if (inLayer) { lid = static_cast<int>(i); break; }
            }
            // if (lid < 0) throw std::runtime_error("Could not assign layer ID to an element.");
            layerId(static_cast<int>(e)) = lid;
            // layerId(static_cast<int>(e)) = 0;
        }
        // -----------------------
        // 12. Finalize Gmsh
        // -----------------------
        gmsh::finalize();

        pos = compactNodes.cast<Float>();
        tet = compactElements.cast<int>();
        layer = layerId.cast<int>();

        nver = pos.rows();
        ntet = tet.rows();

        extract_boundary_and_face_cell_pairs();
        print_info();

        return 1;
    }

        int Mesh::init_from_multi_layer_box(double L, double W, const std::vector<double> &thicknesses, const std::vector<double> &meshSizes) {
        clear();
        Eigen::MatrixXd compactNodes;
        Eigen::MatrixXi compactElements;
        Eigen::VectorXi layerId;

        // -----------------------
        // 1. Validate input
        // -----------------------

        if (thicknesses.empty())
            throw std::runtime_error("thicknesses must not be empty.");
        if (meshSizes.size() != thicknesses.size())
            throw std::runtime_error("meshSizes and thicknesses must have same size.");

        // -----------------------
        // 2. Initialize Gmsh
        // -----------------------
        gmsh::initialize();
        gmsh::option::setNumber("General.Terminal", 1);
        gmsh::model::add("multi_layer_tet_fragmented");


        std::vector<gmsh::vectorpair> outDimTagsMap;
        std::vector<std::pair<int,int>> boxes;

        auto heights = thicknesses;
        std::inclusive_scan(thicknesses.begin(), thicknesses.end(), heights.begin());
        for (std::size_t i = 0; i < heights.size(); ++i)
            boxes.emplace_back(3,gmsh::model::occ::addBox(-L/2, -W/2, 0.0, L, W, heights[i]));

        gmsh::model::occ::synchronize();

        // -----------------------------
        // 2. Fragment all boxes at once
        // -----------------------------
        std::vector<std::pair<int,int>> outDimTags(boxes);
        std::vector<std::vector<std::pair<int,int>>> outMap;

        if (boxes.size()>1) {
            gmsh::model::occ::fragment(boxes, {}, outDimTags, outMap);
            gmsh::model::occ::synchronize();
        }

        // -----------------------
        // 8. Define radius-based mesh-size fields per shell
        // -----------------------

        std::vector<int> thresholdFields(meshSizes.size());
        for (std::size_t i = 0; i < meshSizes.size(); ++i) {
            double dx_i = meshSizes[i];

            int fTh = gmsh::model::mesh::field::add("Constant");
            gmsh::model::mesh::field::setNumber(fTh, "VIn", dx_i);
            gmsh::model::mesh::field::setNumbers(fTh, "VolumesList", {double(outDimTags[i].second)});
            thresholdFields[i] = fTh;
        }

        int fMin = gmsh::model::mesh::field::add("Min");
        {
            std::vector<double> list;
            list.reserve(thresholdFields.size());
            for (int id : thresholdFields) list.push_back(static_cast<double>(id));
            gmsh::model::mesh::field::setNumbers(fMin, "FieldsList", list);
        }
        gmsh::model::mesh::field::setAsBackgroundMesh(fMin);

        // -----------------------
        // 9. Generate final 3D mesh
        // -----------------------
        gmsh::model::mesh::generate(3);
        gmsh::model::mesh::optimize("Netgen");
        // -----------------------
        // 10. Extract nodes and build compact index
        // -----------------------
        std::vector<std::size_t> nodeTags;
        std::vector<double> nodeCoords, nodeParams;
        gmsh::model::mesh::getNodes(nodeTags, nodeCoords, nodeParams);

        const std::size_t nnodes = nodeTags.size();
        if (nodeCoords.size() != 3 * nnodes) {
            throw std::runtime_error("Inconsistent node coordinate size.");
        }

        compactNodes.resize(static_cast<int>(nnodes), 3);
        std::unordered_map<std::size_t, int> tagToCompact;
        tagToCompact.reserve(nnodes);
        for (std::size_t i = 0; i < nnodes; ++i) {
            tagToCompact[nodeTags[i]] = static_cast<int>(i);
            compactNodes(static_cast<int>(i), 0) = nodeCoords[3*i + 0];
            compactNodes(static_cast<int>(i), 1) = nodeCoords[3*i + 1];
            compactNodes(static_cast<int>(i), 2) = nodeCoords[3*i + 2];
        }

        // -----------------------
        // 11. Extract tetrahedra (element type 4)
        // -----------------------
        std::vector<int> elemTypes;
        std::vector<std::vector<std::size_t>> elemTags;
        std::vector<std::vector<std::size_t>> elemNodeTags;
        gmsh::model::mesh::getElements(elemTypes, elemTags, elemNodeTags);

        const int tetType = 4;
        std::vector<std::size_t> tetTags;
        std::vector<std::size_t> tetNodeTags;
        for (std::size_t i = 0; i < elemTypes.size(); ++i) {
            if (elemTypes[i] == tetType) {
                tetTags = elemTags[i];
                tetNodeTags = elemNodeTags[i];
                break;
            }
        }

        const std::size_t nelems = tetTags.size();
        if (nelems == 0) throw std::runtime_error("No tetrahedral elements found.");
        if (tetNodeTags.size() != 4 * nelems) throw std::runtime_error("Unexpected tetrahedral connectivity size.");

        compactElements.resize(static_cast<int>(nelems), 4);
        for (std::size_t e = 0; e < nelems; ++e) {
            for (int k = 0; k < 4; ++k) {
                std::size_t nt = tetNodeTags[4*e + k];
                auto it = tagToCompact.find(nt);
                if (it == tagToCompact.end()) throw std::runtime_error("Node tag not found in compact map.");
                compactElements(static_cast<int>(e), k) = it->second;
            }
        }

        // -----------------------
        // 12. Compute per-element layer ID by average radius
        // -----------------------
        layerId.resize(static_cast<int>(nelems));
        const double tol = 1e-8 * heights.back();

        for (std::size_t e = 0; e < nelems; ++e) {
            double rAvg = 0.0;
            for (int k = 0; k < 4; ++k) {
                int nid = compactElements(static_cast<int>(e), k);
                double x = compactNodes(nid, 0);
                double y = compactNodes(nid, 1);
                double z = compactNodes(nid, 2);

                rAvg += z;
            }
            rAvg /= 4.0;

            int lid = 0;
            int nShells = static_cast<int>(heights.size());
            for (std::size_t i = 0; i < nShells; ++i) {
                double zIn = 0.0;
                if (i>0)
                    zIn = heights[i-1];
                double zOut = heights[i];
                bool inLayer = (rAvg >= zIn - tol) && ((i < nShells - 1 && rAvg < zOut + tol) || (i == nShells - 1 && rAvg <= zOut + tol));
                if (inLayer) { lid = static_cast<int>(i); break; }
            }
            // if (lid < 0) throw std::runtime_error("Could not assign layer ID to an element.");
            layerId(static_cast<int>(e)) = lid;
            // layerId(static_cast<int>(e)) = 0;
        }
        // -----------------------
        // 12. Finalize Gmsh
        // -----------------------
        gmsh::finalize();

        pos = compactNodes.cast<Float>();
        tet = compactElements.cast<int>();
        layer = layerId.cast<int>();

        nver = pos.rows();
        ntet = tet.rows();

        extract_boundary_and_face_cell_pairs();
        print_info();

        return 1;
    }

    int Mesh::init_from_file(const std::string &filename) {
        clear();

        vtkSmartPointer<vtkGenericDataObjectReader> reader = vtkSmartPointer<vtkGenericDataObjectReader>::New();
        reader->SetFileName(filename.c_str());
        reader->ReadAllScalarsOn();
        reader->ReadAllVectorsOn();
        reader->ReadAllTensorsOn();
        reader->ReadAllFieldsOn();
        reader->Update();

        vtkDataObject* obj = reader->GetOutput();
        vtkDataSet* data = vtkDataSet::SafeDownCast(obj);

        if (!data)
        {
            std::cerr << "Failed to read file: " << filename << std::endl;
            return EXIT_FAILURE;
        }


        vtkIdType nPoints = data->GetNumberOfPoints();
        vtkIdType nCells = data->GetNumberOfCells();

        Eigen::VectorXi tetraIds;
        tetraIds.resize(nCells);
        int k = 0;
        for (vtkIdType i = 0; i < nCells; ++i) {
            int cellType = data->GetCellType(i);
            if (cellType==10) {
                tetraIds(k++) = i;
            }
        }
        tetraIds.conservativeResize(k);

        tet.resize(tetraIds.rows(),4);
        vtkIdList *pts=vtkIdList::New();
        k = 0;
        for (int i = 0; i < tetraIds.rows(); ++i) {
            data->GetCellPoints(tetraIds(i), pts);
            tet(i, 0) = pts->GetId(0);
            tet(i, 1) = pts->GetId(1);
            tet(i, 2) = pts->GetId(2);
            tet(i, 3) = pts->GetId(3);
        }
        
        ntet = tet.rows();
        nver = nPoints;

        pos.resize(nPoints, 3);
        for (vtkIdType i = 0; i < nPoints; ++i) {
            double p[3];
            data->GetPoint(i, p);
            pos(i, 0) = p[0];
            pos(i, 1) = p[1];
            pos(i, 2) = p[2];
        }

        int nPDataArray = data->GetPointData()->GetNumberOfArrays();
        int nCDataArray = data->GetCellData()->GetNumberOfArrays();
        int nFDataArray = data->GetFieldData()->GetNumberOfArrays();

        std::cout<< "number PointData : "<<nPDataArray<<std::endl;
        std::cout<< "number CellData : "<<nCDataArray<<std::endl;
        std::cout<< "number FieldData : "<<nFDataArray<<std::endl;

        // for (int i=0; i<data->GetPointData()->GetNumberOfArrays(); ++i) {
        //     auto array = data->GetPointData()->GetAbstractArray(i);
        //     std::cout << array->GetName() << std::endl;
        // }

        if (vtkDataArray* arr = data->GetPointData()->GetArray("X"); arr != nullptr)
            copy_from_vtk_to_eigen(arr, posRef);
        if (vtkDataArray* arr = data->GetPointData()->GetArray("vel"); arr != nullptr)
            copy_from_vtk_to_eigen(arr, vel);
        if (vtkDataArray* arr = data->GetPointData()->GetArray("actin1"); arr != nullptr)
            copy_from_vtk_to_eigen(arr, actin1);
        if (vtkDataArray* arr = data->GetPointData()->GetArray("fiber1"); arr != nullptr)
            copy_from_vtk_to_eigen(arr, fiber1);
        if (vtkDataArray* arr = data->GetPointData()->GetArray("fiber2"); arr != nullptr)
            copy_from_vtk_to_eigen(arr, fiber2);
        if (vtkDataArray* arr = data->GetPointData()->GetArray("fiber3"); arr != nullptr)
            copy_from_vtk_to_eigen(arr, fiber3);
        if (vtkDataArray* arr = data->GetPointData()->GetArray("fiber4"); arr != nullptr)
            copy_from_vtk_to_eigen(arr, fiber4);


        if (vtkDataArray* arr = data->GetPointData()->GetArray("E"); arr != nullptr) {
            copy_from_vtk_to_eigen(arr, E);
            E = E(tetraIds);
        }
        if (vtkDataArray* arr = data->GetPointData()->GetArray("nu"); arr != nullptr) {
            copy_from_vtk_to_eigen(arr, nu);
            nu = nu(tetraIds);
        }
        if (vtkDataArray* arr = data->GetPointData()->GetArray("visc"); arr != nullptr) {
            copy_from_vtk_to_eigen(arr, visc);
            visc = visc(tetraIds);
        }
        if (vtkDataArray* arr = data->GetPointData()->GetArray("plasticity"); arr != nullptr) {
            copy_from_vtk_to_eigen(arr, plasticity);
            plasticity = plasticity(tetraIds);
        }
        if (vtkDataArray* arr = data->GetPointData()->GetArray("grRate1"); arr != nullptr) {
            copy_from_vtk_to_eigen(arr, grRate1);
            grRate1 = grRate1(tetraIds);
        }
        if (vtkDataArray* arr = data->GetPointData()->GetArray("grRate2"); arr != nullptr) {
            copy_from_vtk_to_eigen(arr, grRate2);
            grRate2 = grRate2(tetraIds);
        }
        if (vtkDataArray* arr = data->GetPointData()->GetArray("grRate3"); arr != nullptr) {
            copy_from_vtk_to_eigen(arr, grRate3);
            grRate3 = grRate3(tetraIds);
        }
        if (vtkDataArray* arr = data->GetPointData()->GetArray("k1"); arr != nullptr) {
            copy_from_vtk_to_eigen(arr, k1);
            k1 = k1(tetraIds);
        }
        if (vtkDataArray* arr = data->GetPointData()->GetArray("k2"); arr != nullptr) {
            copy_from_vtk_to_eigen(arr, k2);
            k2 = k2(tetraIds);
        }
        if (vtkDataArray* arr = data->GetPointData()->GetArray("fiberTetra1"); arr != nullptr) {
            copy_from_vtk_to_eigen(arr, fiberTetra1);
            fiberTetra1 = fiberTetra1(tetraIds, Eigen::all);
        }
        if (vtkDataArray* arr = data->GetPointData()->GetArray("fiberTetra2"); arr != nullptr) {
            copy_from_vtk_to_eigen(arr, fiberTetra2);
            fiberTetra2 = fiberTetra2(tetraIds, Eigen::all);
        }
        if (vtkDataArray* arr = data->GetPointData()->GetArray("fiberTetra3"); arr != nullptr) {
            copy_from_vtk_to_eigen(arr, fiberTetra3);
            fiberTetra3 = fiberTetra3(tetraIds, Eigen::all);
        }
        if (vtkDataArray* arr = data->GetPointData()->GetArray("fiberTetra4"); arr != nullptr) {
            copy_from_vtk_to_eigen(arr, fiberTetra4);
            fiberTetra4 = fiberTetra4(tetraIds, Eigen::all);
        }
        if (vtkDataArray* arr = data->GetPointData()->GetArray("actinTetra"); arr != nullptr) {
            copy_from_vtk_to_eigen(arr, actinTetra);
            actinTetra = actinTetra(tetraIds, Eigen::all);
        }
        if (vtkDataArray* arr = data->GetCellData()->GetArray("CellEntityIds"); arr != nullptr) {
            copy_from_vtk_to_eigen(arr, layer);
            layer = layer(tetraIds);
        }
        if (vtkDataArray* arr = data->GetCellData()->GetArray("cell_scalars"); arr != nullptr) {
            copy_from_vtk_to_eigen(arr, layer);
            layer = layer(tetraIds);
        }
        if (vtkDataArray* arr = data->GetCellData()->GetArray("layer"); arr != nullptr) {
            copy_from_vtk_to_eigen(arr, layer);
            layer = layer(tetraIds);
        }
        if (vtkDataArray* arr = data->GetCellData()->GetArray("Fg"); arr != nullptr) {
            copy_from_vtk_to_eigen(arr, Fg);
            Fg = Fg(tetraIds, Eigen::all);
        }
        if (vtkDataArray* arr = data->GetCellData()->GetArray("Fp"); arr != nullptr) {
            copy_from_vtk_to_eigen(arr, Fp);
            Fp = Fp(tetraIds, Eigen::all);
        }
        if (vtkDataArray* arr = data->GetCellData()->GetArray("lamInt"); arr != nullptr) {
            copy_from_vtk_to_eigen(arr, lamInt);
            lamInt = lamInt(tetraIds, Eigen::all);
        }

        extract_boundary_and_face_cell_pairs();

        std::cout << "Mesh file " << filename << " is loaded" << std::endl;
        print_info();
        return 1;
    }

    void Mesh::save_binary(const std::string &filename,
                     const Int4ArrayHost &newTet,
                     const VectorArrayHost &newPos,
                     const std::unordered_map<std::string, std::any>& dataToSave) {
        std::ofstream file;
        file.open(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cout << "File %s writing error!" <<filename<< std::endl;
            return;
        }
        vtkSmartPointer<vtkGenericDataObjectWriter> writer = vtkSmartPointer<vtkGenericDataObjectWriter>::New();
        writer->SetFileName(filename.c_str());

        auto ugrid = vtkSmartPointer<vtkUnstructuredGrid>::New();

        auto points = vtkSmartPointer<vtkPoints>::New();
        for (int i = 0; i < newPos.size(); i++)
            points->InsertNextPoint(newPos[i][0], newPos[i][1], newPos[i][2]); // Point 0
        ugrid->SetPoints(points);

        // Add a vertex (just one point)
        auto tetra = vtkSmartPointer<vtkTetra>::New();
        for (int i = 0; i < newTet.size(); i++) {
            tetra->GetPointIds()->SetId(0, newTet[i].x);
            tetra->GetPointIds()->SetId(1, newTet[i].y);
            tetra->GetPointIds()->SetId(2, newTet[i].z);
            tetra->GetPointIds()->SetId(3, newTet[i].w);
            ugrid->InsertNextCell(tetra->GetCellType(), tetra->GetPointIds());
        }

        auto vtkArray = vtkSmartPointer<vtkFloatArray>::New();
        vtkArray->SetName("time");
        vtkArray->SetNumberOfComponents(1);
        vtkArray->SetNumberOfTuples(1);
        float ftime = globalTime;
        vtkArray->SetTuple(0, &ftime);
        ugrid->GetFieldData()->AddArray(vtkArray);

        for (auto data = dataToSave.begin(); data!=dataToSave.end(); data++) {
            auto vtkArray = vtkSmartPointer<vtkFloatArray>::New();
            vtkArray->SetName(data->first.c_str());
            if (data->second.type() == typeid(IntArrayDev*)) {
                auto cData = std::any_cast<IntArrayDev*>(data->second);
                copy_from_device_array_to_vtk_array(*cData, vtkArray);
            } else if (data->second.type() == typeid(ScalarArrayDev*)) {
                auto cData = std::any_cast<ScalarArrayDev*>(data->second);
                copy_from_device_array_to_vtk_array(*cData, vtkArray);
            } else if (data->second.type() == typeid(VectorArrayDev*)) {
                auto cData = std::any_cast<VectorArrayDev*>(data->second);
                copy_from_device_array_to_vtk_array(*cData, vtkArray);
            } else if (data->second.type() == typeid(TensorArrayDev*)) {
                auto cData = std::any_cast<TensorArrayDev*>(data->second);
                copy_from_device_array_to_vtk_array(*cData, vtkArray);
            }
            if (vtkArray->GetNumberOfTuples()==nver)
                ugrid->GetPointData()->AddArray(vtkArray);
            else if (vtkArray->GetNumberOfTuples()==ntet)
                ugrid->GetCellData()->AddArray(vtkArray);
        }

        writer->SetInputData(ugrid);
        writer->SetFileTypeToBinary();
        writer->Write();
    }

    void Mesh::save_surface(const std::string &filename, const thrust::host_vector<Vector> &new_pos) {
        std::ofstream file;
//            file.open(filename, std::ios::out | std::ios::binary);
        file.open(filename);
        if (!file.is_open()) {
            std::cout << "File %s writing error!" <<filename<< std::endl;
            return;
        }
        file << "# vtk DataFile Version 2.0" << std::endl;
        file << "box, Created by LagSol" << std::endl;
        file << "ASCII" << std::endl;
        file << "DATASET UNSTRUCTURED_GRID" << std::endl;
        file << "POINTS " << nver << " float" << std::endl;
        for (int i = 0; i < nver; i++)
            file << (float)new_pos[i][0] << " " << (float)new_pos[i][1] << " " << (float)new_pos[i][2] << std::endl;
        file << "CELLS " << ntri << " " << ntri * 4 << std::endl;
        for (int i = 0; i < ntri; i++)
            file << "3 " << tri(i, 0) << " " << tri(i, 1) << " " << tri(i, 2) << std::endl;
        file << "CELL_TYPES " << ntri << std::endl;
        for (int i = 0; i < ntri; i++)
            file << "5" << std::endl;

        file << "CELL_DATA " << ntri << std::endl;
        file << "SCALARS layer int 1" << std::endl;
        file << "LOOKUP_TABLE default" << std::endl;
        for (int i = 0; i < ntri; i++) {
            file << layer(boundaryTetIds[i]) << std::endl;
        }
//            file << "TENSORS stress float" << std::endl;
//            for (int i = 0; i < ntet; i++) {
//                file << (float)stress[i].xx << " " << (float)stress[i].xy << " " << (float)stress[i].xz << std::endl <<
//                        (float)stress[i].yx << " " << (float)stress[i].yy << " " << (float)stress[i].yz << std::endl <<
//                        (float)stress[i].zx << " " << (float)stress[i].zy << " " << (float)stress[i].zz << std::endl << std::endl;
//            }
        file.close();
    }
/*
    void Mesh::save_surface(const std::string &filename, const VectorArrayEigen &new_pos, const VectorArrayEigen &vel) {
        std::ofstream file;
//            file.open(filename, std::ios::out | std::ios::binary);
        file.open(filename);
        if (!file.is_open()) {
            std::cout << "File %s writing error!" <<filename<< std::endl;
            return;
        }
        file << "# vtk DataFile Version 2.0" << std::endl;
        file << "box, Created by LagSol" << std::endl;
        file << "ASCII" << std::endl;
        file << "DATASET UNSTRUCTURED_GRID" << std::endl;
        file << "POINTS " << nnbd << " float" << std::endl;
        for (int i = 0; i < nnbd; i++)
            file << (float)new_pos[i].x << " " << (float)new_pos[i].y << " " << (float)new_pos[i].z << std::endl;
        file << "CELLS " << ntri << " " << ntri * 4 << std::endl;
        for (int i = 0; i < ntri; i++)
            file << "3 " << tri(i, 0) << " " << tri(i, 1) << " " << tri(i, 2) << std::endl;
        file << "CELL_TYPES " << ntri << std::endl;
        for (int i = 0; i < ntri; i++)
            file << "5" << std::endl;

        file << "CELL_DATA " << ntri << std::endl;
        file << "SCALARS layer int 1" << std::endl;
        file << "LOOKUP_TABLE default" << std::endl;
        for (int i = 0; i < ntri; i++) {
            file << tetTag(boundaryTetIds[i]) << std::endl;
        }
//            file << "TENSORS stress float" << std::endl;
//            for (int i = 0; i < ntet; i++) {
//                file << (float)stress[i].xx << " " << (float)stress[i].xy << " " << (float)stress[i].xz << std::endl <<
//                        (float)stress[i].yx << " " << (float)stress[i].yy << " " << (float)stress[i].yz << std::endl <<
//                        (float)stress[i].zx << " " << (float)stress[i].zy << " " << (float)stress[i].zz << std::endl << std::endl;
//            }

        file << "POINT_DATA " << ntri << std::endl;
        file << "VECTORS velocity float" << std::endl;
        file << "LOOKUP_TABLE default" << std::endl;
        for (int i = 0; i < nnbd; i++) {
            file << tetTag(boundaryTetIds[i]) << std::endl;
        }
        file.close();
    }
*/

/*
    bool writeOBJ(const std::string str,
                  const Eigen::MatrixXd& V,
                  const Eigen::MatrixXd& C,
                  const Eigen::MatrixXi& F) {
        using namespace std;
        using namespace Eigen;
        assert(V.cols() == 3 && "V should have 3 columns");
        assert(C.cols() == 3 && "C should have 3 columns");
        assert(C.rows() == V.rows() && "C and V should have the same number of rows");
        ofstream s(str);
        if(!s.is_open())
        {
            fprintf(stderr,"IOError: writeOBJ() could not open %s\n",str.c_str());
            return false;
        }

        MatrixX<double> VC(V.rows(),6);

        VC.col(0) = V.col(0);
        VC.col(1) = V.col(1);
        VC.col(2) = V.col(2);
        VC.col(3) = C.col(0);
        VC.col(4) = C.col(1);
        VC.col(5) = C.col(2);
        s<<
         VC.format(IOFormat(StreamPrecision,DontAlignCols," ","\n","v ","","","\n"))<<
         (F.array()+1).format(IOFormat(FullPrecision,DontAlignCols," ","\n","f ","","","\n"));
        return true;
    }
    */
}
