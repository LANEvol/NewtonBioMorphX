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
        std::vector<double> tuple(nComp);
        for (int i = 0; i < nTuples; ++i) {
            arr->GetTuple(i, tuple.data());
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
        constexpr int nComp = sizeof(T)/sizeof(Float);
        vtkArray->SetNumberOfComponents(nComp);
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
        constexpr int nComp = 1;
        vtkArray->SetNumberOfComponents(nComp);
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
        constexpr int nComp = 1;
        vtkArray->SetNumberOfComponents(nComp);
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
        // Local face -> local vertex table. Same convention as the original code:
        // f0 = abc, f1 = acd, f2 = abd, f3 = bcd. The opposite vertex is 6 - sum.
        static constexpr int LF[4][3] = {{0,1,2},{0,2,3},{0,1,3},{1,2,3}};

        // ------------------------------------------------------------------
        // 1. Layers: remap arbitrary tags to 1..nlay
        // ------------------------------------------------------------------
        if (layer.size() == 0) {
            layer.setZero(ntet);
        } else if (layer.size() != ntet) {
            throw std::runtime_error("extract_boundary: layer has " + std::to_string(layer.size()) +
                                     " entries but mesh has " + std::to_string(ntet) + " tets");
        }

        std::vector<int> uniqueTags(layer.data(), layer.data() + ntet);
        std::sort(uniqueTags.begin(), uniqueTags.end());
        uniqueTags.erase(std::unique(uniqueTags.begin(), uniqueTags.end()), uniqueTags.end());
        nlay = static_cast<int>(uniqueTags.size());
        if (nlay > MAX_NLAYERS)
            throw std::runtime_error("extract_boundary: " + std::to_string(nlay) +
                                     " layers exceeds MAX_NLAYERS = " + std::to_string(MAX_NLAYERS));
        for (int i = 0; i < ntet; ++i)
            layer(i) = static_cast<int>(std::lower_bound(uniqueTags.begin(), uniqueTags.end(), layer(i))
                                        - uniqueTags.begin()) + 1;

        // ------------------------------------------------------------------
        // 2. Enumerate all 4*ntet tet faces with a sorted vertex key
        // ------------------------------------------------------------------
        struct FaceRec {
            std::array<int,3> key;  // sorted vertex ids
            int tet;
            int local;              // local face index in 'tet'
        };
        std::vector<FaceRec> recs;
        recs.reserve(4 * static_cast<std::size_t>(ntet));
        for (int t = 0; t < ntet; ++t) {
            for (int f = 0; f < 4; ++f) {
                std::array<int,3> k = { tet(t, LF[f][0]), tet(t, LF[f][1]), tet(t, LF[f][2]) };
                std::sort(k.begin(), k.end());
                recs.push_back({k, t, f});
            }
        }
        // Sort by key, then by tet: the lowest tet of each face comes first,
        // which reproduces the original fac(i,0)/fac(i,1) semantics and ordering.
        std::sort(recs.begin(), recs.end(), [](const FaceRec &a, const FaceRec &b) {
            return a.key != b.key ? a.key < b.key : a.tet < b.tet;
        });

        // ------------------------------------------------------------------
        // 3. Pair faces: 1 owner -> boundary, 2 owners -> interior
        // ------------------------------------------------------------------
        std::vector<std::array<int,3>> facList;   // (local face of cell0, cell0, cell1 or -1)
        std::vector<std::size_t> bdryRec;         // index into recs for each boundary face
        facList.reserve(recs.size() / 2 + recs.size() / 8);
        int nNonManifold = 0, nDegenerate = 0;

        for (std::size_t i = 0; i < recs.size();) {
            std::size_t j = i + 1;
            while (j < recs.size() && recs[j].key == recs[i].key) ++j;
            const std::size_t n = j - i;

            if (n == 1) {
                facList.push_back({recs[i].local, recs[i].tet, -1});
                bdryRec.push_back(i);
            } else {
                if (recs[i].tet == recs[i + 1].tet) ++nDegenerate;  // repeated vertex in a tet
                if (n > 2) ++nNonManifold;                          // >2 tets share a face
                facList.push_back({recs[i].local, recs[i].tet, recs[i + 1].tet});
            }
            i = j;
        }
        if (nNonManifold > 0)
            std::cerr << "extract_boundary: warning, " << nNonManifold
                      << " faces shared by more than two tets (non-manifold mesh)\n";
        if (nDegenerate > 0)
            std::cerr << "extract_boundary: warning, " << nDegenerate
                      << " faces belong to degenerate tets (repeated vertices)\n";

        // ------------------------------------------------------------------
        // 4. Boundary triangles, owning tet, outward orientation
        // ------------------------------------------------------------------
        ntri = static_cast<int>(bdryRec.size());
        tri.resize(ntri, 3);
        boundaryTetIds.resize(ntri);
        for (int k = 0; k < ntri; ++k) {
            const FaceRec &r = recs[bdryRec[k]];
            const int t = r.tet, f = r.local;
            int a = tet(t, LF[f][0]);
            int b = tet(t, LF[f][1]);
            int c = tet(t, LF[f][2]);
            const int d = tet(t, 6 - LF[f][0] - LF[f][1] - LF[f][2]);  // opposite vertex

            const Eigen::Matrix<Float,3,1> pa = pos.row(a).matrix().transpose();
            const Eigen::Matrix<Float,3,1> pb = pos.row(b).matrix().transpose();
            const Eigen::Matrix<Float,3,1> pc = pos.row(c).matrix().transpose();
            const Eigen::Matrix<Float,3,1> pd = pos.row(d).matrix().transpose();
            if ((pb - pa).cross(pc - pa).dot(pd - pa) > Float(0))
                std::swap(b, c);  // make the normal point away from the owning tet

            tri(k, 0) = a; tri(k, 1) = b; tri(k, 2) = c;
            boundaryTetIds(k) = t;
        }

        // ------------------------------------------------------------------
        // 5. Boundary nodes (sorted) and local re-indexing
        // ------------------------------------------------------------------
        std::vector<char> isBdry(nver, 0);
        for (int k = 0; k < ntri; ++k)
            for (int m = 0; m < 3; ++m) isBdry[tri(k, m)] = 1;

        std::vector<int> globalToBdry(nver, -1);
        boundaryNodeIds.clear();
        for (int v = 0; v < nver; ++v) {
            if (isBdry[v]) {
                globalToBdry[v] = static_cast<int>(boundaryNodeIds.size());
                boundaryNodeIds.push_back(v);
            }
        }
        nnbd = static_cast<int>(boundaryNodeIds.size());

        tri_mapped.resize(ntri, 3);
        for (int k = 0; k < ntri; ++k)
            for (int m = 0; m < 3; ++m) tri_mapped(k, m) = globalToBdry[tri(k, m)];

        boundaryPos = pos(boundaryNodeIds, Eigen::placeholders::all);

        // ------------------------------------------------------------------
        // 6. Boundary triangles grouped by layer (cleared first: safe to call twice)
        // ------------------------------------------------------------------
        for (int l = 0; l < MAX_NLAYERS; ++l)
            boundaryTetIdsLayers[l].clear();
        for (int k = 0; k < ntri; ++k)
            boundaryTetIdsLayers[layer(boundaryTetIds(k)) - 1].push_back(k);

        // ------------------------------------------------------------------
        // 7. Face-cell pairs
        // ------------------------------------------------------------------
        nfac = static_cast<int>(facList.size());
        fac.resize(nfac, 3);
        for (int i = 0; i < nfac; ++i) {
            fac(i, 0) = facList[i][0];
            fac(i, 1) = facList[i][1];
            fac(i, 2) = facList[i][2];
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
        //gmsh::option::setNumber("Mesh.Algorithm3D", 10);   // HXT
        gmsh::option::setNumber("General.NumThreads", 0);  // use all cores
        gmsh::model::mesh::generate(3);


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
        //gmsh::option::setNumber("Mesh.Algorithm3D", 10);   // HXT
        gmsh::option::setNumber("General.NumThreads", 0);  // use all cores
        gmsh::model::mesh::generate(3);


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
        //gmsh::option::setNumber("Mesh.Algorithm3D", 10);   // HXT
        gmsh::option::setNumber("General.NumThreads", 0);  // use all cores
        gmsh::option::setNumber("Mesh.OptimizeNetgen", 0);
        gmsh::model::mesh::generate(3);


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
        //gmsh::option::setNumber("Mesh.Algorithm3D", 10);   // HXT
        gmsh::option::setNumber("General.NumThreads", 0);  // use all cores
        gmsh::model::mesh::generate(3);


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
        //gmsh::option::setNumber("Mesh.Algorithm3D", 10);   // HXT
        gmsh::option::setNumber("General.NumThreads", 0);  // use all cores
        gmsh::model::mesh::generate(3);

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
        //gmsh::option::setNumber("Mesh.Algorithm3D", 10);   // HXT
        gmsh::option::setNumber("General.NumThreads", 0);  // use all cores
        gmsh::model::mesh::generate(3);
        // gmsh::option::setNumber("Mesh.OptimizeNetgen", 0);
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

        std::cout<< "# PointData: "<<nPDataArray<<std::endl;
        for (int i=0; i<data->GetPointData()->GetNumberOfArrays(); ++i) {
            auto array = data->GetPointData()->GetAbstractArray(i);
            std::cout <<"\t"<<array->GetName()<< std::endl;
        }
        std::cout<< "# CellData: "<<nCDataArray<<std::endl;
        for (int i=0; i<data->GetCellData()->GetNumberOfArrays(); ++i) {
            auto array = data->GetCellData()->GetAbstractArray(i);
            std::cout <<"\t"<<array->GetName()<< std::endl;
        }
        std::cout<< "# FieldData: "<<nFDataArray<<std::endl;
        for (int i=0; i<data->GetFieldData()->GetNumberOfArrays(); ++i) {
            auto array = data->GetFieldData()->GetAbstractArray(i);
            std::cout <<"\t"<<array->GetName()<< std::endl;
        }

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
            E = E(tetraIds).eval();
        }
        if (vtkDataArray* arr = data->GetPointData()->GetArray("nu"); arr != nullptr) {
            copy_from_vtk_to_eigen(arr, nu);
            nu = nu(tetraIds).eval();
        }
        if (vtkDataArray* arr = data->GetPointData()->GetArray("visc"); arr != nullptr) {
            copy_from_vtk_to_eigen(arr, visc);
            visc = visc(tetraIds).eval();
        }
        if (vtkDataArray* arr = data->GetPointData()->GetArray("plasticity"); arr != nullptr) {
            copy_from_vtk_to_eigen(arr, plasticity);
            plasticity = plasticity(tetraIds).eval();
        }
        if (vtkDataArray* arr = data->GetPointData()->GetArray("grRate1"); arr != nullptr) {
            copy_from_vtk_to_eigen(arr, grRate1);
            grRate1 = grRate1(tetraIds).eval();
        }
        if (vtkDataArray* arr = data->GetPointData()->GetArray("grRate2"); arr != nullptr) {
            copy_from_vtk_to_eigen(arr, grRate2);
            grRate2 = grRate2(tetraIds).eval();
        }
        if (vtkDataArray* arr = data->GetPointData()->GetArray("grRate3"); arr != nullptr) {
            copy_from_vtk_to_eigen(arr, grRate3);
            grRate3 = grRate3(tetraIds).eval();
        }
        if (vtkDataArray* arr = data->GetPointData()->GetArray("k1"); arr != nullptr) {
            copy_from_vtk_to_eigen(arr, k1);
            k1 = k1(tetraIds).eval();
        }
        if (vtkDataArray* arr = data->GetPointData()->GetArray("k2"); arr != nullptr) {
            copy_from_vtk_to_eigen(arr, k2);
            k2 = k2(tetraIds).eval();
        }
        if (vtkDataArray* arr = data->GetPointData()->GetArray("fiberTetra1"); arr != nullptr) {
            copy_from_vtk_to_eigen(arr, fiberTetra1);
            fiberTetra1 = fiberTetra1(tetraIds, Eigen::placeholders::all);
        }
        if (vtkDataArray* arr = data->GetPointData()->GetArray("fiberTetra2"); arr != nullptr) {
            copy_from_vtk_to_eigen(arr, fiberTetra2);
            fiberTetra2 = fiberTetra2(tetraIds, Eigen::placeholders::all);
        }
        if (vtkDataArray* arr = data->GetPointData()->GetArray("fiberTetra3"); arr != nullptr) {
            copy_from_vtk_to_eigen(arr, fiberTetra3);
            fiberTetra3 = fiberTetra3(tetraIds, Eigen::placeholders::all);
        }
        if (vtkDataArray* arr = data->GetPointData()->GetArray("fiberTetra4"); arr != nullptr) {
            copy_from_vtk_to_eigen(arr, fiberTetra4);
            fiberTetra4 = fiberTetra4(tetraIds, Eigen::placeholders::all);
        }
        if (vtkDataArray* arr = data->GetPointData()->GetArray("actinTetra"); arr != nullptr) {
            copy_from_vtk_to_eigen(arr, actinTetra);
            actinTetra = actinTetra(tetraIds, Eigen::placeholders::all);
        }
        if (vtkDataArray* arr = data->GetCellData()->GetArray("CellEntityIds"); arr != nullptr) {
            copy_from_vtk_to_eigen(arr, layer);
            layer = layer(tetraIds).eval();
        }
        if (vtkDataArray* arr = data->GetCellData()->GetArray("cell_scalars"); arr != nullptr) {
            copy_from_vtk_to_eigen(arr, layer);
            layer = layer(tetraIds).eval();
        }
        if (vtkDataArray* arr = data->GetCellData()->GetArray("layer"); arr != nullptr) {
            copy_from_vtk_to_eigen(arr, layer);
            layer = layer(tetraIds).eval();
        }
        if (vtkDataArray* arr = data->GetCellData()->GetArray("Fg"); arr != nullptr) {
            copy_from_vtk_to_eigen(arr, Fg);
            Fg = Fg(tetraIds, Eigen::placeholders::all);
        }
        if (vtkDataArray* arr = data->GetCellData()->GetArray("Fp"); arr != nullptr) {
            copy_from_vtk_to_eigen(arr, Fp);
            Fp = Fp(tetraIds, Eigen::placeholders::all);
        }
        if (vtkDataArray* arr = data->GetCellData()->GetArray("lamInt"); arr != nullptr) {
            copy_from_vtk_to_eigen(arr, lamInt);
            lamInt = lamInt(tetraIds, Eigen::placeholders::all);
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
            std::cerr << "Cannot write " << filename << std::endl;
            return;
        }
        file.close();   // only a writability check; VTK opens the file itself
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
}
