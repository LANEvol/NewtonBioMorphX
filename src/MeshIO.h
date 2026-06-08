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


#ifndef LAGRANGIANSOLID_MESH_IO_H
#define LAGRANGIANSOLID_MESH_IO_H

#include <string>
#include <fstream>
#include <Eigen/Dense>
#include "Typedefs.h"
#include <vector>
#include <map>
#include <any>
#include <unordered_map>
#include <algorithm>
#include "DataStructure.h"
#include "thrust/host_vector.h"

#include <Eigen/Core>

namespace LagSol {

    // Build vertex -> incident tets adjacency
    // T: #T x 4
    // nV: number of vertices
    std::vector<std::vector<int>> build_vertex_to_tets_adjacency(
        const Eigen::MatrixXi& T,
        int nV);

    // Intersect two sorted vectors of ints into result (also sorted)
    inline void intersect_two_sorted(
        const std::vector<int>& a,
        const std::vector<int>& b,
        std::vector<int>& result);

    // Given T (tets), F (boundary faces), and number of vertices,
    // return FT: tetra index for each boundary face (F.row(i)).
    Eigen::VectorXi tetra_for_boundary_faces_v2t(
        const Eigen::MatrixXi& T,
        const Eigen::MatrixXi& F,
        int nV);

    bool isNumber(const std::string& s);
    template <typename DerivedX,typename DerivedI>
    inline void find(const Eigen::DenseBase<DerivedX>& X, Eigen::PlainObjectBase<DerivedI> & I) {
        const int nnz = X.count();
        I.resize(nnz,1);
        int k = 0;
        for(int j = 0;j<X.cols();j++) {
            for(int i = 0;i<X.rows();i++) {
                if(X(i,j)) {
                    I(k) = i+X.rows()*j;
                    k++;
                }
            }
        }
    }

    struct Mesh {
        int nver;
        int ntet;
        int ntri;
        int nfac;
        int nnbd;
        int nlay;
        Eigen::Array<Float, Eigen::Dynamic, 3, Eigen::RowMajor> pos;
        Eigen::Array<Float, Eigen::Dynamic, 3, Eigen::RowMajor> posRef; // reference coordinate
        Eigen::Array<Float, Eigen::Dynamic, 3, Eigen::RowMajor> vel; // velocity

        Eigen::Array<int, Eigen::Dynamic, 4, Eigen::RowMajor> tet;
        Eigen::Array<int, Eigen::Dynamic, 3, Eigen::RowMajor> tri;
        Eigen::Array<int, Eigen::Dynamic, 3, Eigen::RowMajor> fac;

        Eigen::Array<int, Eigen::Dynamic, 1> layer;
        Eigen::Array<int, Eigen::Dynamic, 1> bdryFun;


        Eigen::Array<Float, Eigen::Dynamic, 1> E;
        Eigen::Array<Float, Eigen::Dynamic, 1> nu;
        Eigen::Array<Float, Eigen::Dynamic, 1> visc;
        Eigen::Array<Float, Eigen::Dynamic, 1> plasticity;
        Eigen::Array<Float, Eigen::Dynamic, 1> grRate1;
        Eigen::Array<Float, Eigen::Dynamic, 1> grRate2;
        Eigen::Array<Float, Eigen::Dynamic, 1> grRate3;
        Eigen::Array<Float, Eigen::Dynamic, 1> k1;
        Eigen::Array<Float, Eigen::Dynamic, 1> k2;
        Eigen::Array<Float, Eigen::Dynamic, 9, Eigen::RowMajor> fiberTetra1;
        Eigen::Array<Float, Eigen::Dynamic, 9, Eigen::RowMajor> fiberTetra2;
        Eigen::Array<Float, Eigen::Dynamic, 9, Eigen::RowMajor> fiberTetra3;
        Eigen::Array<Float, Eigen::Dynamic, 9, Eigen::RowMajor> fiberTetra4;
        Eigen::Array<Float, Eigen::Dynamic, 9, Eigen::RowMajor> actinTetra;
        
        Eigen::Array<Float, Eigen::Dynamic, 1> scalarFun;
        Eigen::Array<Float, Eigen::Dynamic, 1> scalarActivityFun;

        Eigen::Array<Float, Eigen::Dynamic, 9, Eigen::RowMajor> fiber1;
        Eigen::Array<Float, Eigen::Dynamic, 9, Eigen::RowMajor> fiber2;
        Eigen::Array<Float, Eigen::Dynamic, 9, Eigen::RowMajor> fiber3;
        Eigen::Array<Float, Eigen::Dynamic, 9, Eigen::RowMajor> fiber4;

        Eigen::Array<Float, Eigen::Dynamic, 9, Eigen::RowMajor> Fp;
        Eigen::Array<Float, Eigen::Dynamic, 9, Eigen::RowMajor> Fg;
        Eigen::Array<Float, Eigen::Dynamic, 3, Eigen::RowMajor> lamInt;

        Eigen::Array<Float, Eigen::Dynamic, 9, Eigen::RowMajor> actin1;
        Eigen::Array<Float, Eigen::Dynamic, 9, Eigen::RowMajor> actin2;
        Eigen::Array<Float, Eigen::Dynamic, 9, Eigen::RowMajor> actin1Prod;
        Eigen::Array<Float, Eigen::Dynamic, 9, Eigen::RowMajor> actin2Prod;
        Eigen::Array<Float, Eigen::Dynamic, 1> actin1Diff;
        Eigen::Array<Float, Eigen::Dynamic, 1> actin2Diff;
        Eigen::Array<Float, Eigen::Dynamic, 1> actin1Deca;
        Eigen::Array<Float, Eigen::Dynamic, 1> actin2Deca;


        Eigen::Array<int, Eigen::Dynamic, 3, Eigen::RowMajor> tri_mapped;
        // boundary... variables are used in contact detection algorithm
        Eigen::Array<Float, Eigen::Dynamic, 3, Eigen::RowMajor> boundaryPos;

        Eigen::VectorXi boundaryTetIds;
        std::vector<int> boundaryTetIdsLayers[MAX_NLAYERS];
//        std::vector<int> boundaryTetIdsTop;
//        std::vector<int> boundaryTetIdsBot;
        std::vector<int> boundaryNodeIds;


        Mesh();
        void clear();
        int init_from_file(const std::string &filename);
        int init_from_multi_layer_box(double L, double W, const std::vector<double> &thicknesses, const std::vector<double> &meshSizes);
        int init_from_multi_layer_disk(double R, const std::vector<double> &thicknesses, const std::vector<double> &meshSizes);
        int init_from_multi_layer_cone(const double &L, const double &apexAng, const std::vector<double> &radii, const std::vector<double> &meshSizes);
        int init_from_multi_layer_tube(const double &L, const std::vector<double> &radii, const std::vector<double> &meshSizes);
        int init_from_multi_layer_sphere(const std::vector<double> &radii, const std::vector<double> &meshSizes);
        int init_from_multi_layer_torus(const double &R, const std::vector<double> &radii, const std::vector<double> &meshSizes);
        void extract_boundary_and_face_cell_pairs();
        void print_info();
        // int init(const std::string &filename);
        void save(const std::string &filename,
                 const thrust::host_vector<int4> &newTet,
                 const thrust::host_vector<int> &newTag,
                 const thrust::host_vector<Vector> &newPos,
                 const thrust::host_vector<Vector> &vel,
                 const thrust::host_vector<Float> &vonMises,
                 const thrust::host_vector<Tensor> &stress);

        void save_binary(const std::string &filename,
                     const Int4ArrayHost &newTet,
                     const VectorArrayHost &newPos,
                     const std::unordered_map<std::string, std::any> &dataToSave);

        void save_surface(const std::string &filename, const thrust::host_vector<Vector> &new_pos);
        // void save_surface(const std::string &filename, const thrust::host_vector<Vector> &new_pos, const thrust::host_vector<Vector> &vel);

    };
/*
    bool writeOBJ(const std::string str,
                  const Eigen::MatrixXd& V,
                  const Eigen::MatrixXd& C,
                  const Eigen::MatrixXi& F);
                  */
    template <typename DerivedV, typename DerivedC, typename DerivedF>
    inline bool writeOBJ(
            const std::string str,
            const Eigen::MatrixBase<DerivedV>& V,
            const Eigen::MatrixBase<DerivedC>& C,
            const Eigen::MatrixBase<DerivedF>& F)
    {
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


    template <typename DerivedV, typename DerivedF>
    inline bool writePLY(
            const std::string& str,
            const Eigen::MatrixBase<DerivedV>& V,
            const Eigen::MatrixBase<DerivedF>& F,
            const std::string& comment="")
    {
        using namespace std;
        using namespace Eigen;
        assert(V.cols() == 3 && "V should have 3 columns");
        ofstream s(str);
        if(!s.is_open())
        {
            fprintf(stderr,"IOError: writePLY() could not open %s\n",str.c_str());
            return false;
        }

        s<<"ply\nformat ascii 1.0\n";
        if (!std::empty(comment))
            s<<"comment "<<comment<<"\n";
        s<<"element vertex "<<V.rows()<<"\n";
        s<<"property float x\n";
        s<<"property float y\n";
        s<<"property float z\n";
        s<<"element face "<<F.rows()<<"\n";
        s<<"property list uchar int vertex_indices\n";
        s<<"end_header\n";

        s<<
         V.format(IOFormat(StreamPrecision,DontAlignCols," ","\n","","","","\n"))<<
         F.array().format(IOFormat(FullPrecision,DontAlignCols," ","\n","3 ","","","\n"));

        return true;
    }


    template <typename DerivedV, typename DerivedC, typename DerivedF>
    inline bool writePLY(
            const std::string& str,
            const Eigen::MatrixBase<DerivedV>& V,
            const Eigen::MatrixBase<DerivedC>& C,
            const Eigen::MatrixBase<DerivedF>& F,
            const std::string& comment="")
    {
        using namespace std;
        using namespace Eigen;
        assert(V.cols() == 3 && "V should have 3 columns");
        assert(C.cols() == 3 && "C should have 3 columns");
        assert(C.rows() == V.rows() && "C and V should have the same number of rows");
        ofstream s(str);
        if(!s.is_open())
        {
            fprintf(stderr,"IOError: writePLY() could not open %s\n",str.c_str());
            return false;
        }

        MatrixX<double> VC(V.rows(),6);

        VC.col(0) = V.col(0);
        VC.col(1) = V.col(1);
        VC.col(2) = V.col(2);
        VC.col(3) = (C.col(0)*255.0).array().floor();
        VC.col(4) = (C.col(1)*255.0).array().floor();
        VC.col(5) = (C.col(2)*255.0).array().floor();

        s<<"ply\nformat ascii 1.0\n";
        if (!std::empty(comment))
            s<<"comment "<<comment<<"\n";
        s<<"element vertex "<<V.rows()<<"\n";
        s<<"property float x\n";
        s<<"property float y\n";
        s<<"property float z\n";
        s<<"property uchar red\n";
        s<<"property uchar green\n";
        s<<"property uchar blue\n";
        s<<"element face "<<F.rows()<<"\n";

//    comment cube.obj created by IVREAD.
//    comment Original data in cube.iv
//    comment converted from OBJ by obj2ply
        s<<"property list uchar int vertex_indices\n";
        s<<"end_header\n";

        s<<
//         V.format(IOFormat(StreamPrecision,DontAlignCols," ","\n","","","","\n"))<<
         VC.format(IOFormat(StreamPrecision,DontAlignCols," ","\n","","","","\n"))<<
         F.array().format(IOFormat(FullPrecision,DontAlignCols," ","\n","3 ","","","\n"));

        return true;
    }

}

#endif //LAGRANGIANSOLID_MESH_IO_H
