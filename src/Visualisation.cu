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

#include "Visualisation.h"
#include "Kernels.h"
#include "MyGUI.h"
#include "igl/signed_distance.h"
#include <igl/histc.h>
#include <igl/opengl/gl.h>

extern "C" {
    #include <libavutil/opt.h>
}


namespace LagSol {
    Eigen::ArrayX3d getSphereTheta(int N, const Float& R,const Float& phi0,const Float& theta0, const Float& dTheta) {
        Eigen::ArrayX3d res(N,3);
        Eigen::RowVector3d center(R * std::sin(phi0) * std::cos(theta0), R * std::sin(phi0) * std::sin(theta0),R * std::cos(phi0));
        Eigen::ArrayXd theta = Eigen::ArrayXd::LinSpaced(N, theta0, theta0 + dTheta);
        res.col(0) = R * theta.cos() * std::sin(phi0);
        res.col(1) = R * theta.sin() * std::sin(phi0);
        res.col(2) = center.z();
        return res;
    }

    Eigen::ArrayX3d getSpherePhi(int N, const Float& R,const Float& theta0,const Float& phi0, const Float& dPhi) {
        Eigen::ArrayX3d res(N,3);
        Eigen::RowVector3d center(R * std::sin(phi0) * std::cos(theta0), R * std::sin(phi0) * std::sin(theta0), R * std::cos(phi0));
        Eigen::ArrayXd phi = Eigen::ArrayXd::LinSpaced(N, phi0, phi0 + dPhi);
        res.col(0) = R*std::cos(theta0)*phi.sin();
        res.col(1) = R*std::sin(theta0)*phi.sin();
        res.col(2) = R*phi.cos();
        return res;
    }

    Eigen::ArrayX3d getTorusTheta(int N, const Float& R, const Float& r,const Float& phi0,const Float& theta0, const Float& dTheta) {
        Eigen::ArrayX3d res(N,3);
        Eigen::ArrayXd theta = Eigen::ArrayXd::LinSpaced(N, theta0, theta0 + dTheta);
        res.col(0) = (R - r * theta.cos()) * std::cos(phi0);
        res.col(1) = (R - r * theta.cos()) * std::sin(phi0);
        res.col(2) = r * theta.sin();
        return res;
    }

    Eigen::ArrayX3d getTorusPhi(int N, const Float& R, const Float& r,const Float& theta0,const Float& phi0, const Float& dPhi) {
        Eigen::ArrayX3d res(N,3);
        Eigen::ArrayXd phi = Eigen::ArrayXd::LinSpaced(N, phi0, phi0 + dPhi);
        res.col(0) = (R - r * std::cos(theta0)) * phi.cos();
        res.col(1) = (R - r * std::cos(theta0)) * phi.sin();
        res.col(2) = r * std::sin(theta0);
        return res;
    }

    /*
    VectorArrayEigen getNormals(const VectorArrayEigen& centers, const VectorArrayEigen& pos, const Eigen::Array<int, Eigen::Dynamic, 3, Eigen::RowMajor>& tri) {
        Eigen::MatrixXd tempCenters = centers.cast<double>().matrix();
        Eigen::MatrixXd distances;
        Eigen::MatrixXi indices;
        Eigen::MatrixX3d tempPos;
        Eigen::MatrixX3d tempNormal;
        igl::AABB<Eigen::MatrixXd,3> tree;
        igl::FastWindingNumberBVH fwn_bvh;
        Eigen::MatrixXd V = pos.cast<double>().matrix();
        Eigen::MatrixXi F = tri.matrix();
        Eigen::MatrixXd FN,VN,EN;
        Eigen::MatrixXi E;
        Eigen::VectorXi EMAP;

        tree.init(V,F);
        igl::per_face_normals(V,F,FN);
        igl::per_vertex_normals(V,F,igl::PER_VERTEX_NORMALS_WEIGHTING_TYPE_ANGLE,FN,VN);
        igl::per_edge_normals(V,F,igl::PER_EDGE_NORMALS_WEIGHTING_TYPE_UNIFORM,FN,EN,E,EMAP);
        igl::signed_distance_pseudonormal(tempCenters,V,F,tree,FN,VN,EN,EMAP,distances,indices,tempPos,tempNormal);

        return VectorArrayEigen(tempNormal.cast<Float>());
    }
    */

    ScalarArrayEigen getSignedDist(const VectorArrayEigen& centers, const VectorArrayEigen& pos, const Eigen::Array<int, Eigen::Dynamic, 3, Eigen::RowMajor>& tri) {
        Eigen::MatrixXd tempCenters = centers.cast<double>().matrix();
        igl::AABB<Eigen::MatrixXd,3> tree;
        igl::FastWindingNumberBVH fwn_bvh;
        Eigen::MatrixXd V = pos.cast<double>().matrix();
        Eigen::MatrixXi F = tri.matrix();
        Eigen::MatrixXd distances;
        distances.resize(centers.rows(),1);
        distances.array() = 0.0;

        /////////////////////////////////////////////////////
        Eigen::MatrixXd FCenter = (V(F.col(0), Eigen::all) + V(F.col(1), Eigen::all) + V(F.col(2), Eigen::all))/3.0;

        Eigen::Array<int,-1,1> bFaceXMax = (((FCenter.array().col(0) - gP.posRefMax[0]).abs() < bcTol * spacing)).cast<int>() * (gP.bcTypeMaxAxis0>0);
        Eigen::Array<int,-1,1> bFaceYMax = (((FCenter.array().col(1) - gP.posRefMax[1]).abs() < bcTol * spacing)).cast<int>() * (gP.bcTypeMaxAxis1>0);
        Eigen::Array<int,-1,1> bFaceZMax = (((FCenter.array().col(2) - gP.posRefMax[2]).abs() < bcTol * spacing)).cast<int>() * (gP.bcTypeMaxAxis2>0);
        Eigen::Array<int,-1,1> bFaceXMin = (((FCenter.array().col(0) - gP.posRefMin[0]).abs() < bcTol * spacing)).cast<int>() * (gP.bcTypeMinAxis0>0);
        Eigen::Array<int,-1,1> bFaceYMin = (((FCenter.array().col(1) - gP.posRefMin[1]).abs() < bcTol * spacing)).cast<int>() * (gP.bcTypeMinAxis1>0);
        Eigen::Array<int,-1,1> bFaceZMin = (((FCenter.array().col(2) - gP.posRefMin[2]).abs() < bcTol * spacing)).cast<int>() * (gP.bcTypeMinAxis2>0);

        Eigen::VectorXi flag = (1-bFaceXMax)*(1-bFaceYMax)*(1-bFaceZMax)*(1-bFaceXMin)*(1-bFaceYMin)*(1-bFaceZMin);

//        Eigen::VectorXi flag = (1-bFaceXMax);//(((FCenter.array().col(0) - maxPosRef.x).abs() > bctolXmax)).cast<int>();

        Eigen::VectorXi ids;
/*
        int nnz = flag.sum();
        std::cout<<nnz<<" "<<FCenter.rows()<<" "<<F.rows()<<std::endl;
        ids.resize(1,nnz);
        {
            int k = 0;
            for(int j = 0;j<nnz;j++)
                if (flag(j)>0)
                    ids(k++) = j;
        }
*/
        igl::find(flag,ids);

        if (flag.sum()>0) {
            Eigen::MatrixXi newF = F(ids, Eigen::all);
            F = newF;
            Eigen::MatrixX3d tempPos;
            Eigen::MatrixXd FN, VN, EN;
            Eigen::MatrixXi E;
            Eigen::VectorXi EMAP;
            Eigen::MatrixXi indices;
            Eigen::MatrixX3d tempNormal;

            igl::per_face_normals(V, F, FN);
            igl::per_vertex_normals(V, F, igl::PER_VERTEX_NORMALS_WEIGHTING_TYPE_ANGLE, FN, VN);
            igl::per_edge_normals(V, F, igl::PER_EDGE_NORMALS_WEIGHTING_TYPE_UNIFORM, FN, EN, E, EMAP);
            tree.init(V, F);
            igl::signed_distance_pseudonormal(tempCenters, V, F, tree, FN, VN, EN, EMAP, distances, indices, tempPos,tempNormal);
        }
        return ScalarArrayEigen(distances.cast<Float>());
    }


    void drawLegend(igl::opengl::ViewerData& vd) {
        vd.clear();
        Float offset = 0.0;
        Eigen::MatrixXd V(4,3);
        V <<offset    ,-1, 0,   // bottom-left
            offset+0.2,-1, 0,   // bottom-right
            offset+0.2, 1, 0,   // top-right
            offset    , 1, 0;   // top-left
        Eigen::MatrixXi F(2,3);
        F << 0, 1, 2,   // first triangle: bottom-left → bottom-right → top-right
             0, 2, 3;   // second triangle: bottom-left → top-right → top-left
        Eigen::Matrix<double,4,1> d(0, 0, 1, 1);

        vd.show_custom_labels = vd.is_visible;
        vd.set_mesh(V, F); // V and F are your mesh vertices and faces
        vd.set_data(d, igl::COLOR_MAP_TYPE_PARULA);
        vd.show_lines = unsigned(0);


        vd.add_label(Eigen::Vector3d(offset,  1.4, 0) , "time = "+std::to_string(globalTime));
        vd.add_label(Eigen::Vector3d(offset,  1.2, 0) , "");
        vd.add_label(Eigen::Vector3d(offset, -1.1, 0) , std::to_string(NAN));
        vd.add_label(Eigen::Vector3d(offset,  1.1, 0) , std::to_string(NAN));

        vd.line_width = 2.0;
        vd.show_custom_labels = vd.is_visible;
        vd.label_size = 1.0;
    }

    void drawHistogram(igl::opengl::ViewerData &data, const ScalarArrayEigen &values, int num_bins) {
        if (values.size() == 0 || num_bins <= 0)
            return;

        // Compute min/max
        double xmin = values.minCoeff();
        double xmax = values.maxCoeff();

        // Build bin edges (num_bins + 1 edges)
        Eigen::VectorXd edges(num_bins + 1);
        for (int i = 0; i <= num_bins; ++i)
            edges[i] = xmin + (xmax - xmin) * double(i) / double(num_bins);

        // Compute histogram counts
        Eigen::VectorXi counts;
        Eigen::VectorXi temp;
        igl::histc(values.matrix(), edges, counts, temp);

        // Normalize heights for visualization
        double max_count = counts.maxCoeff();
        if (max_count <= 0)
            return;

        // Clear previous geometry
        data.clear();

        // Prepare mesh for bars
        Eigen::MatrixXd V(num_bins * 4, 3);   // 4 vertices per bar
        Eigen::MatrixXi F(num_bins * 2, 3);   // 2 triangles per bar
        Eigen::VectorXd qF(num_bins * 2);   // 4 vertices per bar
        Eigen::MatrixXd colors(num_bins * 2, 3);

        double xOfset = 0.0; //2.5
        double xExtent = 0.5;
        double yExtent = 2;
        data.add_label(Eigen::RowVector3d(xOfset - 0.2, yExtent * 0.5 + 0.1 , 0.0), "Mesh quality (mean ratio)");
        for (int i = 0; i < num_bins; ++i) {
            double y0 = ((edges[i]      - edges[0]) / (edges[num_bins] - edges[0]) + 0.2 / double(num_bins) - 0.5) * yExtent ;
            double y1 = ((edges[i + 1]  - edges[0]) / (edges[num_bins] - edges[0]) - 0.2 / double(num_bins) - 0.5) * yExtent;
            double w  = -(double(counts[i]) / max_count) * xExtent;  // normalized height

            int v = 4 * i;

            // Rectangle vertices
            V.row(v + 0) = Eigen::RowVector3d(xOfset, y0, 0.0);
            V.row(v + 1) = Eigen::RowVector3d(xOfset+w, y0, 0.0);
            V.row(v + 2) = Eigen::RowVector3d(xOfset+w, y1, 0.0);
            V.row(v + 3) = Eigen::RowVector3d(xOfset, y1, 0.0);
            // Two triangles
            int f = 2 * i;
            F.row(f + 0) = Eigen::RowVector3i(v + 2, v + 1, v + 0);
            F.row(f + 1) = Eigen::RowVector3i(v + 3, v + 2, v + 0);

            qF(f + 0) = (edges[i]+edges[i+1]) * 0.5;
            qF(f + 1) = (edges[i]+edges[i+1]) * 0.5;

            colors.row(f + 0) << 21./255., 252.0/255.0, 21.0/255.0;
            colors.row(f + 1) << 21./255., 252.0/255.0, 21.0/255.0;

            if ((edges[i]+edges[i+1]) * 0.5 < 0.2) {
                colors.row(f + 0) << 250./255., 252.0/255.0, 21.0/255.0;
                colors.row(f + 1) << 250./255., 252.0/255.0, 21.0/255.0;
            }
            if ((edges[i]+edges[i+1]) * 0.5 < 0.05) {
                colors.row(f + 0) << 250./255., 21.0/255.0, 21.0/255.0;
                colors.row(f + 1) << 250./255., 21.0/255.0, 21.0/255.0;
            }
            // ------------------------- // Add label at bar center // -------------------------

            if (i==0 || i==num_bins-1 || i%int(double(num_bins)/10.0)==0) {
                double yc = 0.5 * (y0 + y1);
                double xc = w - 0.02;
                // offset slightly to the right
                std::string label = std::to_string((edges[i]+edges[i+1]) * 0.5);
                data.add_label(Eigen::RowVector3d(xOfset + 0.1, yc, 0.0), label);
            }
        }

        // Send geometry to viewer
        data.set_mesh(V, F);
        data.show_custom_labels = data.is_visible;
        data.show_lines = unsigned(0);
        // Optional: color the bars

        // data.set_colors(Eigen::RowVector3d(250./255., 252.0/255.0, 21.0/255.0));
        data.set_colors(colors);
    }

    void drawAxis(igl::opengl::ViewerData& vd, CoordinateSystem coord) {
        Eigen::RowVector3d cax1(1.0,0.0,0.0);
        Eigen::RowVector3d cax2(0.0,1.0,0.0);
        Eigen::RowVector3d cax3(0.0,0.0,1.0);
        vd.clear();
        if (coord == CylindricalZ) {
            int N = 50;
            Float r = 0.25;
            Eigen::RowVector3d center(0.0, 0.0, 0.0);
            Eigen::ArrayXd theta = Eigen::ArrayXd::LinSpaced(N, 0.0, 2*M_PI);
            Eigen::ArrayXd x = r * theta.cos();
            Eigen::ArrayXd y = r * theta.sin();;
            Eigen::ArrayXd z = x * 0.0 + center.z();
            Eigen::ArrayX3d P1(N - 1, 3), P2(N - 1, 3);
            P1.col(0) = x(Eigen::seqN(0, N - 2)).transpose();
            P1.col(1) = y(Eigen::seqN(0, N - 2)).transpose();
            P1.col(2) = center.z();
            P2.col(0) = x(Eigen::seqN(1, N - 1)).transpose();
            P2.col(1) = y(Eigen::seqN(1, N - 1)).transpose();
            P2.col(2) = center.y();

            vd.add_edges(center, center + Eigen::RowVector3d(r, 0, 0), cax1);
            vd.add_edges(P1, P2, cax2);
            vd.add_edges(center, center + Eigen::RowVector3d(0, 0, r), cax3);

            vd.add_label(center + Eigen::RowVector3d(r*0.6, r*0.25, 0),"r");
            vd.add_label(center + Eigen::RowVector3d(0,r*1.2, 0),"theta");
            vd.add_label(center + Eigen::RowVector3d(0,0,r*1.2),"Z");
        } else if (coord == CylindricalY) {
            int N = 50;
            Float r = 0.25;
            Eigen::RowVector3d center(0, 0.0, 0.0);
            Eigen::ArrayXd theta = Eigen::ArrayXd::LinSpaced(N, 0.0, 2*M_PI);
            Eigen::ArrayXd x = r * theta.cos();
            Eigen::ArrayXd y = x * 0.0 + center.y();
            Eigen::ArrayXd z = r * theta.sin();;
            Eigen::ArrayX3d P1(N - 1, 3), P2(N - 1, 3);
            P1.col(0) = x(Eigen::seqN(0, N - 2)).transpose();
            P1.col(1) = center.y();
            P1.col(2) = z(Eigen::seqN(0, N - 2)).transpose();
            P2.col(0) = x(Eigen::seqN(1, N - 1)).transpose();
            P2.col(1) = center.y();
            P2.col(2) = z(Eigen::seqN(1, N - 1)).transpose();

            vd.add_edges(center, center + Eigen::RowVector3d(r, 0, 0), cax1);
            vd.add_edges(P1, P2, cax2);
            vd.add_edges(center, center + Eigen::RowVector3d(0, r, 0), cax3);

            vd.add_label(center + Eigen::RowVector3d(r*0.6, r*0.25, 0),"r");
            vd.add_label(center + Eigen::RowVector3d(0,r*1.2, 0),"Y");
            vd.add_label(center + Eigen::RowVector3d(0,0,r*1.2),"theta");
        } else if (coord == ConeAdapted) {
            int N = 50;
            Float alpha = coneDims.apexAng/2.0*M_PI/180.0;
            Float r =0.25;
            Float L = std::min(Float(1000.0*r), r / std::tan(alpha)) ;
            Float ra = std::max(r - L * std::tan(alpha), Float(0.0));
            Eigen::RowVector3d center(0, 0.0, 0.0);
            Eigen::ArrayXd theta = Eigen::ArrayXd::LinSpaced(N, 0.0, 2*M_PI);
            Eigen::ArrayXd x = r * theta.cos();
            Eigen::ArrayXd y = x * 0.0 + center.y();
            Eigen::ArrayXd z = r * theta.sin();;
            Eigen::ArrayX3d P1(N - 1, 3), P2(N - 1, 3);
            Eigen::ArrayX3d P3(N - 1, 3), P4(N - 1, 3);
            P1.col(0) = x(Eigen::seqN(0, N - 2)).transpose();
            P1.col(1) = center.y();
            P1.col(2) = z(Eigen::seqN(0, N - 2)).transpose();
            P2.col(0) = x(Eigen::seqN(1, N - 1)).transpose();
            P2.col(1) = center.y();
            P2.col(2) = z(Eigen::seqN(1, N - 1)).transpose();

            P3.col(0) = x(Eigen::seqN(0, N - 2)).transpose()*ra;
            P3.col(1) = center.y() + L;
            P3.col(2) = z(Eigen::seqN(0, N - 2)).transpose()*ra;
            P4.col(0) = x(Eigen::seqN(1, N - 1)).transpose()*ra;
            P4.col(1) = center.y() + L;
            P4.col(2) = z(Eigen::seqN(1, N - 1)).transpose()*ra;

            vd.add_edges(
                center + Eigen::RowVector3d(r, 0, 0),
                center + Eigen::RowVector3d(r + r*std::cos(alpha), r*std::sin(alpha), 0), cax1);
            //vd.add_edges(center, center + Eigen::RowVector3d(r, 0, 0), cax1);
            vd.add_edges(P1, P2, cax2);
            vd.add_edges(P3, P4, cax2);
            vd.add_edges(center, center + Eigen::RowVector3d(0, L, 0), Eigen::RowVector3d(0.0,0.0,0.0));
            vd.add_edges(
                center + Eigen::RowVector3d(r, 0, 0),
                center + Eigen::RowVector3d(r - L*std::tan(alpha), L, 0), cax3);

            vd.add_label(center + Eigen::RowVector3d(r + r*std::cos(alpha), r*std::sin(alpha), 0) * 1.15,"r");
            vd.add_label(center + Eigen::RowVector3d(r*1.15, L*0.5, 0),"s");
            vd.add_label(center + Eigen::RowVector3d(r*0.25, L*0.5, 0),"Y");
            vd.add_label(center + Eigen::RowVector3d(0,0,r*1.2),"theta");

        } else if (coord == Spherical) {
            Eigen::ArrayX3d xyz;
            Eigen::RowVector3d center(0.0, 0.0, 0.0);
            Float r = 0.25;
            const int N = 100;
            int M = 4;

            vd.add_edges(center, center + Eigen::RowVector3d(r,0,0), cax1);

            for (int i = 0; i < M; i++) {
                xyz = getSpherePhi(N, r, 2. * M_PI * (double) i / (double) M, 0, M_PI) + center.array().replicate<N, 1>();
                vd.add_edges(xyz(Eigen::seqN(0, N - 2), Eigen::all),
                                        xyz(Eigen::seqN(1, N - 1), Eigen::all), cax2);
            }

            M = 1;
            for (int i = -M; i <= M; i++) {
                xyz = getSphereTheta(N, r, M_PI * (double) i / (double) (2 * M), 0, 2.0 * M_PI) + center.array().replicate<N, 1>();
                vd.add_edges(xyz(Eigen::seqN(0, N - 2), Eigen::all),
                                        xyz(Eigen::seqN(1, N - 1), Eigen::all), cax3);
            }

            vd.add_label(center + Eigen::RowVector3d(r*0.6, r*0.25, 0),"r");
            vd.add_label(center + Eigen::RowVector3d(0,r*1.2, 0),"phi");
            vd.add_label(center + Eigen::RowVector3d(0,0,r*1.2),"theta");
        } else if (coord == Toroidal) {
            Eigen::ArrayX3d xyz;
            Float R = 0.25;
            Float r = 2*R/3.0;
            const int N = 100;

            int M;
            // int M = 20;
            // xyz = getTorusPhi(N, R, 0.0, 0.0, 0, 2.0 * M_PI);
            // vd.add_edges(xyz(Eigen::seqN(0, N - 2), Eigen::all),
            //                         xyz(Eigen::seqN(1, N - 1), Eigen::all), Eigen::RowVector3d(0,0,0));

            vd.add_edges(Eigen::RowVector3d(R,0,0),Eigen::RowVector3d(R+r,0,0), cax1);

            // for (int i = 0; i < M; i++) {
            //     Eigen::RowVector3d temp(R, 0.0, 0.0);
            //     vd.add_edges(temp, xyz(i, Eigen::all) + (xyz(i, Eigen::all) - temp.array()) * 0.5, cax1);
            // }

            M = 4;
            for (int i = 0; i < M; i++) {
                xyz = getTorusTheta(N, R, r, 2. * M_PI * (double) i / (double) (M), 0, 2.0 * M_PI);
                vd.add_edges(xyz(Eigen::seqN(0, N - 2), Eigen::all),
                                        xyz(Eigen::seqN(1, N - 1), Eigen::all), cax2);
            }

            M = 2;
            for (int i = 0; i <= M; i++) {
                xyz = getTorusPhi(N, R, r, M_PI * (double) i / (double) (M-1), 0, 2.0 * M_PI);
                vd.add_edges(xyz(Eigen::seqN(0, N - 2), Eigen::all),
                                        xyz(Eigen::seqN(1, N - 1), Eigen::all), cax3);
            }

            vd.add_label(Eigen::RowVector3d(R+r*0.6, r*0.25, 0),"r");
            vd.add_label(Eigen::RowVector3d(R+r*1.0,r*1.5, 0),"phi");
            vd.add_label(Eigen::RowVector3d(0,R+r*0.5,r*1.2),"theta");

        } else {
            double scale = 0.25;
            Eigen::RowVector3d center(0.0, 0.0, -0.0);
            vd.add_edges(center, center + Eigen::RowVector3d(scale, 0, 0), cax1);
            vd.add_edges(center, center + Eigen::RowVector3d(0, scale, 0), cax2);
            vd.add_edges(center, center + Eigen::RowVector3d(0, 0, scale), cax3);

            vd.add_label(Eigen::RowVector3d(1.1,00,00)*scale+center,"X");
            vd.add_label(Eigen::RowVector3d(00,1.1,00)*scale+center,"Y");
            vd.add_label(Eigen::RowVector3d(00,00,1.1)*scale+center,"Z");
        }
        vd.line_width = 2.0;
        vd.show_custom_labels = vd.is_visible;
        vd.label_size = 2.0;

    }

    GLuint compileShader(GLenum type, const char* source) {
        GLuint shader = glCreateShader(type);
        glShaderSource(shader, 1, &source, nullptr);
        glCompileShader(shader);

        // Check for errors
        GLint success;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if (!success) {
            char infoLog[512];
            glGetShaderInfoLog(shader, 512, nullptr, infoLog);
            std::cerr << "Shader compilation error:\n" << infoLog << std::endl;
        }

        return shader;
    }


    GLuint createShaderProgram(const char* vertexSrc, const char* fragmentSrc) {
        GLuint vertexShader = compileShader(GL_VERTEX_SHADER, vertexSrc);
        GLuint fragmentShader = compileShader(GL_FRAGMENT_SHADER, fragmentSrc);

        GLuint program = glCreateProgram();
        glAttachShader(program, vertexShader);
        glAttachShader(program, fragmentShader);
        glLinkProgram(program);

        // Check for linking errors
        GLint success;
        glGetProgramiv(program, GL_LINK_STATUS, &success);
        if (!success) {
            char infoLog[512];
            glGetProgramInfoLog(program, 512, nullptr, infoLog);
            std::cerr << "Shader linking error:\n" << infoLog << std::endl;
        }

        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);

        return program;
    }

    void setupColorBar(GLuint& VAO, GLuint& VBO, GLuint& EBO) {
        float vertices[] = {
            // x, y, value
            -0.9f, -0.9f, 0.0f,
             0.9f, -0.9f, 1.0f,
            -0.9f, -0.8f, 0.0f,
             0.9f, -0.8f, 1.0f
        };

        unsigned int indices[] = {
            0, 1, 2,
            1, 3, 2
        };

        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);
        glGenBuffers(1, &EBO);

        glBindVertexArray(VAO);

        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);

        glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)(2 * sizeof(float)));
        glEnableVertexAttribArray(1);
    }

    bool ColorbarPlugin::post_draw() {
        static GLuint VAO = 0, VBO = 0, EBO = 0;
        static GLuint shaderProgram = 0;

        if (shaderProgram == 0) {

            const char* vertexSrc = R"(
            #version 330 core
            layout(location = 0) in vec2 aPos;
            layout(location = 1) in float aValue;
            out float vValue;
            void main() {
                vValue = aValue;
                gl_Position = vec4(aPos, 0.0, 1.0);
            }
            )";

            const char* fragmentSrc = R"(
            #version 330 core
            in float vValue; // passed from vertex shader
            out vec4 color;

            vec3 colormap_parula(float t) {
                t = clamp(t, 0.0, 1.0);
                vec3 c0 = vec3(0.2081, 0.1663, 0.5292);
                vec3 c1 = vec3(0.2394, 0.3176, 0.7093);
                vec3 c2 = vec3(0.2666, 0.4786, 0.7766);
                vec3 c3 = vec3(0.3042, 0.6307, 0.7736);
                vec3 c4 = vec3(0.3685, 0.7804, 0.7250);
                vec3 c5 = vec3(0.5459, 0.8987, 0.6154);
                vec3 c6 = vec3(0.9946, 0.9062, 0.1439);

                if (t < 0.17) return mix(c0, c1, t / 0.17);
                else if (t < 0.33) return mix(c1, c2, (t - 0.17) / (0.33 - 0.17));
                else if (t < 0.5) return mix(c2, c3, (t - 0.33) / (0.5 - 0.33));
                else if (t < 0.67) return mix(c3, c4, (t - 0.5) / (0.67 - 0.5));
                else if (t < 0.83) return mix(c4, c5, (t - 0.67) / (0.83 - 0.67));
                else return mix(c5, c6, (t - 0.83) / (1.0 - 0.83));
            }

            vec3 colormap_jet(float t) {
                return vec3(
                    clamp(1.5 - abs(4.0 * t - 3.0), 0.0, 1.0),
                    clamp(1.5 - abs(4.0 * t - 2.0), 0.0, 1.0),
                    clamp(1.5 - abs(4.0 * t - 1.0), 0.0, 1.0)
                );
            }


            void main() {
                float t = clamp(vValue, 0.0, 1.0); // normalize y
                color = vec4(colormap_parula(t), 1.0);
            }
            )";



//             const char* vertexSrc = R"(
//             #version 330 core
//             layout(location = 0) in vec2 aPos;
//             layout(location = 1) in float aValue;
//             out float vValue;
//             void main() {
//                 vValue = aValue;
//                 gl_Position = vec4(aPos, 0.0, 1.0);
//             }
//         )";
//
//             const char* fragmentSrc = R"(
//             #version 330 core
//             in float vValue;
//             out vec4 FragColor;
//             void main() {
//                 FragColor = vec4(vValue, 0.0, 1.0 - vValue, 1.0);
//             }
//
//         )";

            shaderProgram = createShaderProgram(vertexSrc, fragmentSrc);
            setupColorBar(VAO, VBO, EBO);
        }

        glUseProgram(shaderProgram);
        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);


        // glTexImage2D()

        // float x_ndc = 0.95f;  // near right edge
        // float y_ndc = 0.95f;  // near top edge
        // float z_ndc = 0.0f;   // near plane
        // Eigen::Vector4f ndc_pos(x_ndc, y_ndc, z_ndc, 1.0f);
        // Eigen::Matrix4f proj_matrix = viewer->core().proj.cast<float>();
        // Eigen::Matrix4f view_matrix = viewer->core().view.cast<float>();
        // Eigen::Matrix4f view_proj_inv = (proj_matrix * view_matrix).inverse();
        // Eigen::Vector4f world_pos = view_proj_inv * ndc_pos;
        // world_pos /= world_pos.w();  // Perspective divide
        //

        std::cout<<viewer->core(1).viewport<<std::endl;
        // viewer->data(2).clear_labels();
        // viewer->data(2).add_label(world_pos.head<3>().cast<double>(), "Your Label");
        // // viewer->data().add_label(leftLabelPos, "Min");
        // // viewer->data().add_label(rightLabelPos, "Max");
        return false; // return true if you want to block default drawing
    }


    VideoWriter::VideoWriter(const std::string& filename, int width, int height, int fps)
        : filename(filename), width(width), height(height), fps(fps) {}

    VideoWriter::~VideoWriter() {
        cleanup();
    }

    void VideoWriter::addFrame(uint8_t* rgb) {
        uint8_t* in_data[1] = { rgb };
        int in_linesize[1] = { 4 * width };

        // Adapting values for flipping the buffer rgb.
        in_data[0] = rgb + (height - 1) * in_linesize[0];
        in_linesize[0] = -in_linesize[0];

        // Convert RGB → YUV420P
        sws_scale(sws, in_data, in_linesize, 0, height,
                  frame->data, frame->linesize);

        frame->pts = pts++;
        int ret = avcodec_send_frame(ctx, frame);
        if (ret < 0) return;

        AVPacket pkt;
        av_init_packet(&pkt);
        pkt.data = nullptr;
        pkt.size = 0;

        while (true) {
            ret = avcodec_receive_packet(ctx, &pkt);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
                break;
            if (ret < 0)
                break;

            // CRITICAL: rescale timestamps
            av_packet_rescale_ts(&pkt, ctx->time_base, stream->time_base);

            av_interleaved_write_frame(fmt, &pkt);
            av_packet_unref(&pkt);
        }
    }

    void VideoWriter::close() {
        cleanup();
    }

    int64_t pts = 0;

//private:
    std::string filename;
    int width, height, fps;

    AVFormatContext* fmt = nullptr;
    AVCodecContext* ctx = nullptr;
    AVCodec* codec = nullptr;
    AVStream* stream = nullptr;
    SwsContext* sws = nullptr;
    AVFrame* frame = nullptr;

    // -----------------------------
    // Initialization
    // -----------------------------
    void VideoWriter::init() {
        avformat_alloc_output_context2(&fmt, nullptr, nullptr, filename.c_str());

        codec = avcodec_find_encoder(AV_CODEC_ID_H264);
        stream = avformat_new_stream(fmt, codec);

        ctx = avcodec_alloc_context3(codec);
        ctx->width = width;
        ctx->height = height;
        ctx->pix_fmt = AV_PIX_FMT_YUV420P;
        ctx->time_base = AVRational{1, fps};
        ctx->framerate = AVRational{fps, 1};
        ctx->gop_size = 12;
        ctx->max_b_frames = 2;

        // Stream time base MUST match codec time base
        stream->time_base = ctx->time_base;

        av_opt_set(ctx->priv_data, "preset", "slow", 0);
        av_opt_set(ctx->priv_data, "crf", "18", 0);
        avcodec_open2(ctx, codec, nullptr);
        avcodec_parameters_from_context(stream->codecpar, ctx);

        if (!(fmt->oformat->flags & AVFMT_NOFILE))
            avio_open(&fmt->pb, filename.c_str(), AVIO_FLAG_WRITE);

        avformat_write_header(fmt, nullptr);

        sws = sws_getContext(
            width, height, AV_PIX_FMT_RGBA,
            width, height, AV_PIX_FMT_YUV420P,
            SWS_BILINEAR, nullptr, nullptr, nullptr);

        frame = av_frame_alloc();
        frame->format = ctx->pix_fmt;
        frame->width = width;
        frame->height = height;
        av_frame_get_buffer(frame, 32);
    }

    // -----------------------------
    // Cleanup + flush
    // -----------------------------
    void VideoWriter::cleanup() {
        if (!fmt || !ctx)
            goto free_only;

        // Flush encoder
        avcodec_send_frame(ctx, nullptr);

        {
            AVPacket pkt;
            av_init_packet(&pkt);
            pkt.data = nullptr;
            pkt.size = 0;

            while (true) {
                int ret = avcodec_receive_packet(ctx, &pkt);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
                    break;
                if (ret < 0)
                    break;

                av_packet_rescale_ts(&pkt, ctx->time_base, stream->time_base);
                av_interleaved_write_frame(fmt, &pkt);
                av_packet_unref(&pkt);
            }
        }

        av_write_trailer(fmt);

    free_only:
        if (sws) sws_freeContext(sws);
        if (frame) av_frame_free(&frame);
        if (ctx) avcodec_free_context(&ctx);

        if (fmt) {
            if (fmt->pb)
                avio_close(fmt->pb);
            avformat_free_context(fmt);
        }

        fmt = nullptr;
        ctx = nullptr;
        codec = nullptr;
        stream = nullptr;
        sws = nullptr;
        frame = nullptr;
    }
}