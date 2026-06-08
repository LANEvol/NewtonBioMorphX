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

#ifndef LAGRANGIANSOLID_VISUAL_H
#define LAGRANGIANSOLID_VISUAL_H
#include "DataStructure.h"
#include "MeshIO.h"
#include <igl/opengl/glfw/Viewer.h>
#include "Eigen/Core"
#include <GL/gl.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
}


#define RAND_FLOAT (((double)rand()/(double )RAND_MAX-0.5)*2.0)
namespace LagSol {
    Eigen::ArrayX3d getSphereTheta(int N, const Float& R,const Float& phi0,const Float& theta0, const Float& dTheta);
    Eigen::ArrayX3d getSpherePhi(int N, const Float& R,const Float& theta0,const Float& phi0, const Float& dPhi);
    Eigen::ArrayX3d getTorusTheta(int N, const Float& R, const Float& r,const Float& phi0,const Float& theta0, const Float& dTheta);
    Eigen::ArrayX3d getTorusPhi(int N, const Float& R, const Float& r,const Float& theta0,const Float& phi0, const Float& dPhi);
    VectorArrayEigen getNormals(const VectorArrayEigen& centers, const VectorArrayEigen& pos, const Eigen::Array<int, Eigen::Dynamic, 3, Eigen::RowMajor>& tri);
    ScalarArrayEigen getSignedDist(const VectorArrayEigen& centers, const VectorArrayEigen& pos, const Eigen::Array<int, Eigen::Dynamic, 3, Eigen::RowMajor>& tri);
    void drawLegend(igl::opengl::ViewerData& vd);
    void drawAxis(igl::opengl::ViewerData& vd, CoordinateSystem coord);
    void drawHistogram(igl::opengl::ViewerData &data, const ScalarArrayEigen &values, int num_bins);


    class ColorbarPlugin : public igl::opengl::glfw::ViewerPlugin {
    public:
        bool post_draw() override;
        ColorbarPlugin() {};
    };

    class VideoWriter {
    public:
        VideoWriter(const std::string& filename, int width, int height, int fps);
        ~VideoWriter();
        void addFrame(uint8_t* rgb);
        void close();

        int64_t pts = 0;
        std::string filename;
        int width, height, fps;

        AVFormatContext* fmt = nullptr;
        AVCodecContext* ctx = nullptr;
        AVCodec* codec = nullptr;
        AVStream* stream = nullptr;
        SwsContext* sws = nullptr;
        AVFrame* frame = nullptr;

        void init();
        void cleanup();
    };

}

#endif //LAGRANGIANSOLID_VISUAL_H
