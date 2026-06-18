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


#ifndef LAGRANGIANSOLID_CUDA_MYGUI_H
#define LAGRANGIANSOLID_CUDA_MYGUI_H

#include "backends/imgui_impl_sdl.h"
#include "backends/imgui_impl_opengl3.h"
#include <SDL2/SDL.h>

#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiPlugin.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <imgui.h>
#include <igl/per_corner_normals.h>
#include <igl/marching_tets.h>
#include "GLFW/glfw3.h"
#include "DeviceData.h"
#include "Visualisation.h"
#include "Parameters.h"

#include "igl/find.h"
#include "igl/stb/write_image.h"

namespace LagSol {

    extern bool modelChanged;
    extern bool applyChanges;
    extern bool reloadMeshFile;
    extern bool compileFlag;
    extern bool defaultParamFlag;
    extern bool resetSimFlag;
    extern bool initSimFlag;
    extern bool renderNewData;
    extern bool setNewMeshToViewer;
    extern bool basisChanged;

    extern Parameters gP;


    extern int averageInterval;
    extern bool firstTimeSetMeshToViewer;
    extern bool simDiverged;
    extern bool saveVTK;
    extern bool savePLY;
    extern Float saveTime;
    extern bool saveMP4;
    extern int frames;
    extern int oldFrame;

    struct BaseDims {
        int nLayers = 1;
        float H[MAX_NLAYERS], spacing[MAX_NLAYERS];
    };

    struct BoxDims : public BaseDims {
        float L, W;
        BoxDims() {
            L = 50;
            W = 50;
            for (int i = 0; i < MAX_NLAYERS; i++) {
                H[i] = L / 10;
                spacing[i] = H[i] / 5.0;
            }
        }
    };

    struct DiskDims : public BaseDims {
        float R;
        DiskDims() {
            R = 25;
            for (int i = 0; i < MAX_NLAYERS; i++) {
                H[i] = R / 5;
                spacing[i] = H[i] / 5.0;
            }
        }
    };

    struct TubeDims  : public BaseDims {
        float L, Ri;
        TubeDims() {
            Ri = 25;
            L = 50;
            for (int i = 0; i < MAX_NLAYERS; i++) {
                H[i] = L / 10;
                spacing[i] = H[i] / 5.0;
            }
        }
    };

    struct ConeDims  : public BaseDims {
        float L, Ri, apexAng;
        ConeDims() {
            Ri = 25;
            apexAng = 60;
            L = 50;
            for (int i = 0; i < MAX_NLAYERS; i++) {
                H[i] = L / 10;
                spacing[i] = H[i] / 5.0;
            }
        }
    };

    struct SphereDims  : public BaseDims {
        float Ri;
        SphereDims() {
            Ri = 25;
            for (int i = 0; i < MAX_NLAYERS; i++) {
                H[i] = Ri / 5;
                spacing[i] = H[i] / 5.0;
            }
        }
    };

    struct TorusDims  : public BaseDims {
        float R;
        float Ri;
        TorusDims() {
            R = 25;
            Ri = 10;
            for (int i = 0; i < MAX_NLAYERS; i++) {
                H[i] = R / 5;
                spacing[i] = H[i] / 5.0;
            }
        }
    };

    extern BoxDims boxDims;
    extern DiskDims diskDims;
    extern TubeDims tubeDims;
    extern ConeDims coneDims;
    extern SphereDims sphereDims;
    extern TorusDims torusDims;
    // extern float domainSize[3+MAX_NLAYERS];
    // extern float domainDx[3];
    // extern float domainIsoDx[MAX_NLAYERS];
    // extern float domainRadii[1+MAX_NLAYERS];
    // extern float domainRTorus;




    extern float sliceMag[3];
    extern float timeFactor, enDiff[1000], kinEnergy;
    extern double FPS;
    extern double minTimeFactor;
    extern int sUpdate;
    extern int visibleData;
    extern bool showRank1Tensors[8];
    extern float scaleRank1Tensors[8];
    extern float fontScale;
    //static Float spacingRef;
    //static Float minCoord[3],maxCoord[3];
    extern float pert;
    extern int gWWidth, gWHeight;

    extern int selectedLayer;

    extern bool crinkleClip;
    extern bool inverseClip;
    extern int meshVisState;
    extern int sliceDir;
    extern bool paramChanged;
    extern bool colorChanged;
    extern bool bcChanged;
    extern double clockTime;

    extern size_t simIter;
    extern size_t maxIter;
    extern int searchIter;
    extern int iterCounter;
    extern int videoCounter;
    extern bool showBasis;
    extern bool showMeshQ;
    extern int numBins;
    extern bool showEdges;
    extern bool showOrtho;

    extern bool run;
    extern bool shutDown;

    extern Eigen::ArrayX3d triCenter;
    extern Eigen::MatrixXd sliceVertex;
    extern Eigen::MatrixXi sliceFace;
    extern Eigen::VectorXd sliceData;
    extern Eigen::MatrixXd Normals;
    extern Eigen::MatrixX4d layerFaceColors;

    extern Eigen::ArrayX3d triCenter2;
    extern Eigen::MatrixXd sliceVertex2;
    extern Eigen::MatrixXi sliceFace2;
    extern Eigen::VectorXd sliceData2;
    extern Eigen::MatrixXd Normals2;
    extern Eigen::MatrixX4d layerFaceColors2;

    extern Eigen::Array4Xf layersColorEigen;
    extern Eigen::Array4Xf rank1TensorColors;

    extern Float kinEnergyTol;
    extern Float timeThreshold;

    extern Eigen::Vector4f bkgColor;
    extern Eigen::Matrix<double,256,3> rgb;
    extern GLuint colorMapId;

    extern BasicGeometry inputGeom;
    extern std::string inputMeshFileName;
    extern std::string outputFileName;

    extern Mesh mesh;
    extern DeviceData data;
    extern DeviceDataPtrManaged* dataPtr;

    extern thrust::device_vector<Vector> oldPos;
    extern thrust::device_vector<Vector> oldPosRef;
    extern thrust::device_vector<Vector> oldVel;
    extern thrust::device_vector<Tensor> oldFp;
    extern Float oldGlobalTime;


    extern Float spacing;
    extern Float charL;
    extern Float charLRef;
    extern Float dt;
    extern Float maxDt;
    extern Vector maxPos;
    extern Vector minPos;
    extern Float minR;
    extern Float maxR;


    extern int nLayers;
    extern int nFibers;
    extern Float globalTime;


    extern float contactFact;
    extern float contactFactVel;
    extern float repThickness;
    extern Float bcTol;

    extern Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> R;
    extern Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> G;
    extern Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> B;
    extern Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic> A;

    extern Eigen::Quaternionf trackball_angle, current_trackball_angle;
    extern bool resetViewFlag;

    void recover_from_snapshot();
    void save_to_snapshot();
    void get_domain_limits(const VectorArrayDev& pos);
    bool compile();
    void init(bool fromMesh = true);
    bool octreeSearch(DeviceData& data, const Vector& _minPos, const Vector& _maxPos);
    void simulate();
    void draw_viewer_window();
}

#endif //LAGRANGIANSOLID_CUDA_MYGUI_H
