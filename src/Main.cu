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

#include <chrono>
#include <iostream>
#include <typeinfo>
#include <any>
#include <bitset>

#include <immintrin.h>
#include "thrust/extrema.h"
#include <thrust/iterator/constant_iterator.h>

#include "MeshIO.h"
#include "thrust/logical.h"
#include "Kernels.h"
#include "Visualisation.h"

#include "glad/glad.h"            // Initialize with gl3wInit()

#include "MyGUI.h"
#include <thread>

#include <filesystem>
#include <GL/gl.h>
#include <GL/gl.h>
namespace fs = std::filesystem;

#include <mutex>
#include <condition_variable>
#include <unordered_map>
#include "TextIO.h"

#include <Eigen/Core>
#include <imgui.h>
#include <imgui_internal.h>
#include "unistd.h"
#include "sys/stat.h"
#include "MeshOp.h"

std::mutex mtx;
std::condition_variable runSim;
using namespace LagSol;

//std::tuple<std::string, std::any, int> varName{"aaaa",&data.lamInt,-1};
std::vector<std::tuple<std::string, std::any, int>> varNamesColor = {
    {"Layers", &data.layer,-1}, {"Velocity", &data.vel,-1},
    {"Shear stress", &data.vonMises,-1}, {"Pressure", &data.pressure,-1},
    {"Plastic deformation", &data.Fp,-2}, {"Growth deformation", &data.Fg,-2},
    // {"Actin filaments", &data.actinTetra_vis,-1},
    // {"Fiber 1", &data.fiberTetra1_vis,-1}, {"Fiber 2", &data.fiberTetra2_vis,-1},
    // {"Fiber 3", &data.fiberTetra3_vis,-1}, {"Fiber 4", &data.fiberTetra4_vis,-1},
    // {"Gene 0", &data.genes, 0}, {"Gene 1", &data.genes, 1},
    // {"Gene 2", &data.genes, 2}, {"Gene 3", &data.genes, 3},
    // {"Gene 4", &data.genes, 4}, {"Gene 5", &data.genes, 5},
    // {"force", &data.force,-1}, {"lamInt", &data.lamInt,-1},
};

std::vector<std::tuple<std::string, TensorArrayDev*, int>> rank1TensorsNames = {
    {"growth rate", &data.grRate1,-1},
    {"growth rate", &data.grRate2,-1},
    {"growth rate", &data.grRate3,-1},
    {"Actin filaments", &data.actin,-1},
    {"Fiber 1", &data.fiber1,-1},
    {"Fiber 2", &data.fiber2,-1},
    {"Fiber 3", &data.fiber3,-1},
    {"Fiber 4", &data.fiber4,-1},
};

std::unordered_map<std::string, std::any> dataToSave = {
    {"layer", &data.layer},
    {"vel", &data.vel},
    // {"X", &data.posRef},
    {"vonMisesTetra", &data.vonMises} ,
    {"pressure", &data.pressure} ,
    // {"ScalarFun", &data.scalarFunction},
    // {"actin1Deca", &data.actin1Deca},
    // {"actin2Deca", &data.actin2Deca},
    // {"fiber1", &data.fiber1},
    // {"fiber2", &data.fiber2},
    // {"fiber3", &data.fiber3},
    // {"fiber4", &data.fiber4},
    // {"stress", &data.stress},
    // {"Fg", &data.Fg},
    // {"Fp", &data.Fp},
    // {"actin1", &data.actin1},
    // {"actin1Prod", &data.actin1Prod},
    // {"actin1Diff", &data.actin1Diff},
    // {"actin2", &data.actin2},
    // {"actin2Prod", &data.actin2Prod},
    // {"actin2Diff", &data.actin2Diff},
    // {"genes", &data.genes},
    // {"genesProd", &data.genesProd},
    // {"genesDiff", &data.genesDiff},
    // {"genesDeca", &data.genesDeca},
    // {"tempVec1", &data.tempVec1},
    // {"tempVec2", &data.tempVec2},
    // {"tempVec3", &data.tempVec3},
    // {"tempVec4", &data.tempVec4},

    // {"tempTens1", &data.tempTens1},
    // {"tempTens2", &data.tempTens2},
    // {"tempTens3", &data.tempTens3},
    // {"tempTens4", &data.tempTens4},

    // {"lamInt", &data.lamInt},
    // {"normal", &data.normalTetra},
};

void clampScalarArray(ScalarArrayEigen& scalar, Float& minS, Float& maxS, const Float & alpha) {
    minS = NAN;
    maxS = NAN;
    if (scalar.rows()>0) {
        std::vector<Float> percentile(scalar.data(), scalar.data() + scalar.size());
        int indMin = std::max(static_cast<int>(alpha * percentile.size()),int(0));
        int indMax = std::min(static_cast<int>((1.0-alpha) * percentile.size()),int(percentile.size()-1));
        std::nth_element(percentile.begin(), percentile.begin() + indMin, percentile.end());
        std::nth_element(percentile.begin(), percentile.begin() + indMax, percentile.end());
        minS = percentile[indMin];
        maxS = percentile[indMax];
        std::transform(scalar.data(), scalar.data() + scalar.size(), scalar.data(),
                       [minS,maxS](const Float &f) { return std::clamp(f, minS, maxS); });
    }
}

bool isSimDiverged() {
    get_domain_limits(data.pos);
    return thrust::any_of(data.force.begin(), data.force.end(), is_not_a_finite_Vector())
    // || ((minPos - maxPos).mag() > charLRef*5.0f)
    || thrust::any_of(data.vel.begin(), data.vel.end(), is_not_a_finite_Vector(charLRef/maxDt));
}

std::string projectFileName;
std::chrono::time_point<std::chrono::system_clock> mStartTime;
Float mGrowthProgress;

int runMode = 0;

bool ButtonCenteredOnLine(const char* label, float alignment = 0.5f) {
    ImGuiStyle& style = ImGui::GetStyle();
    float size = ImGui::CalcTextSize(label).x + style.FramePadding.x * 2.0f;
    float avail = ImGui::GetContentRegionAvail().x;
    float off = (avail - size) * alignment;
    if (off > 0.0f)
        ImGui::SetCursorPosX(ImGui::GetCursorPosX() + off);
    return ImGui::Button(label);
}

bool inline ImGuiMessage(const bool& cond, const std::string& title, const std::string& str) {
    bool res = false;
    if (cond)
        ImGui::OpenPopup(title.c_str());
    if (ImGui::BeginPopupModal(title.c_str())) {
        ImGui::Text(str.c_str());
        if (ButtonCenteredOnLine("Close")) {
            ImGui::CloseCurrentPopup();
            res = true;
        }
        // if (!cond)
        //     ImGui::CloseCurrentPopup();
        ImGui::EndPopup();
    }
    return res;
}

void getAverageAndSTD(float* avgSTD, const float *data, int N) {
    int wSum = 0;
    avgSTD[0] = 0.0f;
    avgSTD[1] = 0.0f;
    for (int i=0; i<N; i++) {
        if (std::isfinite(data[i])) {
            avgSTD[0] += data[i];
            wSum += 1;
        }
    }
    if (wSum>0) {
        avgSTD[0] = avgSTD[0] / Float(wSum);
        for (int i=0; i<N; i++) {
	    if (std::isfinite(data[i]))
            avgSTD[1] += (data[i] - avgSTD[0]) * (data[i] - avgSTD[0]);
        }
        avgSTD[1] = sqrt(avgSTD[1] / (Float(wSum)-1.0));
    } else {
    	avgSTD[0] = NAN;
    	avgSTD[1] = NAN;
    }
    if (wSum<2) {
    	avgSTD[1] = NAN;        
    }	    
}

#define IMGUI_DISABLE_WIDGET {ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true); ImGui::PushStyleVar(ImGuiStyleVar_Alpha, ImGui::GetStyle().Alpha * 0.5f);}
#define IMGUI_ENABLE_WIDGET {ImGui::PopItemFlag(); ImGui::PopStyleVar();}

void inline ImGuiParamReadRow(const std::string& lable, std::string &expression , bool &useMeshDef, const bool meshDefAvailable, bool& paramChanged) {
    ImGui::TableNextRow();
    ImGui::TableNextColumn();
    ImGui::Text(lable.c_str());
    ImGui::TableNextColumn();

    int bufSize = 1024;
    char buf[bufSize];
    std::fill_n(buf, 1024, 0);
    std::copy_n(expression.begin(), std::min(bufSize, (int) expression.size()), buf);

    if (!meshDefAvailable)
        IMGUI_DISABLE_WIDGET
    if (ImGui::Checkbox(("##File"+lable).c_str(), &useMeshDef))
        paramChanged = true;
    if (ImGui::IsItemHovered())
        ImGui::SetTooltip(("Use mesh file data for "+lable).c_str());
    if (!meshDefAvailable)
        IMGUI_ENABLE_WIDGET


    ImGui::SameLine();

    if (useMeshDef && meshDefAvailable)
        IMGUI_DISABLE_WIDGET

    if (ImGui::InputTextEx(("##"+lable).c_str(), NULL, buf, bufSize, ImVec2(-1,0), ImGuiInputTextFlags_CtrlEnterForNewLine)) {
        expression = std::string(buf);
        paramChanged = true;
    }
    if (useMeshDef && meshDefAvailable)
        IMGUI_ENABLE_WIDGET

}


VectorArrayEigen computeMaxEigenvectors_mapped(const TensorArrayEigen& mats) {
    using Mat3 = Eigen::Matrix<Float, 3, 3, Eigen::RowMajor>;
    using Vec3 = Eigen::Vector3<Float>;

    const int n = mats.rows();
    VectorArrayEigen result(n, 3);
    result.setZero();

    for (int i = 0; i < n; ++i) {
        // Map the row directly to a 3×3 matrix (no copy)
        Eigen::Map<const Mat3> A(mats.row(i).data());
        // Symmetrize: S = 1/2 (A + Aᵀ)
        Mat3 S = 0.5 * (A + A.transpose());

        // Eigen decomposition (fast for symmetric matrices)
        try {
            Eigen::SelfAdjointEigenSolver<Mat3> solver(S);
            // Largest eigenvector (eigenvalues sorted ascending)
            Vec3 v = solver.eigenvectors().col(2) * std::sqrt(solver.eigenvalues()(2));
            result.row(i) = v.transpose();
        } catch (const std::exception& e) {
        }

        // Store result
    }
    return result;
}


TensorArrayEigen device_to_eigen_tensor(const TensorArrayDev& devData) {
    const int nComp = sizeof(Tensor)/sizeof(Float);
    typedef Eigen::Map<Eigen::Array<Float,Eigen::Dynamic,nComp,Eigen::RowMajor>> EigenMapType;
    TensorArrayHost hostData(devData);
    EigenMapType mappedData(reinterpret_cast<Float *>(hostData.data()), hostData.size(), nComp);
    return TensorArrayEigen(mappedData);
};

template<class  T>
ScalarArrayEigen device_to_vis_data(const thrust::device_vector<T>& devData, const Eigen::VectorXi& tetMap, int component) {
    const int nComp = sizeof(T)/sizeof(Float);
    typedef Eigen::Map<Eigen::Array<Float,Eigen::Dynamic,nComp,Eigen::RowMajor>> EigenMapType;
    thrust::host_vector<T> hostData(devData);
    EigenMapType mappedData(reinterpret_cast<Float *>(hostData.data()), hostData.size(), nComp);
    ScalarArrayEigen visData;
    if (component>=0 && component < nComp)
        visData = mappedData.col(component);
    else if ((component == -2) && nComp==9)
        visData = (mappedData.rowwise().squaredNorm() - Float(3.0)).abs().sqrt();
    else
        visData = mappedData.rowwise().norm();
    if (devData.size() == mesh.ntet)
        visData = visData(tetMap);
    return visData;
};

template<>
ScalarArrayEigen device_to_vis_data<Float>(const thrust::device_vector<Float>& devData, const Eigen::VectorXi& tetMap, int component) {
    thrust::host_vector<Float> hostData(devData);
    ScalarArrayEigen visData(Eigen::Map<ScalarArrayEigen>(reinterpret_cast<Float *>(hostData.data()), hostData.size()));
    if (hostData.size() == mesh.ntet)
        visData = visData(tetMap);
    return visData;
}

template<class  T>
ScalarArrayEigen device_to_vis_data_section(const thrust::device_vector<T>& devData, const Eigen::VectorXi& tetMap, const Eigen::SparseMatrix<Float>& BC, int component) {
    const int nComp = sizeof(T)/sizeof(Float);
    typedef typename Eigen::Map<Eigen::Array<Float,Eigen::Dynamic,nComp,Eigen::RowMajor>> EigenMapType;
    thrust::host_vector<T> hostData(devData);
    EigenMapType mappedData(reinterpret_cast<Float *>(hostData.data()), hostData.size(), nComp);
    ScalarArrayEigen visData;
    if (component>=0 && component < nComp)
        visData = mappedData.col(component);
    else if ((component == -2) && nComp==9)
        visData = (mappedData.rowwise().squaredNorm() - Float(3.0)).abs().sqrt();
    else
        visData = mappedData.rowwise().norm();
    if (hostData.size() == mesh.ntet)
        visData = visData(tetMap);
    else
        visData = (BC * visData.matrix()).array();

    return visData;
};

template<>
ScalarArrayEigen device_to_vis_data_section<Float>(const thrust::device_vector<Float>& devData, const Eigen::VectorXi& tetMap, const Eigen::SparseMatrix<Float>& BC, int component) {
    thrust::host_vector<Float> hostData(devData);
    ScalarArrayEigen visData(Eigen::Map<ScalarArrayEigen>(reinterpret_cast<Float *>(hostData.data()), hostData.size()));
    if (hostData.size() == mesh.ntet)
        visData = visData(tetMap);
    else
        visData = (BC * visData.matrix()).array();
    return visData;
}


ScalarArrayEigen device_to_vis_data_tet_mesh(const std::any &selectedData, int selectedDataComp, const Eigen::VectorXi& tetMap) {
    ScalarArrayEigen tempData;
    if (selectedData.type() == typeid(ScalarArrayDev*)) {
        auto cData = std::any_cast<ScalarArrayDev*>(selectedData);
        tempData = device_to_vis_data(*cData, tetMap, selectedDataComp);
    } else if (selectedData.type() == typeid(VectorArrayDev*)) {
        auto cData = std::any_cast<VectorArrayDev*>(selectedData);
        tempData = device_to_vis_data(*cData, tetMap, selectedDataComp);
    } else if (selectedData.type() == typeid(TensorArrayDev*)) {
        auto cData = std::any_cast<TensorArrayDev*>(selectedData);
        tempData = device_to_vis_data(*cData, tetMap, selectedDataComp);
    }
    return tempData;
}

ScalarArrayEigen device_to_vis_data_tet_section(const std::any &selectedData, int selectedDataComp, const Eigen::VectorXi& tetMap, const Eigen::SparseMatrix<Float>& BC) {
    ScalarArrayEigen tempData;
    if (selectedData.type() == typeid(ScalarArrayDev*)) {
        auto cData = std::any_cast<ScalarArrayDev*>(selectedData);
        tempData = device_to_vis_data_section(*cData, tetMap, BC, selectedDataComp);
    } else if (selectedData.type() == typeid(VectorArrayDev*)) {
        auto cData = std::any_cast<VectorArrayDev*>(selectedData);
        tempData = device_to_vis_data_section(*cData, tetMap, BC, selectedDataComp);
    } else if (selectedData.type() == typeid(TensorArrayDev*)) {
        auto cData = std::any_cast<TensorArrayDev*>(selectedData);
        tempData = device_to_vis_data_section(*cData, tetMap, BC, selectedDataComp);
    }
    return tempData;
}

int main(int argc, char** argv) {

    gP2.init();
    gP.init();

    if (argc > 1) {
        for (int i=0; i<averageInterval; i++)
            enDiff[i] = INFINITY;

        projectFileName = std::string(argv[1]);
        std::cout<<"reading project from "<<projectFileName<<std::endl;
        readSetting(projectFileName);

        if (inputGeom == Arbitrary) {
            if (inputMeshFileName.compare(inputMeshFileName.length() - 4, inputMeshFileName.length(), ".vtk") == 0) {
                std::cout<<"loading mesh from "<<inputMeshFileName<<std::endl;
                if (mesh.init_from_file(inputMeshFileName)>0)
                    init();
                else
                    _ERROR_MESSAGE("Input mesh can't be loaded !");
            } else
                _ERROR_MESSAGE("Input mesh in not a vtk file !");
        }  else if (inputGeom == Box) {
            mesh.init_from_multi_layer_box(boxDims.L,boxDims.W,
                std::vector<double>(boxDims.H, boxDims.H + nLayers),
                std::vector<double>(boxDims.spacing, boxDims.spacing + nLayers));
            init();
        }  else if (inputGeom == Disk) {
            mesh.init_from_multi_layer_disk(diskDims.R,
                std::vector<double>(diskDims.H, diskDims.H + nLayers),
                std::vector<double>(diskDims.spacing, diskDims.spacing + nLayers));
            init();
        }  else if (inputGeom == Tube) {
            std::vector<double> radii; radii.push_back(tubeDims.Ri);
            for (int i=0; i<nLayers; i++) {radii.push_back(radii.back() + tubeDims.H[i]);}
            mesh.init_from_multi_layer_tube(tubeDims.L ,radii, std::vector<double>(tubeDims.spacing, tubeDims.spacing + nLayers));
            init();
        }  else if (inputGeom == Cone) {
            std::vector<double> radii; radii.push_back(coneDims.Ri);
            for (int i=0; i<nLayers; i++) {radii.push_back(radii.back() + coneDims.H[i]/std::cos(coneDims.apexAng * 0.5 * M_PI / 180.0));}
            mesh.init_from_multi_layer_cone(coneDims.L,coneDims.apexAng * M_PI / 180.0, radii, std::vector<double>(coneDims.spacing, coneDims.spacing + nLayers));
            init();
        }  else if (inputGeom == Sphere) {
            std::vector<double> radii; radii.push_back(sphereDims.Ri);
            for (int i=0; i<nLayers; i++) {radii.push_back(radii.back() + sphereDims.H[i]);}
            mesh.init_from_multi_layer_sphere(radii, std::vector<double>(sphereDims.spacing, sphereDims.spacing + nLayers));
            init();
        }  else if (inputGeom == Torus) {
            std::vector<double> radii; radii.push_back(torusDims.Ri);
            for (int i=0; i<nLayers; i++) {radii.push_back(radii.back() + torusDims.H[i]);}
            mesh.init_from_multi_layer_torus(torusDims.R, radii, std::vector<double>(torusDims.spacing, torusDims.spacing + nLayers));
            init();
        }

        get_domain_limits(data.pos);
        layerFaceColors.resize(mesh.ntri, 4);

        runMode = 1;
        colorChanged = false;
        bcChanged = false;
        modelChanged = true;
        paramChanged = true;
    };

    // For Linux systems
    // std::filesystem::current_path(std::filesystem::canonical("/proc/self/exe").parent_path()); //setting path

    mStartTime = std::chrono::system_clock::now();
    mGrowthProgress = 0.0;

    auto nogui = [] {
        layerFaceColors = layersColorEigen(Eigen::all, mesh.layer(mesh.boundaryTetIds) - 1).cast<double>().transpose();

        data.isRigid.assign(mesh.nver, 0);
        cudaDeviceSynchronize();
        _LAUNCH(mesh.ntet, 256, mark_rigid_nodes) (dataPtr, mesh.ntet);
        cudaDeviceSynchronize();
        if (compile()) {
            paramChanged = false;
        } else {
            _ERROR_MESSAGE("failed to compile nvrtc kernels.")
        }
        compileFlag = false;
        resetSimFlag = true;

        bcChanged = false;
        // _LUNCH(mesh.ntet, 256, compute_nodalENu) (dataPtr, mesh.ntet);
        // cudaDeviceSynchronize();
        _LAUNCH(mesh.nver, 256, compute_bids) (dataPtr, bcTol*spacing, mesh.nver);
        cudaDeviceSynchronize();

        ScalarArrayEigen distance = getSignedDist(mesh.pos,mesh.pos,mesh.tri);
        ScalarArrayDev distanceDev(mesh.nver,0.0);
        thrust::copy(reinterpret_cast<Float*>(distance.data()), reinterpret_cast<Float*>(distance.data())+mesh.nver, distanceDev.begin());
        _LAUNCH(mesh.ntet, 256, compute_normal_from_dist) (dataPtr, thrust::raw_pointer_cast(distanceDev.data()), mesh.ntet);
        cudaDeviceSynchronize();
        _LAUNCH(mesh.ntet, 256, compute_orthogonal_basis) (dataPtr, mesh.ntet);
        cudaDeviceSynchronize();

        run = true;
        runSim.notify_all();
        int k = 0;
        while (!shutDown) {
            if (std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - mStartTime).count()>1000.0) {
                // float wSum = 1e-10;
                // float enDiffAvgSum = 0.0;
                // for (int tt=0; tt<averageInterval; tt++) {
                //     if (std::isfinite(enDiff[tt])) {
                //         enDiffAvgSum += enDiff[tt];
                //         wSum += 1.0;
                //     }
                // }
                // float enDiffAvg = enDiffAvgSum / wSum;

                float avgSTD[2] = {INFINITY, INFINITY};
                getAverageAndSTD(avgSTD, enDiff, averageInterval);
                float energySTD = avgSTD[1];

                std::cout<<"time "<<globalTime<<
                           " , dt ( " <<timeFactor<<" * "<< maxDt <<" )" <<
                           " , iter "<<simIter<<", total energy std "<<energySTD<<", fps "<< FPS <<std::endl;
                mStartTime = std::chrono::system_clock::now();
            }

            if (shutDown) {
                if (isSimDiverged()) {
                    recover_from_snapshot();
                }

                // thrust::host_vector<Vector> tempPos(data.pos);
                // Eigen::Map<VectorArrayEigen> visPos(&tempPos[0][0], mesh.nver, 3);
                // Eigen::MatrixXf visPosA(visPos.cast<float>());
                // ScalarArrayEigen visR,visG,visB;
                // computeNodalColor(visR,visG,visB);
                // Eigen::MatrixXd nodalColor(mesh.boundaryNodeIds.size(),3);
                // nodalColor.col(0) = visR(mesh.boundaryNodeIds).cast<double>();
                // nodalColor.col(1) = visG(mesh.boundaryNodeIds).cast<double>();
                // nodalColor.col(2) = visB(mesh.boundaryNodeIds).cast<double>();
                // writePLY(output,
                //         visPosA(mesh.boundaryNodeIds,Eigen::all).cast<double>(),
                //         nodalColor,
                //         mesh.tri_mapped.matrix(),
                //         "growth "+std::to_string(growthProgress)+
                //         " iterations "+std::to_string(simiter)+
                //         " timeFactor "+std::to_string(timeFactor)+
                //         " kinetic/total energy "+std::to_string(enDiffAvg)+
                //         " fps "+std::to_string(FPS));
                // fs::permissions(output,fs::perms::owner_all | fs::perms::group_all | fs::perms::others_all);
                // std::cout<<"\t# "<<output<<" is saved." <<std::endl;

                if (saveVTK) {
                    std::string fname = outputFileName;
                    fname.replace(fname.end()-4,fname.end(),".vtk");
                    mesh.save_binary(fname, data.tet, data.pos, dataToSave);
                    fs::permissions(fname,fs::perms::owner_all | fs::perms::group_all | fs::perms::others_all);
                    std::cout<<"\t# "<<fname<<" is saved." <<std::endl;
                }
            }
        }
    };

    auto draw_mesh = [] {
        VideoWriter *videoWriter = nullptr;
        igl::opengl::glfw::Viewer viewer;

        viewer.launch_init(false,"NewtonBioMorphX",0,0);
        viewer.append_mesh(); // data_id = 1
        viewer.append_mesh(); // data_id = 2
        viewer.append_mesh(); // data_id = 3
        viewer.append_mesh(); // data_id = 4

        //////////////////////////////////////////////////////////////////////////////////////
        viewer.append_core(viewer.core_list[0].viewport);
        viewer.append_core(viewer.core_list[0].viewport);
        viewer.append_core(viewer.core_list[0].viewport);

        igl::opengl::ViewerData* vd_mesh  = &viewer.data(0);
        igl::opengl::ViewerData* vd_mesh2 = &viewer.data(1);
        igl::opengl::ViewerData* vd_axis = &viewer.data(2);
        igl::opengl::ViewerData* vd_lege = &viewer.data(3);
        igl::opengl::ViewerData* vd_hist = &viewer.data(4);

        igl::opengl::ViewerCore* vc_mesh  = &viewer.core_list[0];
        // igl::opengl::ViewerCore* vc_mesh2 = &viewer.core_list[1];
        igl::opengl::ViewerCore* vc_axis = &viewer.core_list[1];
        igl::opengl::ViewerCore* vc_lege = &viewer.core_list[2];
        igl::opengl::ViewerCore* vc_hist = &viewer.core_list[3];


        vd_mesh->line_width = 1.0;
        vd_mesh2->line_width = 1.0;

        vc_mesh->background_color = bkgColor;
        vc_mesh->rotation_type = igl::opengl::ViewerCore::ROTATION_TYPE_TRACKBALL;
        vc_mesh->is_animating = true;
        vc_mesh->animation_max_fps = 30.;
        vc_mesh->trackball_angle = trackball_angle;

        // sync_core(*vc_mesh, vc_mesh2);

        vc_lege->background_color = bkgColor;
        vc_lege->view = Eigen::Matrix4f::Identity(); // No camera rotation
        vc_lege->camera_zoom = 1.0f;
        vc_lege->camera_translation = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
        vc_lege->viewport = Eigen::Vector4f(
                            vc_mesh->viewport[0] - vc_mesh->viewport[2] * 3.0/8.0,
                            vc_mesh->viewport[1],
                            vc_mesh->viewport[2],
                            vc_mesh->viewport[3]);
        vc_lege->orthographic = true;
        vc_lege->is_animating = true;
        vc_lege->animation_max_fps = 30.;

        vc_axis->background_color = bkgColor;
        vc_axis->view = Eigen::Matrix4f::Identity(); // No camera rotation
        vc_axis->camera_zoom = 1.0f;
        vc_axis->camera_translation = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
        vc_axis->viewport = Eigen::Vector4f(
            vc_mesh->viewport[0]-vc_mesh->viewport[2]*0.375,
            vc_mesh->viewport[1]-vc_mesh->viewport[3]*0.375,
            vc_mesh->viewport[2],
            vc_mesh->viewport[3]);
        vc_axis->is_animating = true;
        vc_axis->animation_max_fps = 30.;

        vc_hist->background_color = bkgColor;
        vc_hist->view = Eigen::Matrix4f::Identity(); // No camera rotation
        vc_hist->camera_zoom = 1.0f;
        vc_hist->camera_translation = Eigen::Vector3f(0.0f, 0.0f, 0.0f);
        vc_hist->orthographic = true;
        vc_hist->is_animating = true;
        vc_hist->animation_max_fps = 30.;
        vc_hist->viewport = Eigen::Vector4f(
                            vc_mesh->viewport[0] + vc_mesh->viewport[2] * 3.0/8.0,
                            vc_mesh->viewport[1],
                            vc_mesh->viewport[2],
                            vc_mesh->viewport[3]);


        viewer.callback_key_down = [&](igl::opengl::glfw::Viewer &v, unsigned char key, int modifier) -> bool{
            glfwSetWindowShouldClose(viewer.window, GL_FALSE);
            return false;
        };

        viewer.callback_pre_draw = [&](igl::opengl::glfw::Viewer &v) -> bool {
            // sync_core(*vc_mesh, vc_mesh2);
            vc_axis->trackball_angle = vc_mesh->trackball_angle;
            current_trackball_angle = vc_mesh->trackball_angle;
            return false;
        };

        viewer.callback_post_resize = [&](igl::opengl::glfw::Viewer& v, int w, int h) {
            // Your adaptive behavior here
            glViewport(0, 0, w, h);
            vc_mesh->viewport = Eigen::Vector4f(0, 0, w, h);
            vc_lege->viewport = Eigen::Vector4f(
                                vc_mesh->viewport[0] - vc_mesh->viewport[2] * 3.0/8.0,
                                vc_mesh->viewport[1],
                                vc_mesh->viewport[2],
                                vc_mesh->viewport[3]);
            vc_hist->viewport = Eigen::Vector4f(
                                vc_mesh->viewport[0] + vc_mesh->viewport[2] * 3.0/8.0,
                                vc_mesh->viewport[1],
                                vc_mesh->viewport[2],
                                vc_mesh->viewport[3]);
            vc_axis->viewport = Eigen::Vector4f(
                vc_mesh->viewport[0]-vc_mesh->viewport[2]*0.375,
                vc_mesh->viewport[1]-vc_mesh->viewport[3]*0.375,
                vc_mesh->viewport[2],
                vc_mesh->viewport[3]);
            return false;
        };

        vd_mesh->set_visible(true,  vc_mesh->id);
        vd_mesh->set_visible(false, vc_lege->id);
        vd_mesh->set_visible(false, vc_axis->id);
        vd_mesh->set_visible(false, vc_hist->id);

        vd_mesh2->set_visible(true,  vc_mesh->id);
        vd_mesh2->set_visible(false, vc_lege->id);
        vd_mesh2->set_visible(false, vc_axis->id);
        vd_mesh2->set_visible(false, vc_hist->id);

        vd_axis->set_visible(true,  vc_axis->id);
        vd_axis->set_visible(false, vc_mesh->id);
        vd_axis->set_visible(false, vc_lege->id);
        vd_axis->set_visible(false, vc_hist->id);

        vd_lege->set_visible(true,  vc_lege->id);
        vd_lege->set_visible(false, vc_mesh->id);
        vd_lege->set_visible(false, vc_axis->id);
        vd_lege->set_visible(false, vc_hist->id);

        vd_hist->set_visible(false, vc_mesh->id);
        vd_hist->set_visible(false, vc_lege->id);
        vd_hist->set_visible(false, vc_axis->id);
        vd_hist->set_visible(true, vc_hist->id);

        drawLegend(*vd_lege);
        vd_lege->set_visible(false, vc_lege->id);

        ////////////////////////////////////////////////////////////////////////////////////////
        layersColorEigen.col(0) = Eigen::Vector4f(0.36,0.75,0.71,1.0);
        layersColorEigen.col(1) = Eigen::Vector4f(1.00,0.71,0.61,1.0);
        layersColorEigen.col(2) = Eigen::Vector4f(0.23,0.23,0.31,1.0);
        layersColorEigen.col(3) = Eigen::Vector4f(0.23,1.00,0.31,1.0);
        layersColorEigen.col(4) = Eigen::Vector4f(0.23,0.23,1.00,1.0);
        layersColorEigen.col(5) = Eigen::Vector4f(1.00,0.23,1.00,1.0);

        rank1TensorColors.col(0) = Eigen::Vector4f(1.0,0.0,0.0,1.0);
        rank1TensorColors.col(1) = Eigen::Vector4f(0.0,1.0,0.0,1.0);
        rank1TensorColors.col(2) = Eigen::Vector4f(0.0,0.0,1.0,1.0);
        rank1TensorColors.col(3) = Eigen::Vector4f(0.2,0.2,0.2,1.0);
        rank1TensorColors.col(4) = Eigen::Vector4f(0.8,0.8,0.8,1.0);
        rank1TensorColors.col(5) = Eigen::Vector4f(0.8,0.8,0.8,1.0);
        rank1TensorColors.col(6) = Eigen::Vector4f(0.8,0.8,0.8,1.0);
        rank1TensorColors.col(7) = Eigen::Vector4f(0.8,0.8,0.8,1.0);

        glfwSwapInterval(0);

        while (!glfwWindowShouldClose(viewer.window)&&!shutDown) {
            //            shutDown = glfwWindowShouldClose(viewer.window);
            // while (applyChanges) {}
        // if (!applyChanges) {
        if (!modelChanged) {

            vc_mesh->orthographic = showOrtho;
            vc_axis->orthographic = showOrtho;

            vd_mesh->show_lines = (vd_mesh->show_lines & ~vc_mesh->id) | (unsigned(showEdges) & vc_mesh->id);
            vd_mesh2->show_lines = vd_mesh->show_lines;
            vd_axis->set_visible(showBasis, vc_axis->id);
            vd_hist->set_visible(showMeshQ, vc_hist->id);
            vd_lege->set_visible(visibleData > 0, vc_lege->id);

            if (basisChanged) {
                basisChanged = false;
                drawAxis(*vd_axis, gP.grCoordType);
            }

            Eigen::SparseMatrix<Float> BC;
            Eigen::SparseMatrix<double> BC2;
            Eigen::VectorXi J,J2, boundaryTetIds;

            setNewMeshToViewer = setNewMeshToViewer || (renderNewData && meshVisState>0);
            if (setNewMeshToViewer) {
                setNewMeshToViewer = false;
                vd_mesh->clear();
                vd_mesh2->clear();
                if (mesh.pos.rows()>0) {
                    thrust::host_vector<Vector> tempPos(data.pos);
                    Eigen::Map<VectorArrayEigen> visPos(&tempPos[0][0], mesh.nver, 3);
                    if (meshVisState == 0) {
                        vd_mesh->set_mesh(visPos.cast<double>(), mesh.tri);
                        layerFaceColors = layersColorEigen(Eigen::all, mesh.layer(mesh.boundaryTetIds) - 1 ).cast<double>().transpose();
                    } else if (meshVisState == 1) {
                        igl::marching_tets(visPos.cast<double>().matrix(),
                                           mesh.tet.matrix(),
                                           visPos.col(sliceDir).cast<double>().matrix(),
                                           minPos[sliceDir] + sliceMag[sliceDir] * (maxPos[sliceDir] - minPos[sliceDir]),
                                           sliceVertex,
                                           sliceFace,
                                           J,
                                           BC);
                        vd_mesh->double_sided = true;
                        vd_mesh->set_mesh(sliceVertex, sliceFace);
                        layerFaceColors.resize(J.rows(), 4);
                        if (J.rows() > 0)
                            layerFaceColors = layersColorEigen(Eigen::all, mesh.layer(J) - 1 ).cast<double>().transpose();
                    } else if (meshVisState == 2 && !crinkleClip) {
                        igl::marching_tets(visPos.cast<double>().matrix(),
                                           mesh.tet.matrix(),
                                           visPos.col(sliceDir).cast<double>().matrix() * (inverseClip?(-1.0):(1.0)),
                                           (minPos[sliceDir] + sliceMag[sliceDir] * (maxPos[sliceDir] - minPos[sliceDir])) * (inverseClip?(-1.0):(1.0)),
                                           sliceVertex,
                                           sliceFace,
                                           J,
                                           BC);
                        vd_mesh->double_sided = false;
                        vd_mesh->set_mesh(sliceVertex, sliceFace);
                        layerFaceColors.resize(J.rows(), 4);
                        if (J.rows() > 0)
                            layerFaceColors = layersColorEigen(Eigen::all, mesh.layer(J) - 1 ).cast<double>().transpose();

                        // Plane: point P and normal N
                        Eigen::RowVector3d P(0, 0, 0);
                        Eigen::RowVector3d N(0, 0, 0);
                        N(sliceDir) = -(inverseClip?(-1.0):(1.0));
                        P(sliceDir) = minPos[sliceDir] + sliceMag[sliceDir] * (maxPos[sliceDir] - minPos[sliceDir]);
                        clip_mesh_against_plane_with_BC(visPos.cast<double>().matrix(), mesh.tri.matrix(), P, N, sliceVertex2, sliceFace2, J2, BC2);
                        vd_mesh2->double_sided = false;
                        vd_mesh2->set_mesh(sliceVertex2, sliceFace2);
                        layerFaceColors2.resize(J2.rows(), 4);
                        if (J2.rows() > 0)
                            layerFaceColors2 = layersColorEigen(Eigen::all, mesh.layer(mesh.boundaryTetIds(J2)) - 1 ).cast<double>().transpose();

                    } else if (meshVisState == 2 && crinkleClip) {
                        Eigen::VectorXi selectedTetIds;
                        VectorArrayEigen tetCenter = (visPos(mesh.tet.col(0),Eigen::all)
                                                      + visPos(mesh.tet.col(1),Eigen::all)
                                                      + visPos(mesh.tet.col(2),Eigen::all)
                                                      + visPos(mesh.tet.col(3),Eigen::all)) * 0.25;

                        if (inverseClip)
                            igl::find(tetCenter.col(sliceDir) > (minPos[sliceDir] + sliceMag[sliceDir] * (maxPos[sliceDir] - minPos[sliceDir])), selectedTetIds);
                        else
                            igl::find(tetCenter.col(sliceDir) < (minPos[sliceDir] + sliceMag[sliceDir] * (maxPos[sliceDir] - minPos[sliceDir])), selectedTetIds);
                        Eigen::MatrixXi tri;
                        Eigen::MatrixXi tet = mesh.tet(selectedTetIds,Eigen::all);
                        igl::boundary_facets(tet, tri);
                        vd_mesh->set_mesh(visPos.cast<double>(), tri);
                        boundaryTetIds = tetra_for_boundary_faces_v2t(mesh.tet, tri, mesh.nver);
                        if (tri.rows()>0)
                            layerFaceColors = layersColorEigen(Eigen::all, mesh.layer(boundaryTetIds) - 1 ).cast<double>().transpose();
                    }

                    if (firstTimeSetMeshToViewer) {
                        resetViewFlag = true;
                        firstTimeSetMeshToViewer = false;
                    }
                }
            }

            if (renderNewData) {
                renderNewData = false;
                if (!data.pos.empty()) {
                    ScalarArrayEigen tempData;
                    ScalarArrayEigen tempData2;
                    Float minVal(NAN), maxVal(NAN);
                    auto selectedDataTitle = std::get<0>(varNamesColor[visibleData]);
                    auto selectedData = std::get<1>(varNamesColor[visibleData]);
                    int selectedDataComp = std::get<2>(varNamesColor[visibleData]);

                    vd_mesh->clear_edges();
                    vd_mesh2->clear_edges();

                    if (std::any_of(showRank1Tensors, showRank1Tensors+rank1TensorsNames.size(), [](bool v){ return v; })) {
                        _LAUNCH_NVRTC(mesh.ntet, 128, compute_for_vis_nvrtc.kernel, {&dataPtr, &globalTime, &mesh.ntet});
                        cudaDeviceSynchronize();
                    }

                    if (meshVisState == 0) {
                        thrust::host_vector<Vector> tempPos(data.pos);
                        Eigen::Map<VectorArrayEigen> visPos(&tempPos[0][0], mesh.nver, 3);

                        vd_mesh->set_vertices(visPos.cast<double>());
                        igl::per_corner_normals(vd_mesh->V, vd_mesh->F, 70.0, Normals);
                        vd_mesh->set_normals(Normals);
                        vd_mesh->F_uv.resize(0,0);
                        tempData = device_to_vis_data_tet_mesh(selectedData, selectedDataComp, mesh.boundaryTetIds);

                        if (std::any_of(showRank1Tensors, showRank1Tensors+rank1TensorsNames.size(), [](bool v){ return v; })) {
                            triCenter = (visPos(mesh.tri.col(0), Eigen::all)
                                + visPos(mesh.tri.col(1), Eigen::all)
                                + visPos(mesh.tri.col(2), Eigen::all)).cast<double>() / 3.0;
                            for (int i=0; i<rank1TensorsNames.size(); i++)
                                if (showRank1Tensors[i]) {
                                    double sFacor = scaleRank1Tensors[i];
                                    TensorArrayEigen tempTensor = device_to_eigen_tensor(*std::get<1>(rank1TensorsNames[i]));
                                    VectorArrayEigen tempVector = computeMaxEigenvectors_mapped(tempTensor(mesh.boundaryTetIds,Eigen::all));
                                    Eigen::RowVector3d color(rank1TensorColors(0,i),rank1TensorColors(1,i),rank1TensorColors(2,i));
                                    vd_mesh->add_edges(triCenter-sFacor*tempVector.cast<double>(),triCenter+sFacor*tempVector.cast<double>(),color);
                                }
                        }
                    } else if (meshVisState == 1) {
                        if (std::any_of(showRank1Tensors, showRank1Tensors+rank1TensorsNames.size(), [](bool v){ return v; })) {
                            triCenter = (sliceVertex(sliceFace.col(0), Eigen::all)
                                + sliceVertex(sliceFace.col(1), Eigen::all)
                                + sliceVertex(sliceFace.col(2), Eigen::all)) / 3.0;
                            for (int i=0; i<rank1TensorsNames.size(); i++)
                                if (showRank1Tensors[i]) {
                                    double sFacor = scaleRank1Tensors[i];
                                    TensorArrayEigen tempTensor = device_to_eigen_tensor(*std::get<1>(rank1TensorsNames[i]));
                                    VectorArrayEigen tempVector = computeMaxEigenvectors_mapped(tempTensor(mesh.boundaryTetIds,Eigen::all));
                                    Eigen::RowVector3d color(rank1TensorColors(0,i),rank1TensorColors(1,i),rank1TensorColors(2,i));
                                    vd_mesh->add_edges(triCenter-sFacor*tempVector.cast<double>(),triCenter+sFacor*tempVector.cast<double>(),color);
                                }
                        }
                        tempData = device_to_vis_data_tet_section(selectedData, selectedDataComp, J, BC);
                    } else if (meshVisState == 2 && !crinkleClip) {
                        if (std::any_of(showRank1Tensors, showRank1Tensors+rank1TensorsNames.size(), [](bool v){ return v; })) {
                            triCenter = (sliceVertex(sliceFace.col(0), Eigen::all)
                                + sliceVertex(sliceFace.col(1), Eigen::all)
                                + sliceVertex(sliceFace.col(2), Eigen::all)) / 3.0;
                            triCenter2 = (sliceVertex2(sliceFace2.col(0), Eigen::all)
                                + sliceVertex2(sliceFace2.col(1), Eigen::all)
                                + sliceVertex2(sliceFace2.col(2), Eigen::all)) / 3.0;
                            for (int i=0; i<rank1TensorsNames.size(); i++)
                                if (showRank1Tensors[i]) {
                                    double sFacor = scaleRank1Tensors[i];
                                    TensorArrayEigen tempTensor = device_to_eigen_tensor(*std::get<1>(rank1TensorsNames[i]));
                                    VectorArrayEigen tempVector = computeMaxEigenvectors_mapped(tempTensor(J,Eigen::all));
                                    Eigen::RowVector3d color(rank1TensorColors(0,i),rank1TensorColors(1,i),rank1TensorColors(2,i));
                                    vd_mesh->add_edges(triCenter-sFacor*tempVector.cast<double>(),triCenter+sFacor*tempVector.cast<double>(),color);
                                    VectorArrayEigen tempVector2 = computeMaxEigenvectors_mapped(tempTensor(mesh.boundaryTetIds(J2),Eigen::all));
                                    vd_mesh->add_edges(triCenter2-sFacor*tempVector2.cast<double>(),triCenter2+sFacor*tempVector2.cast<double>(),color);
                                }
                        }

                        igl::per_corner_normals(vd_mesh->V, vd_mesh->F, 70.0, Normals);
                        vd_mesh->set_normals(Normals);
                        vd_mesh->F_uv.resize(0,0);
                        igl::per_corner_normals(vd_mesh2->V, vd_mesh2->F, 70.0, Normals);
                        vd_mesh2->set_normals(Normals);
                        vd_mesh2->F_uv.resize(0,0);
                        tempData = device_to_vis_data_tet_section(selectedData, selectedDataComp, J, BC);
                        tempData2 = device_to_vis_data_tet_section(selectedData, selectedDataComp, mesh.boundaryTetIds(J2), BC2.cast<Float>());
                    } else if (meshVisState == 2 && crinkleClip) {
                        tempData = device_to_vis_data_tet_mesh(selectedData, selectedDataComp, boundaryTetIds);
                    }

                    if (visibleData == 0) {
                        vd_mesh->show_texture = unsigned(0);
                        vd_mesh->set_colors(layerFaceColors);
                        if (vd_mesh2->V.rows()>0) {
                            vd_mesh2->show_texture = unsigned(0);
                            vd_mesh2->set_colors(layerFaceColors2);
                        }
                    } else {
                        if (meshVisState == 2 && tempData.rows() > 0 && tempData2.rows() > 0) {
                            int N1 = tempData.rows();
                            int N2 = tempData2.rows();
                            ScalarArrayEigen tempData3(N1+N2);
                            tempData3 << tempData, tempData2;
                            clampScalarArray(tempData3, minVal, maxVal, 0.01);
                            vd_mesh->set_data(tempData3.head(N1).cast<double>(), minVal, maxVal, igl::COLOR_MAP_TYPE_PARULA);
                            vd_mesh2->set_data(tempData3.tail(N2).cast<double>(), minVal, maxVal, igl::COLOR_MAP_TYPE_PARULA);
                        } else if (tempData.rows() > 0) {
                            clampScalarArray(tempData, minVal, maxVal, 0.01);
                            vd_mesh->set_data(tempData.cast<double>(), igl::COLOR_MAP_TYPE_PARULA);
                        }
                    }
                    vd_lege->set_labels(vd_lege->labels_positions,
                        std::vector<std::string>({
                            "time = "+std::to_string(globalTime),
                            selectedDataTitle,
                            std::to_string(minVal),
                            std::to_string(maxVal)}));
                    if (showMeshQ) {
                        _LAUNCH(mesh.ntet, 256, compute_mesh_quality) (dataPtr, mesh.ntet);
                        cudaDeviceSynchronize();
                        thrust::host_vector<Float> tetQual(data.tetQual);
                        drawHistogram(*vd_hist, ScalarArrayEigen(Eigen::Map<ScalarArrayEigen>(reinterpret_cast<Float *>(tetQual.data()), tetQual.size())), numBins);
                    }
                }

                if ((frames + 1) * sUpdate < simIter) {
                    // if (savePLY) {
                    //     char fname[16];
                    //     snprintf(fname, 16, "_%04d.ply", frames);
                    //     ScalarArrayEigen visR, visG, visB;
                    //     computeNodalColor(visR, visG, visB);
                    //     Eigen::MatrixXd nodalColor(mesh.boundaryNodeIds.size(), 3);
                    //     nodalColor.col(0) = visR(mesh.boundaryNodeIds).cast<double>();
                    //     nodalColor.col(1) = visG(mesh.boundaryNodeIds).cast<double>();
                    //     nodalColor.col(2) = visB(mesh.boundaryNodeIds).cast<double>();
                    //     std::string tempOutput = output;
                    //     tempOutput.replace(tempOutput.end()-4,tempOutput.end(),fname);
                    //     writePLY(tempOutput, vd_mesh->V(mesh.boundaryNodeIds, Eigen::all), nodalColor, mesh.tri_mapped.matrix());
                    //     fs::permissions(tempOutput, fs::perms::owner_all | fs::perms::group_all | fs::perms::others_all);
                    // }
                    if (saveVTK) {
                        char fname[16];
                        snprintf(fname, 16, "_%04d.vtk", frames);
                        std::string tempOutput = outputFileName;
                        tempOutput.replace(tempOutput.end()-4,tempOutput.end(),fname);
                        mesh.save_binary(tempOutput, data.tet, data.pos, dataToSave);
                        fs::permissions(tempOutput,fs::perms::owner_all | fs::perms::group_all | fs::perms::others_all);
                        std::cout<<"\t# "<<tempOutput<<" is saved." <<std::endl;
                    }
                    frames++;
                }
                cudaDeviceSynchronize();
            }

            if (saveMP4) {
                if (videoWriter == nullptr) {
                    glfwGetFramebufferSize(viewer.window, &gWWidth, &gWHeight);
                    gWWidth = (gWWidth / 2) * 2;
                    gWHeight = (gWHeight / 2) * 2;
                    videoWriter = new VideoWriter(("output_"+std::to_string(videoCounter++)+".mp4"), gWWidth, gWHeight, 30);
                    videoWriter->init();
                }
                // int width, height;
                // glfwGetFramebufferSize(viewer.window, &width, &height);
                // if ((R.rows()!=gWWidth) || (R.cols()!=gWHeight)) {
                //     R.resize(gWWidth,gWHeight);
                //     G.resize(gWWidth,gWHeight);
                //     B.resize(gWWidth,gWHeight);
                //     A.resize(gWWidth,gWHeight);
                // }
                char fname[16];
                // snprintf(fname, 16, "_%04d.png", frames);
                // vc_mesh->draw_buffer(*vd_mesh, false, R, G, B, A);
                // std::string tempOutput = outputFileName;
                // tempOutput.replace(tempOutput.end()-4,tempOutput.end(),fname);
                // igl::stb::write_image(tempOutput, R, G, B, A);
                // fs::permissions(tempOutput, fs::perms::owner_all | fs::perms::group_all | fs::perms::others_all);

                int fb_w = 0, fb_h = 0;
                glfwGetFramebufferSize(viewer.window, &fb_w, &fb_h);
                fb_w = (fb_w / 2) * 2;
                fb_h = (fb_h / 2) * 2;
                std::vector<uint8_t> buffer(fb_w * fb_h * 4);
                glReadPixels(0, 0, fb_w, fb_h, GL_RGBA, GL_UNSIGNED_BYTE, &buffer[0]);
                buffer.resize(gWWidth * gWHeight * 4);
                videoWriter->addFrame(buffer.data());
            } else if (videoWriter != nullptr) {
                videoWriter->close();
                delete videoWriter;
                videoWriter = nullptr;
            }

            if (resetViewFlag) {
                vc_mesh->align_camera_center(vd_mesh->V, vd_mesh->F);
                vc_mesh->camera_zoom = 1.0f;
                vc_mesh->camera_translation << 0, 0, 0;
                vc_mesh->trackball_angle = trackball_angle;
                resetViewFlag = false;
            }
        }

            viewer.launch_rendering(false);
        }
    };

    auto fem_iterate = [] {
        potEnergy = 0.001;

        auto start = std::chrono::steady_clock::now();
        auto end = std::chrono::steady_clock::now();
        while (!shutDown) {
            if (!run) {
                std::unique_lock<std::mutex> lck(mtx);
                runSim.wait(lck);
            } else {

                simulate();

                simDiverged = simDiverged || (!std::isfinite(minPos.norm1()) || !std::isfinite(maxPos.norm1())) || !std::isfinite(maxDt);
                enDiff[simIter % averageInterval] = thrust::reduce(data.potEnergy.begin(), data.potEnergy.end())
                / thrust::reduce(data.tetVol.begin(), data.tetVol.end());

                if (simIter>=maxIter || (kinEnergy<kinEnergyTol && simIter > averageInterval)) {
                    run = false;
                }

                if (simIter % averageInterval == 0) {
                    simDiverged = isSimDiverged();
                    if (!simDiverged) {
                        save_to_snapshot();
                    }

                    float avgSTD[2] = {INFINITY, INFINITY};
                    getAverageAndSTD(avgSTD, enDiff, averageInterval);
                    kinEnergy = avgSTD[1];
                }

                if (simDiverged) {
                    recover_from_snapshot();
                    _LAUNCH(mesh.ntet, 256, compute_volume) (dataPtr, mesh.ntet);
                    cudaDeviceSynchronize();
                    _LAUNCH_NVRTC(mesh.ntet, 32, compute_critical_timestep_nvrtc.kernel, {&dataPtr, &globalTime, &mesh.ntet});
                    cudaDeviceSynchronize();
                    maxDt = *thrust::min_element(data.dt.begin(), data.dt.end());

                    timeFactor = timeFactor * 0.75;
                    dt = maxDt * timeFactor;
                    get_domain_limits(data.pos);
                    _LAUNCH(mesh.nnbd, 128, read_boundary_nodes) (dataPtr, spacing, mesh.nnbd);
                    _LAUNCH(mesh.ntri, 128, read_boundary_faces) (dataPtr, spacing, mesh.ntri);
                    octreeSearch(data, minPos - Vector(spacing * 10.0f), maxPos + Vector(spacing * 10.0f));
                    potEnergy = 0.0;
                    kinEnergy = NAN;
                    for (int i=0; i<averageInterval; i++)
                        enDiff[i] = INFINITY;
                    renderNewData = true;
                    simDiverged = false;
                    std::cout << "\t# Rollback to time-point "<< oldGlobalTime
                        << " and continue with time-step factor "<<timeFactor
                        << " (*"<<maxDt<<")"
                    << std::endl;
                }

                simIter++;
                if (simIter%sUpdate == 0) {
                    renderNewData = true;
                    FPS = 1e3 * static_cast<double>(simIter - iterCounter) /
                          static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>
                                  (std::chrono::steady_clock::now() - start).count());
                    iterCounter = simIter;
                    cudaDeviceSynchronize();
                    start = std::chrono::steady_clock::now();
                }

                //if ((savePLY) && ((runMode==1) && (std::round(((growthProgress - mGrowthProgress)*100.0)>=(saveTime - 1e-5))) || shutDown))
                if ((savePLY || saveVTK) &&
                    ((runMode==1) && (std::sin(globalTime/saveTime * M_PI) * std::sin((globalTime-dt)/saveTime * M_PI)<=0.0)
                    || shutDown))
                    if (!isSimDiverged()) {
                        char tempStr[50];
                        if (savePLY) {
                            std::string tempOutput = outputFileName;
                            std::sprintf(tempStr,"_%04d.ply",frames++);
                            tempOutput.replace(tempOutput.end()-4,tempOutput.end(),tempStr);
                            thrust::host_vector<Vector> tempPos(data.pos);
                            Eigen::Map<VectorArrayEigen> visPos(&tempPos[0][0], mesh.nver, 3);
                            Eigen::MatrixXf visPosA(visPos.cast<float>());
                            writePLY(tempOutput,visPosA(mesh.boundaryNodeIds, Eigen::all).cast<double>() ,mesh.tri_mapped.matrix());
                            fs::permissions(tempOutput,fs::perms::owner_all | fs::perms::group_all | fs::perms::others_all);
                            std::cout<<"\t# "<<tempOutput<<" is saved." <<std::endl;
                        }
                        if (saveVTK) {
                            std::string tempOutput = outputFileName;
                            std::sprintf(tempStr,"_%04d.vtk",frames++);
                            tempOutput.replace(tempOutput.end()-4,tempOutput.end(),tempStr);
                            mesh.save_binary(tempOutput, data.tet, data.pos, dataToSave);
                            fs::permissions(tempOutput,fs::perms::owner_all | fs::perms::group_all | fs::perms::others_all);
                            std::cout<<"\t# "<<tempOutput<<" is saved." <<std::endl;
                        }
                    }
            }
        }
    };

    auto draw_menu = [] {
        bool lowTimeFactorFlag = false;
        bool maxIterFlag = false;

        // Setup SDL
        // (Some versions of SDL before <2.0.10 appears to have performance/stalling issues on a minority of Windows systems,
        // depending on whether SDL_INIT_GAMECONTROLLER is enabled or disabled.. updating to latest version of SDL is recommended!)
        if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER | SDL_INIT_GAMECONTROLLER) != 0) {
            printf("Error: %s\n", SDL_GetError());
            return -1;
        }

        // Decide GL+GLSL versions
#ifdef __APPLE__
        // GL 3.2 Core + GLSL 150
        const char* glsl_version = "#version 150";
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, SDL_GL_CONTEXT_FORWARD_COMPATIBLE_FLAG); // Always required on Mac
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 2);
#else
        // GL 3.0 + GLSL 130
        const char *glsl_version = "#version 150";
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, 0);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
#endif

        // Create window with graphics context
        SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
        SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
        SDL_GL_SetAttribute(SDL_GL_STENCIL_SIZE, 8);
        SDL_WindowFlags window_flags = (SDL_WindowFlags) (SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE |
                                                          SDL_WINDOW_ALLOW_HIGHDPI);
        SDL_Window *window = SDL_CreateWindow("", SDL_WINDOWPOS_CENTERED,SDL_WINDOWPOS_CENTERED,640, 720, window_flags);
        SDL_GLContext gl_context = SDL_GL_CreateContext(window);
        SDL_GL_MakeCurrent(window, gl_context);
        SDL_GL_SetSwapInterval(1); // Enable vsync

        bool err = gladLoadGLLoader((GLADloadproc) SDL_GL_GetProcAddress) == 0;;
        if (err) {
            fprintf(stderr, "Failed to initialize OpenGL loader!\n");
            return 1;
        }

        // Setup Dear ImGui context
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO &io = ImGui::GetIO();
        // (void) io;
        //io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
        //io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls
        //io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;
        // Setup Dear ImGui style
        ImGui::StyleColorsDark();
        //ImGui::StyleColorsClassic();

        // Setup Platform/Renderer backends
        ImGui_ImplSDL2_InitForOpenGL(window, gl_context);
        ImGui_ImplOpenGL3_Init(glsl_version);
        // Our state
        ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
        static const ImWchar ranges[] = {
            0x0020, 0x00FF,   // Basic Latin + Latin-1
            0x0370, 0x03FF,   // Greek
            0
        };
        io.Fonts->AddFontFromFileTTF("../share/fonts/LiberationMono-Regular.ttf",20,    nullptr, ranges);
        io.Fonts->Build();

        SDL_Event event;

        // Main loop
        while (!shutDown) {
            // Poll and handle events (inputs, window resize, etc.)
            // You can read the io.WantCaptureMouse, io.WantCaptureKeyboard flags to tell if dear imgui wants to use your inputs.
            // - When io.WantCaptureMouse is true, do not dispatch mouse input data to your main application.
            // - When io.WantCaptureKeyboard is true, do not dispatch keyboard input data to your main application.
            // Generally you may always pass all inputs to dear imgui, and hide them from your application based on those two flags.
            while (SDL_PollEvent(&event)) {
                ImGui_ImplSDL2_ProcessEvent(&event);
                if (event.type == SDL_QUIT)
                    shutDown = true;
                if (event.type == SDL_WINDOWEVENT && event.window.event == SDL_WINDOWEVENT_CLOSE &&
                    event.window.windowID == SDL_GetWindowID(window))
                    shutDown = true;
            }

            // Start the Dear ImGui frame
            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplSDL2_NewFrame();
            ImGui::NewFrame();

            ImGui::SetNextWindowPos(ImVec2(0,0), ImGuiCond_Always);
            ImGui::SetNextWindowSize(ImVec2(io.DisplaySize.x, io.DisplaySize.y), ImGuiCond_Always);

            ImGui::Begin("Control Panel", NULL, ImGuiWindowFlags_NoResize);
            ImGui::SetWindowFontScale(fontScale);
            float wW = ImGui::GetContentRegionAvail().x;
            float wP = ImGui::GetStyle().FramePadding.x;

            if (run)
                IMGUI_DISABLE_WIDGET

            if (ImGui::Button("Load project", ImVec2((wW - wP) * 0.5f, 0))) {
                std::string fname = igl::file_dialog_open();
                if (fname.length() != 0) {
                    modelChanged = true;
                    basisChanged = true;
                    firstTimeSetMeshToViewer = true;
                    applyChanges = true;
                    readSetting(fname);
                }
            }
            ImGui::SameLine();
            if (ImGui::Button("Load mesh file", ImVec2((wW - wP) * 0.5f, 0))) {
                inputGeom = Arbitrary;
                std::string newFilename = igl::file_dialog_open();
                if (newFilename != std::string("")) {
                    gP.grCoordType = NormalTangent;
                    basisChanged = true;
                    modelChanged = true;
                    firstTimeSetMeshToViewer = true;
                    inputMeshFileName = newFilename;
                    std::string fname_short = inputMeshFileName;
                    auto sep = inputMeshFileName.find_last_of("/\\");
                    if (sep != std::string::npos)
                        fname_short = fname_short.substr(sep + 1);
                    applyChanges = true;
                }
            }

            if (ImGui::CollapsingHeader("Generate mesh")) {
                ImGui::Separator();
                ImGui::Columns(2, "Geometry Columns", false);
                if (ImGui::RadioButton("Box", reinterpret_cast<int *>(&inputGeom), Box)) {
                    modelChanged = true;
                    inputGeom = Box;
                    gP.grCoordType = Cartesian;
                    basisChanged = true;
                }
                if (ImGui::RadioButton("Disk", reinterpret_cast<int *>(&inputGeom), Disk)) {
                    modelChanged = true;
                    inputGeom = Disk;
                    gP.grCoordType = CylindricalZ;
                    basisChanged = true;
                }
                if (ImGui::RadioButton("Tube", reinterpret_cast<int *>(&inputGeom), 3))  {
                    modelChanged = true;
                    inputGeom = Tube;
                    gP.grCoordType = CylindricalY;
                    basisChanged = true;
                }
                ImGui::NextColumn();
                if (ImGui::RadioButton("Cone", reinterpret_cast<int *>(&inputGeom), 4))  {
                    modelChanged = true;
                    inputGeom = Cone;
                    gP.grCoordType = ConeAdapted;
                    basisChanged = true;
                }
                if (ImGui::RadioButton("Sphere", reinterpret_cast<int *>(&inputGeom), 5)) {
                    modelChanged = true;
                    inputGeom = Sphere;
                    gP.grCoordType = Spherical;
                    basisChanged = true;
                }
                if (ImGui::RadioButton("Torus", reinterpret_cast<int *>(&inputGeom), 6)) {
                    modelChanged = true;
                    inputGeom = Torus;
                    gP.grCoordType = Toroidal;
                    basisChanged = true;
                }
                ImGui::Columns(1);


                if (inputGeom == Box) {
                    ImGui::Separator();
                    ImGui::BeginTable("tab0",3);
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text("Number of layers");
                    ImGui::TableNextColumn();
                    modelChanged = ImGui::InputInt("##nlayer", &boxDims.nLayers) || modelChanged;
                    boxDims.nLayers = std::max(1, std::min(boxDims.nLayers, MAX_NLAYERS));
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text("Length");
                    ImGui::TableNextColumn();
                    modelChanged = ImGui::InputFloat("##Length", &boxDims.L, 1.0f, 1.0f, "%.3f") || modelChanged;
                    boxDims.L = std::max(0.0f, boxDims.L);
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text("Width");
                    ImGui::TableNextColumn();
                    modelChanged = ImGui::InputFloat("##Width", &boxDims.W, 1.0f, 1.0f, "%.3f")  || modelChanged;
                    boxDims.W = std::max(0.0f, boxDims.W);
                    ImGui::TableNextColumn();
                    ImGui::Text("Mesh spacing");
                    ImGui::EndTable();

                    ImGui::BeginTable("tab1",3);
                    for (int i=0; i<boxDims.nLayers; i++) {
                        auto layerStr = std::to_string(i+1);
                        ImGui::TableNextRow();
                        ImGui::TableNextColumn();
                        ImGui::Text((std::string("Thickness ") + layerStr).c_str());
                        ImGui::SameLine();
                        colorChanged = ImGui::ColorEdit4((std::string("##c") + layerStr).c_str(), layersColorEigen.col(i).data(), ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel) || colorChanged;
                        ImGui::TableNextColumn();
                        modelChanged = ImGui::InputFloat((std::string("##r") + layerStr).c_str(), &boxDims.H[i], 1.0f, 1.0f, "%.3f")  || modelChanged;
                        boxDims.H[i] = std::max(0.0f, boxDims.H[i]);
                        ImGui::TableNextColumn();
                        modelChanged = ImGui::InputFloat((std::string("##dx") + layerStr).c_str(), &boxDims.spacing[i], 0.1f, 1.0f, "%.3f")  || modelChanged;
                        boxDims.spacing[i] = std::max(0.0f, boxDims.spacing[i]);
                    }
                    ImGui::EndTable();
                } else if (inputGeom == Disk) { // Disk
                    ImGui::Separator();
                    ImGui::BeginTable("tab0",3);
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text("Number of layers");
                    ImGui::TableNextColumn();
                    modelChanged = ImGui::InputInt("##nlayer", &diskDims.nLayers) || modelChanged;
                    diskDims.nLayers = std::max(1, std::min(diskDims.nLayers, MAX_NLAYERS));

                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text("R");
                    ImGui::TableNextColumn();
                    modelChanged = ImGui::InputFloat("##R", &diskDims.R, 1.0f, 1.0f, "%.3f") || modelChanged;
                    diskDims.R = std::max(0.0f, diskDims.R);
                    ImGui::TableNextColumn();
                    ImGui::Text("Mesh spacing");
                    ImGui::EndTable();
                    ImGui::BeginTable("tab1",3);
                    for (int i = 0; i<diskDims.nLayers; i++) {
                        auto layerStr = std::to_string(i+1);
                        ImGui::TableNextRow();
                        ImGui::TableNextColumn();
                        ImGui::Text((std::string("Thickness ") + layerStr).c_str());
                        ImGui::SameLine();
                        colorChanged = ImGui::ColorEdit4((std::string("##c") + layerStr).c_str(), layersColorEigen.col(i).data(), ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel) || colorChanged;
                        ImGui::TableNextColumn();
                        modelChanged = ImGui::InputFloat((std::string("##r") + layerStr).c_str(), &diskDims.H[i], 1.0f, 1.0f, "%.3f")  || modelChanged;
                        diskDims.H[i] = std::max(0.0f, diskDims.H[i]);
                        ImGui::TableNextColumn();
                        modelChanged = ImGui::InputFloat((std::string("##dx") + layerStr).c_str(), &diskDims.spacing[i], 0.1f, 1.0f, "%.3f")  || modelChanged;
                        diskDims.spacing[i] = std::max(0.0f, diskDims.spacing[i]);
                    }
                    ImGui::EndTable();
                } else if (inputGeom == Tube) {
                    ImGui::Separator();
                    ImGui::BeginTable("tab0",3);
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text("Number of layers");
                    ImGui::TableNextColumn();
                    modelChanged = ImGui::InputInt("##nlayer", &tubeDims.nLayers) || modelChanged;
                    tubeDims.nLayers = std::max(1, std::min(tubeDims.nLayers, MAX_NLAYERS));
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text("Length");
                    ImGui::TableNextColumn();
                    modelChanged = ImGui::InputFloat("##Length", &tubeDims.L, 1.0f, 1.0f, "%.3f") || modelChanged;
                    tubeDims.L = std::max(0.0f, tubeDims.L);
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text("Radius of the hole");
                    ImGui::TableNextColumn();
                    modelChanged = ImGui::InputFloat("##radius", &tubeDims.Ri, 1.0f, 1.0f, "%.3f")  || modelChanged;
                    tubeDims.Ri = std::max(0.0f, tubeDims.Ri);
                    ImGui::TableNextColumn();
                    ImGui::Text("Mesh spacing");
                    ImGui::EndTable();
                    ImGui::BeginTable("tab1",3);
                    for (int i = 0; i<tubeDims.nLayers; i++) {
                        auto layerStr = std::to_string(i+1);
                        ImGui::TableNextRow();
                        ImGui::TableNextColumn();
                        ImGui::Text((std::string("Thickness ") + layerStr).c_str());
                        ImGui::SameLine();
                        colorChanged = ImGui::ColorEdit4((std::string("##c") + layerStr).c_str(), layersColorEigen.col(i).data(), ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel) || colorChanged;
                        ImGui::TableNextColumn();
                        modelChanged = ImGui::InputFloat((std::string("##r") + layerStr).c_str(), &tubeDims.H[i], 1.0f, 1.0f, "%.3f")  || modelChanged;
                        tubeDims.H[i] = std::max(0.0f, tubeDims.H[i]);
                        ImGui::TableNextColumn();
                        modelChanged = ImGui::InputFloat((std::string("##dx") + layerStr).c_str(), &tubeDims.spacing[i], 0.1f, 1.0f, "%.3f")  || modelChanged;
                        tubeDims.spacing[i] = std::max(0.0f, tubeDims.spacing[i]);
                    }
                    ImGui::EndTable();
                } else if (inputGeom == Cone) {
                    ImGui::Separator();
                    ImGui::BeginTable("tab0",3);
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text("Number of layers");
                    ImGui::TableNextColumn();
                    modelChanged = ImGui::InputInt("##nlayer", &coneDims.nLayers) || modelChanged;
                    coneDims.nLayers = std::max(1, std::min(coneDims.nLayers, MAX_NLAYERS));
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text("Apex angle (°)");
                    ImGui::TableNextColumn();
                    modelChanged = ImGui::InputFloat("##apexAngle", &coneDims.apexAng, 1.0f, 1.0f, "%.3f")  || modelChanged;
                    coneDims.apexAng = std::max(0.0f, std::min(180.0f, coneDims.apexAng));
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text("Length");
                    ImGui::TableNextColumn();
                    modelChanged = ImGui::InputFloat("##Length", &coneDims.L, 1.0f, 1.0f, "%.3f") || modelChanged;
                    coneDims.L = std::max(0.0f, coneDims.L);
                    if (modelChanged)
                        coneDims.L = min(coneDims.L,
                            (std::accumulate(coneDims.H, coneDims.H + coneDims.nLayers, 0.0)/std::cos(coneDims.apexAng*0.5*M_PI/180.0) + coneDims.Ri)/std::tan(coneDims.apexAng*0.5*M_PI/180.0) * 0.99);
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text("Radius of the hole");
                    ImGui::TableNextColumn();
                    modelChanged = ImGui::InputFloat("##baseRadius", &coneDims.Ri, 1.0f, 1.0f, "%.3f")  || modelChanged;
                    coneDims.Ri = std::max(0.0f, coneDims.Ri);
                    ImGui::TableNextColumn();
                    ImGui::Text("Mesh spacing");
                    ImGui::EndTable();
                    ImGui::BeginTable("tab1",3);
                    for (int i = 0; i<coneDims.nLayers; i++) {
                        auto layerStr = std::to_string(i+1);
                        ImGui::TableNextRow();
                        ImGui::TableNextColumn();
                        ImGui::Text((std::string("Thickness ") + layerStr).c_str());
                        ImGui::SameLine();
                        colorChanged = ImGui::ColorEdit4((std::string("##c") + layerStr).c_str(), layersColorEigen.col(i).data(), ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel) || colorChanged;
                        ImGui::TableNextColumn();
                        modelChanged = ImGui::InputFloat((std::string("##r") + layerStr).c_str(), &coneDims.H[i], 1.0f, 1.0f, "%.3f")  || modelChanged;
                        coneDims.H[i] = std::max(0.0f, coneDims.H[i]);
                        ImGui::TableNextColumn();
                        modelChanged = ImGui::InputFloat((std::string("##dx") + layerStr).c_str(), &coneDims.spacing[i], 0.1f, 1.0f, "%.3f")  || modelChanged;
                        coneDims.spacing[i] = std::max(0.0f, coneDims.spacing[i]);
                    }
                    ImGui::EndTable();
                } else if (inputGeom == Sphere) {
                    ImGui::Separator();
                    ImGui::BeginTable("tab0",3);
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text("Number of layers");
                    ImGui::TableNextColumn();
                    modelChanged = ImGui::InputInt("##nlayer", &sphereDims.nLayers) || modelChanged;
                    sphereDims.nLayers = std::max(1, std::min(sphereDims.nLayers, MAX_NLAYERS));
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text("Radius of the hole");
                    ImGui::TableNextColumn();
                    modelChanged = ImGui::InputFloat("##radius", &sphereDims.Ri, 1.0f, 1.0f, "%.3f")  || modelChanged;
                    sphereDims.Ri = std::max(0.0f, sphereDims.Ri);
                    ImGui::TableNextColumn();
                    ImGui::Text("Mesh spacing");
                    ImGui::EndTable();
                    ImGui::BeginTable("tab1",3);
                    for (int i=0; i < sphereDims.nLayers; i++) {
                        auto layerStr = std::to_string(i+1);
                        ImGui::TableNextRow();
                        ImGui::TableNextColumn();
                        ImGui::Text((std::string("Thickness ") + layerStr).c_str());
                        ImGui::SameLine();
                        colorChanged = ImGui::ColorEdit4((std::string("##c") + layerStr).c_str(), layersColorEigen.col(i).data(), ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel) || colorChanged;
                        ImGui::TableNextColumn();
                        modelChanged = ImGui::InputFloat((std::string("##r") + layerStr).c_str(), &sphereDims.H[i], 1.0f, 1.0f, "%.3f")  || modelChanged;
                        sphereDims.H[i] = std::max(0.0f, sphereDims.H[i]);
                        ImGui::TableNextColumn();
                        modelChanged = ImGui::InputFloat((std::string("##dx") + layerStr).c_str(), &sphereDims.spacing[i], 0.1f, 1.0f, "%.3f")  || modelChanged;
                        sphereDims.spacing[i] = std::max(0.0f, sphereDims.spacing[i]);
                    }
                    ImGui::EndTable();
                } else if (inputGeom == Torus) {
                    ImGui::Separator();
                    ImGui::BeginTable("tab0",3);
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text("Number of layers");
                    ImGui::TableNextColumn();
                    modelChanged = ImGui::InputInt("##nlayer", &torusDims.nLayers) || modelChanged;
                    torusDims.nLayers = std::max(1, std::min(torusDims.nLayers, MAX_NLAYERS));
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text("Torus radius");
                    ImGui::TableNextColumn();
                    modelChanged = ImGui::InputFloat("##RTorus", &torusDims.R, 1.0f, 1.0f, "%.3f")  || modelChanged;
                    torusDims.R = std::max(0.0f, torusDims.R);
                    if (modelChanged)
                        torusDims.R = max(torusDims.R, std::accumulate(torusDims.H, torusDims.H + torusDims.nLayers, torusDims.Ri) + torusDims.spacing[torusDims.nLayers-1] * 0.5);
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text("Radius of the hole");
                    ImGui::TableNextColumn();
                    modelChanged = ImGui::InputFloat("##radius", &torusDims.Ri, 1.0f, 1.0f, "%.3f")  || modelChanged;
                    torusDims.Ri = std::max(0.0f, torusDims.Ri);
                    ImGui::TableNextColumn();
                    ImGui::Text("Mesh spacing");
                    ImGui::EndTable();
                    ImGui::BeginTable("tab1",3);
                    for (int i=0; i < torusDims.nLayers; i++) {
                        auto layerStr = std::to_string(i+1);
                        ImGui::TableNextRow();
                        ImGui::TableNextColumn();
                        ImGui::Text((std::string("Thickness ") + layerStr).c_str());
                        ImGui::SameLine();
                        colorChanged = ImGui::ColorEdit4((std::string("##c") + layerStr).c_str(), layersColorEigen.col(i).data(), ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel) || colorChanged;
                        ImGui::TableNextColumn();
                        modelChanged = ImGui::InputFloat((std::string("##r") + layerStr).c_str(), &torusDims.H[i], 1.0f, 1.0f, "%.3f")  || modelChanged;
                        torusDims.H[i] = std::max(0.0f, torusDims.H[i]);
                        ImGui::TableNextColumn();
                        modelChanged = ImGui::InputFloat((std::string("##dx") + layerStr).c_str(), &torusDims.spacing[i], 0.1f, 1.0f, "%.3f")  || modelChanged;
                        torusDims.spacing[i] = std::max(0.0f, torusDims.spacing[i]);
                    }
                    ImGui::EndTable();
                }
                ImGui::Separator();
                if (ImGui::TreeNode("Presets")) {
                    std::vector<std::tuple<std::string, std::string, std::string>> presets = {
                        {"1. Elastic box with tangential growth",         "../examples/box_growth.txt",                   "Elastic box with tangential growth"},
                        {"2. Plastic box with active tensile filaments",  "../examples/box_tensile_force_spine.txt",      "Plastic box with active tensile filaments (actin)\n   -> Spine"},
                        {"3. Elastic disk with radial growth",            "../examples/disk_growth_to_cone.txt",          "Elastic disk with radial growth -> Cone"},
                        {"4. Elastic tube with circumferential growth",   "../examples/cylinder_theta_growth.txt",        "Elastic tube with circumferential growth"},
                        {"5. Elastic tube with growth & collagen fibers", "../examples/cylinder_croc.txt",                "Elastic tube with isotropic growth & collagen fibers\n   -> Crocodile pattern"},
                        {"6. Elastic cone with circumferential growth",   "../examples/cone_theta_growth.txt",            "Elastic cone with circumferential growth"},
                        {"7. Elastic sphere with tangential growth",      "../examples/sphere_growth.txt",                "Three-layered elastic sphere with rigid middle layer and\n   tangential growth of inner & outer layers."},
                        {"8. Two-layered elastic sphere growth 1",        "../examples/sphere_invagination.txt",          "Two-layered elastic sphere with orthogonal growth \n   directions -> pseudo-torus"},
                        {"9. Two-layered elastic sphere growth 2",        "../examples/sphere_invagination_2.txt",        "Two-layered elastic sphere with inner & outer growth"},
                        {"10. Elastic sphere with periodic growth 1",     "../examples/sphere_invagination_hexagon.txt",  "Elastic sphere with periodic growth -> Hexagonal fruit"},
                        {"11. Elastic sphere with periodic growth 2",     "../examples/sphere_croissant.txt",             "Elastic sphere with periodic growth -> Croissant"},
                        {"12. Elastic torus",                             "../examples/torus.txt",                        "Elastic torus"},
                    };
                    static int item_selected_idx = -1; // Here we store our selected data as an index.

                    static bool item_highlight = false;
                    int item_highlighted_idx = -1; // Here we store our highlighted data as an index.
                    if (ImGui::BeginListBox("##listbox 1", ImVec2(-1, 0))) {
                        for (int n = 0; n < presets.size(); n++) {
                            const bool is_selected = (item_selected_idx == n);
                            if (ImGui::Selectable(std::get<0>(presets[n]).c_str(), is_selected)) {
                                item_selected_idx = n;
                                modelChanged = true;
                                basisChanged = true;
                                firstTimeSetMeshToViewer = true;
                                readSetting(std::get<1>(presets[n]).c_str());
                            }
                            if (ImGui::IsItemHovered()) {
                                ImGui::SetTooltip(std::get<2>(presets[n]).c_str());
                                item_highlighted_idx = n;
                            }

                            // Set the initial focus when opening the combo (scrolling + keyboard navigation focus)
                            if (is_selected)
                                ImGui::SetItemDefaultFocus();
                        }
                        ImGui::EndListBox();
                    }
                    ImGui::TreePop();
                }


                basisChanged = basisChanged || modelChanged;

                if (!modelChanged)
                    IMGUI_DISABLE_WIDGET

                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(ImColor(0, 255, 0)));
                applyChanges = ImGui::Button("Apply Changes", ImVec2(-1, 0)) || applyChanges;
                ImGui::PopStyleColor();
                if (!modelChanged) {
                    ImGui::PopItemFlag();
                    ImGui::PopStyleVar();
                }
                if (applyChanges) {
                    ImGui::ProgressBar(1.0);   // empty overlay
                    // Draw left‑aligned text on top of the bar
                    ImGui::SameLine();
                    ImGui::SetCursorPosX(ImGui::GetCursorPosX() - wW); // move cursor back
                    ImGui::TextUnformatted("Generating mesh ... please wait");
                    compileFlag = true;
                }
            }

            if (inputGeom == Box) {
                nLayers = boxDims.nLayers;
            } else if (inputGeom == Disk) {
                nLayers = diskDims.nLayers;
            } else if (inputGeom == Tube) {
                nLayers = tubeDims.nLayers;
            } else if (inputGeom == Cone) {
                nLayers = coneDims.nLayers;
                gP.apex = coneDims.apexAng*M_PI/180.0;
                gP.rRefMin = coneDims.Ri;
                gP.rRefMax = std::accumulate(coneDims.H, coneDims.H + coneDims.nLayers, 0.0)/std::cos(coneDims.apexAng*0.5*M_PI/180.0) + coneDims.Ri;
            } else if (inputGeom == Sphere) {
                nLayers = sphereDims.nLayers;
                // gP.rRefMin = sphereDims.Ri;
                // gP.rRefMax = std::accumulate(sphereDims.H, sphereDims.H+nLayers, sphereDims.Ri);
            } else if (inputGeom == Torus) {
                nLayers = torusDims.nLayers;
                gP.RTorus = torusDims.R;
            }

            if (run) {
                ImGui::PopItemFlag();
                ImGui::PopStyleVar();
            }

            if (modelChanged)
                IMGUI_DISABLE_WIDGET


            if (ImGui::CollapsingHeader("Coordinate basis")) {
                ImGui::Columns(2, "Basis Columns", false);

                basisChanged = ImGui::RadioButton("Normal/tangent", reinterpret_cast<int *>(&gP.grCoordType), CoordinateSystem::NormalTangent) || basisChanged;
                basisChanged = ImGui::RadioButton("Cartesian", reinterpret_cast<int *>(&gP.grCoordType), CoordinateSystem::Cartesian) || basisChanged;

                float tmpfloat1 = float(gP.rRefMax);
                float tmpfloat2[2] = {float(gP.rRefMin), float(gP.rRefMax)};
                float tmpfloat3[3] = {float(gP.RTorus), float(gP.rRefMin), float(gP.rRefMax)};

                basisChanged = ImGui::RadioButton("Cylindrical Z", reinterpret_cast<int *>(&gP.grCoordType), CoordinateSystem::CylindricalZ) || basisChanged;
                if (gP.grCoordType == CylindricalZ) {
                    ImGui::SameLine();
                    basisChanged = ImGui::InputFloat("Max r", &tmpfloat1) || basisChanged;
                    gP.rRefMax = tmpfloat1;
                }

                basisChanged = ImGui::RadioButton("Cylindrical Y", reinterpret_cast<int *>(&gP.grCoordType), CoordinateSystem::CylindricalY) || basisChanged;

                if (gP.grCoordType == CylindricalY) {
                    ImGui::SameLine();
                    basisChanged = ImGui::InputFloat2("Min/Max r", tmpfloat2) || basisChanged;
                    gP.rRefMin = tmpfloat2[0]; gP.rRefMax = tmpfloat2[1];
                }

                ImGui::NextColumn();

                basisChanged = ImGui::RadioButton("Cone adapted", reinterpret_cast<int *>(&gP.grCoordType), CoordinateSystem::ConeAdapted) || basisChanged;
                if (gP.grCoordType == ConeAdapted) {
                }

                basisChanged = ImGui::RadioButton("Spherical", reinterpret_cast<int *>(&gP.grCoordType), CoordinateSystem::Spherical) || basisChanged;
                if (gP.grCoordType == Spherical) {
                    ImGui::SameLine();
                    basisChanged = ImGui::InputFloat2("Min/Max r", tmpfloat2) || basisChanged;
                    gP.rRefMin = tmpfloat2[0]; gP.rRefMax = tmpfloat2[1];
                }

                basisChanged = ImGui::RadioButton("Toroidal", reinterpret_cast<int *>(&gP.grCoordType), CoordinateSystem::Toroidal) || basisChanged;
                if (gP.grCoordType == Toroidal) {
                    ImGui::SameLine();
                    basisChanged = ImGui::InputFloat3("R, Min/Max r", tmpfloat3) || basisChanged;
                    gP.RTorus = tmpfloat3[0]; gP.rRefMin = tmpfloat3[1]; gP.rRefMax = tmpfloat3[2];
                }

                paramChanged = basisChanged || paramChanged;
                ImGui::Columns(1);
            }

            selectedLayer = min(selectedLayer,nLayers);
//            if (ImGui::CollapsingHeader("Select layer")) {
            if (ImGui::CollapsingHeader("Select layer",ImGuiTreeNodeFlags_Leaf)) {

                for (int i = 0; i < nLayers; i++) {
                    ImGui::RadioButton(("Layer " + std::to_string(i)).c_str(), &selectedLayer, i);
                    ImGui::SameLine();
                    colorChanged = ImGui::ColorEdit4((" ####LC##" + std::to_string(i)).c_str(), layersColorEigen.col(i).data(),
                             ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel) || colorChanged;
                    ImGui::SameLine();
                    ImGui::Text("  ");
                    if (((i+1)%3!=0) && i!=(nLayers-1))
                        ImGui::SameLine();
                }
            }

            defaultParamFlag = ImGui::Button("Default parameters", ImVec2(-1, 0)) || defaultParamFlag;

            paramChanged = ImGui::Checkbox("Rigid", &gP.isRigidLayer[selectedLayer]) || paramChanged;

            if (gP.isRigidLayer[selectedLayer])
                IMGUI_DISABLE_WIDGET

            if (ImGui::CollapsingHeader("Isotropic property functions")) {
                ImGui::SameLine();
                ImGui::Text("%s", (" of layer " + std::to_string(selectedLayer)).c_str());

                ImGui::BeginTable("##table_Isotropic", 2);
                ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthFixed);
                ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);

                ImGuiParamReadRow("Young's modulus ", gP2.E[selectedLayer], gP.useMeshDef_E, mesh.E.rows()==mesh.ntet, paramChanged);
                ImGui::Separator();
                ImGuiParamReadRow("Poisson's ratio ", gP2.nu[selectedLayer], gP.useMeshDef_nu, mesh.nu.rows()==mesh.ntet, paramChanged);
                ImGui::Separator();
                ImGuiParamReadRow("Viscosity ", gP2.visc[selectedLayer], gP.useMeshDef_viscosity, mesh.visc.rows()==mesh.ntet, paramChanged);
                ImGui::Separator();
                ImGuiParamReadRow("Plasticity coeff. ", gP2.plasticity[selectedLayer], gP.useMeshDef_plasticity, mesh.plasticity.rows()==mesh.ntet, paramChanged);
                if (ImGui::IsItemHovered())
                    ImGui::SetTooltip("Ranges from 0 (elastic) to 1 (plastic)");
                ImGui::EndTable();
            }
            if (ImGui::CollapsingHeader("Fibers & Tensile force functions")) {
                ImGui::SameLine();
                ImGui::Text("%s", (" of layer " + std::to_string(selectedLayer)).c_str());

                ImGui::BeginTable("##table_anisotropy", 2);
                ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthFixed);
                ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);


                // ImGui::BeginTable("##num_fibers",2);
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Number of fiber groups"); ImGui::SameLine();
                ImGui::TableNextColumn();
                paramChanged = ImGui::InputInt("##nFibers", &nFibers) || paramChanged;
                nFibers = min(max(nFibers,0),4);
                // ImGui::EndTable();

                if (nFibers>0) {
                    ImGuiParamReadRow("Fiber k1", gP2.k1[selectedLayer], gP.useMeshDef_k1, mesh.k1.rows()==mesh.ntet, paramChanged);
                    ImGuiParamReadRow("Fiber k2", gP2.k2[selectedLayer], gP.useMeshDef_k2, mesh.k2.rows()==mesh.ntet, paramChanged);
                    for (int i = 0; i < nFibers; i++) {
                        if (i==0)
                            ImGuiParamReadRow("Fiber 1", gP2.fiber1_Ref[selectedLayer], gP.useMeshDef_fiber1_Ref, mesh.fiberTetra1.rows()==mesh.ntet, paramChanged);
                        if (i==1)
                            ImGuiParamReadRow("Fiber 2", gP2.fiber2_Ref[selectedLayer], gP.useMeshDef_fiber2_Ref, mesh.fiberTetra2.rows()==mesh.ntet, paramChanged);
                        if (i==2)
                            ImGuiParamReadRow("Fiber 3", gP2.fiber3_Ref[selectedLayer], gP.useMeshDef_fiber3_Ref, mesh.fiberTetra3.rows()==mesh.ntet, paramChanged);
                        if (i==3)
                            ImGuiParamReadRow("Fiber 4", gP2.fiber4_Ref[selectedLayer], gP.useMeshDef_fiber4_Ref, mesh.fiberTetra4.rows()==mesh.ntet, paramChanged);
                    }
                }
                ImGui::Separator();
                ImGuiParamReadRow("Active tensile force", gP2.actin_Ref[selectedLayer], gP.useMeshDef_actin_Ref, mesh.actinTetra.rows()==mesh.ntet, paramChanged);
                ImGui::EndTable();
            }

            std::string dir1 = "X";
            std::string dir2 = "Y";
            std::string dir3 = "Z";
            if        (gP.grCoordType==CoordinateSystem::NormalTangent) { // NT
                dir1 = "Normal";
                dir2 = "Tangent";
                dir3 = "Tangent";
            } else if (gP.grCoordType==CoordinateSystem::CylindricalZ) { // Cylinder
                dir1 = "Radial (R)";
                dir2 = "Angular (θ)";
                dir3 = "Longitudinal (Z)";
            } else if (gP.grCoordType==CoordinateSystem::CylindricalY) { // Cylinder
                dir1 = "Radial (R)";
                dir2 = "Longitudinal (Y)";
                dir3 = "Angular (θ)";
            } else if (gP.grCoordType==CoordinateSystem::ConeAdapted) { // Cylinder
                dir1 = "Surface normal (N)";
                dir2 = "Slant direction (s)";
                dir3 = "Angular (Θ)";
            } else if (gP.grCoordType==CoordinateSystem::Spherical) { // Sphere
                dir1 = "Radial (R)";
                dir2 = "Polar (Θ)";
                dir3 = "Azimuthal (Φ)";
            } else if (gP.grCoordType==CoordinateSystem::Toroidal) { // Torus
                dir1 = "Radial (R)";
                dir2 = "Poloidal (Θ)";
                dir3 = "Toroidal (Φ)";
            }

            if (ImGui::CollapsingHeader("Growth rate functions")) {
                ImGui::SameLine();
                ImGui::Text("%s", (" of layer " + std::to_string(selectedLayer)).c_str());

                ImGui::BeginTable("##growth_function_table", 2);
                ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthFixed);
                ImGui::TableSetupColumn("Value", ImGuiTableColumnFlags_WidthStretch);

                ImGuiParamReadRow(dir1, gP2.grRate1_Ref[selectedLayer], gP.useMeshDef_grRate1_Ref, mesh.grRate1.rows()==mesh.ntet, paramChanged);
                ImGuiParamReadRow(dir2, gP2.grRate2_Ref[selectedLayer], gP.useMeshDef_grRate2_Ref, mesh.grRate2.rows()==mesh.ntet, paramChanged);
                if (gP.grCoordType!=0)
                    ImGuiParamReadRow(dir3, gP2.grRate3_Ref[selectedLayer], gP.useMeshDef_grRate3_Ref, mesh.grRate3.rows()==mesh.ntet, paramChanged);
                else
                    gP2.grRate3_Ref[selectedLayer] = gP2.grRate2_Ref[selectedLayer];
                ImGui::EndTable();
            }

            if (gP.isRigidLayer[selectedLayer])
                IMGUI_ENABLE_WIDGET

            if (ImGui::CollapsingHeader("Boundary conditions")) {
                ImGui::BeginTable("table2", 4);
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::TableNextColumn();
                ImGui::Text("Free");
                ImGui::TableNextColumn();
                ImGui::Text("Fixed");
                ImGui::TableNextColumn();
                ImGui::Text("Tangent slide");
                if (gP.grCoordType == CoordinateSystem::Spherical || gP.grCoordType == CoordinateSystem::Toroidal) {
                    ImGui::TableNextRow(); ImGui::TableNextColumn();
                    ImGui::Text("Inner surface"); ImGui::TableNextColumn();
                    bcChanged = ImGui::RadioButton("##xm free",    &gP.bcTypeMinAxis0, 0) || bcChanged; ImGui::TableNextColumn();
                    bcChanged = ImGui::RadioButton("##xm fixed",   &gP.bcTypeMinAxis0, 1) || bcChanged; ImGui::TableNextColumn();
                    bcChanged = ImGui::RadioButton("##xm tangent", &gP.bcTypeMinAxis0, 2) || bcChanged;

                    ImGui::TableNextRow(); ImGui::TableNextColumn();
                    ImGui::Text("Outer surface"); ImGui::TableNextColumn();
                    bcChanged = ImGui::RadioButton("##xp free",    &gP.bcTypeMaxAxis0, 0) || bcChanged; ImGui::TableNextColumn();
                    bcChanged = ImGui::RadioButton("##xp fixed",   &gP.bcTypeMaxAxis0, 1) || bcChanged; ImGui::TableNextColumn();
                    bcChanged = ImGui::RadioButton("##xp tangent", &gP.bcTypeMaxAxis0, 2) || bcChanged;
                } else if (gP.grCoordType == CoordinateSystem::CylindricalZ) {
                    ImGui::TableNextRow(); ImGui::TableNextColumn();
                    ImGui::Text("Outer surface"); ImGui::TableNextColumn();
                    bcChanged = ImGui::RadioButton("##xp free",    &gP.bcTypeMaxAxis0, 0) || bcChanged; ImGui::TableNextColumn();
                    bcChanged = ImGui::RadioButton("##xp fixed",   &gP.bcTypeMaxAxis0, 1) || bcChanged; ImGui::TableNextColumn();
                    bcChanged = ImGui::RadioButton("##xp tangent", &gP.bcTypeMaxAxis0, 2) || bcChanged;

                    ImGui::TableNextRow(); ImGui::TableNextColumn();
                    ImGui::Text("Min Z plane"); ImGui::TableNextColumn();
                    bcChanged = ImGui::RadioButton("##zm free",    &gP.bcTypeMinAxis1, 0) || bcChanged; ImGui::TableNextColumn();
                    bcChanged = ImGui::RadioButton("##zm fixed",   &gP.bcTypeMinAxis1, 1) || bcChanged; ImGui::TableNextColumn();
                    bcChanged = ImGui::RadioButton("##zm tangent", &gP.bcTypeMinAxis1, 2) || bcChanged;

                    ImGui::TableNextRow(); ImGui::TableNextColumn();
                    ImGui::Text("Max Z plane"); ImGui::TableNextColumn();
                    bcChanged = ImGui::RadioButton("##zp free",    &gP.bcTypeMaxAxis1, 0) || bcChanged; ImGui::TableNextColumn();
                    bcChanged = ImGui::RadioButton("##zp fixed",   &gP.bcTypeMaxAxis1, 1) || bcChanged; ImGui::TableNextColumn();
                    bcChanged = ImGui::RadioButton("##zp tangent", &gP.bcTypeMaxAxis1, 2) || bcChanged;

                } else if (gP.grCoordType == CoordinateSystem::CylindricalY  || gP.grCoordType == CoordinateSystem::ConeAdapted) {
                    ImGui::TableNextRow(); ImGui::TableNextColumn();
                    ImGui::Text("Inner surface"); ImGui::TableNextColumn();
                    bcChanged = ImGui::RadioButton("##xm free",    &gP.bcTypeMinAxis0, 0) || bcChanged; ImGui::TableNextColumn();
                    bcChanged = ImGui::RadioButton("##xm fixed",   &gP.bcTypeMinAxis0, 1) || bcChanged; ImGui::TableNextColumn();
                    bcChanged = ImGui::RadioButton("##xm tangent", &gP.bcTypeMinAxis0, 2) || bcChanged;

                    ImGui::TableNextRow(); ImGui::TableNextColumn();
                    ImGui::Text("Outer surface"); ImGui::TableNextColumn();
                    bcChanged = ImGui::RadioButton("##xp free",    &gP.bcTypeMaxAxis0, 0) || bcChanged; ImGui::TableNextColumn();
                    bcChanged = ImGui::RadioButton("##xp fixed",   &gP.bcTypeMaxAxis0, 1) || bcChanged; ImGui::TableNextColumn();
                    bcChanged = ImGui::RadioButton("##xp tangent", &gP.bcTypeMaxAxis0, 2) || bcChanged;

                    ImGui::TableNextRow(); ImGui::TableNextColumn();
                    ImGui::Text("Min Y plane"); ImGui::TableNextColumn();
                    bcChanged = ImGui::RadioButton("##zm free",    &gP.bcTypeMinAxis1, 0) || bcChanged; ImGui::TableNextColumn();
                    bcChanged = ImGui::RadioButton("##zm fixed",   &gP.bcTypeMinAxis1, 1) || bcChanged; ImGui::TableNextColumn();
                    bcChanged = ImGui::RadioButton("##zm tangent", &gP.bcTypeMinAxis1, 2) || bcChanged;

                    ImGui::TableNextRow(); ImGui::TableNextColumn();
                    ImGui::Text("Max Y plane"); ImGui::TableNextColumn();
                    bcChanged = ImGui::RadioButton("##zp free",    &gP.bcTypeMaxAxis1, 0) || bcChanged; ImGui::TableNextColumn();
                    bcChanged = ImGui::RadioButton("##zp fixed",   &gP.bcTypeMaxAxis1, 1) || bcChanged; ImGui::TableNextColumn();
                    bcChanged = ImGui::RadioButton("##zp tangent", &gP.bcTypeMaxAxis1, 2) || bcChanged;

                } else {
                    ImGui::TableNextRow(); ImGui::TableNextColumn();
                    ImGui::Text("Min X plane"); ImGui::TableNextColumn();
                    bcChanged = ImGui::RadioButton("##xm free",    &gP.bcTypeMinAxis0, 0) || bcChanged; ImGui::TableNextColumn();
                    bcChanged = ImGui::RadioButton("##xm fixed",   &gP.bcTypeMinAxis0, 1) || bcChanged; ImGui::TableNextColumn();
                    bcChanged = ImGui::RadioButton("##xm tangent", &gP.bcTypeMinAxis0, 2) || bcChanged;

                    ImGui::TableNextRow(); ImGui::TableNextColumn();
                    ImGui::Text("Max X plane"); ImGui::TableNextColumn();
                    bcChanged = ImGui::RadioButton("##xp free",    &gP.bcTypeMaxAxis0, 0) || bcChanged; ImGui::TableNextColumn();
                    bcChanged = ImGui::RadioButton("##xp fixed",   &gP.bcTypeMaxAxis0, 1) || bcChanged; ImGui::TableNextColumn();
                    bcChanged = ImGui::RadioButton("##xp tangent", &gP.bcTypeMaxAxis0, 2) || bcChanged;

                    ImGui::TableNextRow(); ImGui::TableNextColumn();
                    ImGui::Text("Min Y plane"); ImGui::TableNextColumn();
                    bcChanged = ImGui::RadioButton("##ym free",    &gP.bcTypeMinAxis1, 0) || bcChanged; ImGui::TableNextColumn();
                    bcChanged = ImGui::RadioButton("##ym fixed",   &gP.bcTypeMinAxis1, 1) || bcChanged; ImGui::TableNextColumn();
                    bcChanged = ImGui::RadioButton("##ym tangent", &gP.bcTypeMinAxis1, 2) || bcChanged;

                    ImGui::TableNextRow(); ImGui::TableNextColumn();
                    ImGui::Text("Max Y plane"); ImGui::TableNextColumn();
                    bcChanged = ImGui::RadioButton("##yp free",    &gP.bcTypeMaxAxis1, 0) || bcChanged; ImGui::TableNextColumn();
                    bcChanged = ImGui::RadioButton("##yp fixed",   &gP.bcTypeMaxAxis1, 1) || bcChanged; ImGui::TableNextColumn();
                    bcChanged = ImGui::RadioButton("##yp tangent", &gP.bcTypeMaxAxis1, 2) || bcChanged;

                    ImGui::TableNextRow(); ImGui::TableNextColumn();
                    ImGui::Text("Min Z plane"); ImGui::TableNextColumn();
                    bcChanged = ImGui::RadioButton("##zm free",    &gP.bcTypeMinAxis2, 0) || bcChanged; ImGui::TableNextColumn();
                    bcChanged = ImGui::RadioButton("##zm fixed",   &gP.bcTypeMinAxis2, 1) || bcChanged; ImGui::TableNextColumn();
                    bcChanged = ImGui::RadioButton("##zm tangent", &gP.bcTypeMinAxis2, 2) || bcChanged;

                    ImGui::TableNextRow(); ImGui::TableNextColumn();
                    ImGui::Text("Max Z plane"); ImGui::TableNextColumn();
                    bcChanged = ImGui::RadioButton("##zp free",    &gP.bcTypeMaxAxis2, 0) || bcChanged; ImGui::TableNextColumn();
                    bcChanged = ImGui::RadioButton("##zp fixed",   &gP.bcTypeMaxAxis2, 1) || bcChanged; ImGui::TableNextColumn();
                    bcChanged = ImGui::RadioButton("##zp tangent", &gP.bcTypeMaxAxis2, 2) || bcChanged;
                }
                ImGui::EndTable();
            }

            if (ImGui::CollapsingHeader("Simulation Parameters")) {
                // float fDamping = gP.damping;
                // ImGui::InputFloat("Damping", &fDamping, 1.0f, 0.05f, "%.2f");
                // gP.damping = std::min(std::max(fDamping, 0.0f), 50.0f);
                int maxIterInt = maxIter;
                ImGui::InputInt("Maximum iteration", &maxIterInt);
                maxIter = std::max(maxIterInt, 0);
                float fkinEnergyTol = kinEnergyTol;
                ImGui::InputFloat("Minimum ΔE", &fkinEnergyTol, 1.0f, 1e-4f, "%.5e");
                kinEnergyTol = std::max(fkinEnergyTol, 0.0f);
                ImGui::InputFloat("Contact weight", &contactFact, 10.0f, 0.05f, "%.2f");
                contactFact = std::max(contactFact, 0.0f);
                // ImGui::InputFloat("Contact damping", &contactFactVel, 1.0f, 0.05f, "%.2f");
                // contactFactVel = std::max(contactFactVel, 0.0f);
                ImGui::InputFloat("Repulsive distance ratio", &repThickness, 0.01f, 0.05f, "%.2f");
                repThickness = std::max(repThickness, 0.0f);
                ImGui::InputInt("Search iteration", &searchIter);
                searchIter = std::max(searchIter, 0);
            }

            // Draw options
            if (ImGui::CollapsingHeader("Draw Options")) {
                setNewMeshToViewer = ImGui::RadioButton("Surface", &meshVisState, 0) || setNewMeshToViewer;
                renderNewData |= setNewMeshToViewer;
                ImGui::SameLine();
                renderNewData = ImGui::RadioButton("Section", &meshVisState, 1) || renderNewData;
                ImGui::SameLine();
                renderNewData = ImGui::RadioButton("Clip", &meshVisState, 2) || renderNewData;
                if (meshVisState == 2) {
                    renderNewData = ImGui::Checkbox("Crinkle", &crinkleClip) || renderNewData;
                    ImGui::SameLine();
                    renderNewData = ImGui::Checkbox("Inverse", &inverseClip) || renderNewData;
                }

                if (meshVisState > 0) {
                    renderNewData = ImGui::RadioButton("X plane", &sliceDir, 0) || renderNewData;
                    ImGui::SameLine();
                    renderNewData = ImGui::InputFloat("##X", &sliceMag[0], 0.01f, 1.0f, "%.3f") || renderNewData;
                    renderNewData = ImGui::RadioButton("Y plane", &sliceDir, 1) || renderNewData;
                    ImGui::SameLine();
                    renderNewData = ImGui::InputFloat("##Y", &sliceMag[1], 0.01f, 1.0f, "%.3f") || renderNewData;
                    renderNewData = ImGui::RadioButton("Z plane", &sliceDir, 2) || renderNewData;
                    ImGui::SameLine();
                    renderNewData = ImGui::InputFloat("##Z", &sliceMag[2], 0.01f, 1.0f, "%.3f") || renderNewData;
                }
                ImGui::Separator();
                ImGui::Text("Scalar color:");
                ImGui::Columns(2, "varNameColorColumns", false);
                for (int i = 0; i < varNamesColor.size(); ++i) {
                    if (ImGui::Selectable(std::get<0>(varNamesColor[i]).c_str(), visibleData == i)) {
                        visibleData = i;
                        renderNewData = true;
                    }
                    ImGui::NextColumn();
                }
                ImGui::Columns(1);
                ImGui::Separator();


                ImGui::Columns(2, "varNameLineColumns", false);
                ImGui::Text("Rank 1 tensors");
                ImGui::NextColumn();
                ImGui::Text("Scale factor");
                ImGui::NextColumn();
                for (int i=0; i<rank1TensorsNames.size(); ++i) {
                    std::string widgetName = std::get<0>(rank1TensorsNames[i]);
                    if (i==0) widgetName += " " + dir1;
                    if (i==1) widgetName += " " + dir2;
                    if (i==2) widgetName += " " + dir3;
                    renderNewData = (ImGui::Checkbox((widgetName + " ##edges").c_str(), &showRank1Tensors[i])) || renderNewData;
                    ImGui::NextColumn();
                    //if (i<3)
                    //    renderNewData = ImGui::InputFloat(("##"+widgetName + "_scale").c_str(), &scaleRank1Tensors[i], 10, 100) || renderNewData;
                    //else
                        renderNewData = ImGui::InputFloat(("##"+widgetName + "_scale").c_str(), &scaleRank1Tensors[i], 1, 10) || renderNewData;
                    ImGui::SameLine();

                    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(
                        ImColor(int(255*rank1TensorColors(0,i)), int(255*rank1TensorColors(1,i)), int(255*rank1TensorColors(2,i)))));
                    ImGui::Text("___");
                    ImGui::PopStyleColor();
                    ImGui::NextColumn();

                    //renderNewData = ImGui::ColorEdit4((widgetName+"Color").c_str(), filamentFibersColor.col(i).data(), ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel) || renderNewData;
                    //ImGui::NextColumn();
                }

                ImGui::Columns(1);


                ImGui::Separator();

                ImGui::Checkbox("Orthographic", &showOrtho);
                ImGui::Checkbox("Axes", &showBasis);
                ImGui::Checkbox("Edges", &showEdges);
                ImGui::Checkbox("Mesh quality", &showMeshQ);
                if (showMeshQ) {
                    ImGui::SameLine();
                    renderNewData = ImGui::InputInt("##bins", &numBins, 10, 10) || renderNewData;
                    numBins = std::max(numBins, 10);
                }

                // ImGui::SameLine();
                // ImGui::Checkbox("Axes", &showBasis);
                // ImGui::SameLine();
                // ImGui::Checkbox("Edges", &showEdges);
                // ImGui::SameLine();
                // ImGui::ColorEdit4("Background", viewer.core().background_color.data(), ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_PickerHueWheel);

                ImGui::Separator();
                ImGui::InputInt("Frame iterations", &sUpdate, 10, 10);
                sUpdate = std::max(sUpdate, 1);
//                ImGui::InputFloat("Font scale", &fontScale, 0.5, 0.5);
//                fontScale = std::max(fontScale, 0.5f);

                ImGui::Text("Reset view plane : ");
                if (ImGui::Button(" X+Y ")) {
                    resetViewFlag = true;
                    trackball_angle = Eigen::Quaternionf::Identity();
                }
                ImGui::SameLine();
                if (ImGui::Button(" X+Z ")) {
                    resetViewFlag = true;
                    trackball_angle = Eigen::Quaternionf(1.f/ sqrt(2.f),1.f/ sqrt(2.f),0.,0.0);
                }
                ImGui::SameLine();
                if (ImGui::Button(" Z+Y ")) {
                    resetViewFlag = true;
                    trackball_angle = Eigen::Quaternionf(1.f/ sqrt(2.f),0.,-1.f/ sqrt(2.f),0.0);
                }
                if (ImGui::Button(" X-Y ")) {
                    resetViewFlag = true;
                    trackball_angle = Eigen::Quaternionf(0.0f,1.0f,0.0f,0.0f);
                }
                ImGui::SameLine();
                if (ImGui::Button(" X-Z ")) {
                    resetViewFlag = true;
                    trackball_angle = Eigen::Quaternionf(1.f/ sqrt(2.f),-1.f/ sqrt(2.f),0.,0.0);
                }
                ImGui::SameLine();
                if (ImGui::Button(" Z-Y ")) {
                    resetViewFlag = true;
                    trackball_angle = Eigen::Quaternionf(1.f/ sqrt(2.f),0.,1.f/ sqrt(2.f),0.0);
                }
            }

            ImGui::Separator();

            float ftemp = timeFactor;
            ImGui::SliderFloat("Time factor", &ftemp, 0.0f, 1.0f);
            timeFactor = ftemp;

            ImGui::Separator();
            run = run && !paramChanged;

            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(ImColor(0, 255, 0)));
            if (!paramChanged) IMGUI_DISABLE_WIDGET
            compileFlag = ImGui::Button("Compile", ImVec2(-1, 0)) || compileFlag;
            if (!paramChanged) IMGUI_ENABLE_WIDGET
            ImGui::PopStyleColor();

            if (!run) {
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(ImColor(0, 255, 0)));
                if (paramChanged) IMGUI_DISABLE_WIDGET
                run = ImGui::Button("Run simulation", ImVec2(-1, 0));
                if (paramChanged) IMGUI_ENABLE_WIDGET
                if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled) && paramChanged) {
                    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(ImColor(255, 255, 255)));
                    ImGui::SetTooltip("Push ""Compile"" button first !");
                    ImGui::PopStyleColor();
                }
                if (run || shutDown)
                    runSim.notify_all();
                ImGui::PopStyleColor();
            } else {
                lowTimeFactorFlag = false;
                maxIterFlag = false;
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(ImColor(255, 0, 0)));
                run = !ImGui::Button("Pause simulation", ImVec2(-1, 0));
                ImGui::PopStyleColor();
                run = run && (timeFactor>1e-4);
            }

            if (saveMP4) {
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(ImColor(255, 0, 0)));
                if (ImGui::Button("Pause movie", ImVec2(-1, 0)))
                    saveMP4 = false;
                ImGui::PopStyleColor();
            } else {
                ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(ImColor(0, 255, 0)));
                if (ImGui::Button("Record movie", ImVec2(-1, 0))) {
                    saveMP4 = true;
                }
                ImGui::PopStyleColor();
            }

            if (!lowTimeFactorFlag)
                lowTimeFactorFlag = ImGuiMessage(timeFactor<1e-4, "Warning !", "Time-step became too small.\nYou may need to change the model parameters.\nPress \"Reset\" button before running again.");
            if (!maxIterFlag)
                maxIterFlag = ImGuiMessage(simIter>=maxIter || (kinEnergy<kinEnergyTol && simIter > averageInterval),
                    "Stop condition is reached!", "To resume, change the max iteration or total energy limits.");

            if (ImGui::Button("Save Project", ImVec2(-1, 0))) {
                std::string fname = igl::file_dialog_save();
                if (fname.length() != 0) {
                    fname = fname +
                            ((fname.compare(fname.length() - 4, fname.length(), ".txt") == 0) ? "" : ".txt");
                    //mesh.save_surface(fname, thrust::host_vector<Vector>(data.pos));
                    writeSetting(fname);
                }
            }

            resetSimFlag = ImGui::Button("Reset", ImVec2((wW - wP) * 0.5f, 0)) || resetSimFlag;
            ImGui::SameLine(0, wP);
            reloadMeshFile = ImGui::Button("Reload", ImVec2((wW - wP) * 0.5f, 0)) || reloadMeshFile;

            if (ImGui::Button("Save VTK", ImVec2((wW - wP) * 0.5f, 0))) {
                std::string fname = igl::file_dialog_save();
                if (fname.length() != 0) {
                    fname = fname + ((fname.compare(fname.length() - 4, fname.length(), ".vtk") == 0) ? "" : ".vtk");
                    mesh.save_binary(fname, data.tet, data.pos, dataToSave);
                }
            }

            ImGui::SameLine(0, wP);

            if (ImGui::Button("Save PLY", ImVec2((wW - wP) * 0.5f, 0))) {
                // std::string fname = igl::file_dialog_save();
                // if (fname.length() != 0) {
                //     fname = fname +
                //             ((fname.compare(fname.length() - 4, fname.length(), ".ply") == 0) ? "" : ".ply");
                //
                //     thrust::host_vector<Vector> tempPos(data.pos);
                //     Eigen::Map<VectorArrayEigen> visPos(&tempPos[0][0], mesh.nver, 3);
                //     Eigen::MatrixXf visPosA(visPos.cast<float>());
                //     ScalarArrayEigen visR,visG,visB;
                //     computeNodalColor(visR,visG,visB);
                //     Eigen::MatrixXd nodalColor(mesh.boundaryNodeIds.size(),3);
                //     nodalColor.col(0) = visR(mesh.boundaryNodeIds).cast<double>();
                //     nodalColor.col(1) = visG(mesh.boundaryNodeIds).cast<double>();
                //     nodalColor.col(2) = visB(mesh.boundaryNodeIds).cast<double>();
                //
                //     writePLY(fname,visPosA(mesh.boundaryNodeIds, Eigen::all).cast<double>(), nodalColor,mesh.tri_mapped.matrix());
                //     fs::permissions(fname,fs::perms::owner_all | fs::perms::group_all | fs::perms::others_all);
                // }
            }

            // if (ImGui::Button("Save PLY", ImVec2((wW - wP) * 0.5f, 0))) {
            //     if ((R.rows()!=gWWidth) || (R.cols()!=gWHeight)) {
            //         R.resize(gWWidth,gWHeight);
            //         G.resize(gWWidth,gWHeight);
            //         B.resize(gWWidth,gWHeight);
            //         A.resize(gWWidth,gWHeight);
            //     }
            //     char fname[16];
            //     // snprintf(fname, 16, "_%04d.png", frames);
            //     vc_mesh->draw_buffer(*vd_mesh, false, R, G, B, A);
            //     // std::string tempOutput = outputFileName;
            //     // tempOutput.replace(tempOutput.end()-4,tempOutput.end(),fname);
            //     // igl::stb::write_image(tempOutput, R, G, B, A);
            //     // fs::permissions(tempOutput, fs::perms::owner_all | fs::perms::group_all | fs::perms::others_all);
            //     uint8_t* rgb = packRGB(R, G, B);
            //     videoWriter.addFrame(rgb);
            //     delete[] rgb;
            // }

            ImGui::Text("Save frames :");
            ImGui::SameLine();
            ImGui::Checkbox("VTK", &saveVTK);
            ImGui::SameLine();
            ImGui::Checkbox("PLY", &savePLY);

            ImGui::Separator();
            if (ImGui::CollapsingHeader("Simulation Info", ImGuiTreeNodeFlags_DefaultOpen)) {
                ImGui::Text("X Range  : %.3f, %.3f", minPos[0], maxPos[0]);
                ImGui::Text("Y Range  : %.3f, %.3f", minPos[1], maxPos[1]);
                ImGui::Text("Z Range  : %.3f, %.3f", minPos[2], maxPos[2]);
                ImGui::Text("R Range  : %.3f, %.3f", minR, maxR);
                ImGui::Text("Time                   : %.5f", globalTime);
                ImGui::Text("Time factor x dt       : %.5f x %.5f", timeFactor, maxDt);

                // ImGui::Text("Total energy           : %.5f",
                //     thrust::reduce(data.potEnergy.begin(), data.potEnergy.end())
                //         / thrust::reduce(data.tetVol.begin(), data.tetVol.end()));
                ImGui::Text("Total energy           : %.5e",kinEnergy);
                ImGui::Text("Number of nodes        : %d", mesh.nver);
                ImGui::Text("Number of tetrahedra   : %d", mesh.ntet);
                ImGui::Text("Number of triangle     : %d", mesh.ntri);
                ImGui::Text("Iteration              : %d", (int)simIter);
                ImGui::Text("Iterations per seconds : %.3f", isfinite(FPS) ? FPS : 0.0);
            }

            if (modelChanged) {
                ImGui::PopItemFlag();
                ImGui::PopStyleVar();
            }

            ImGui::End();

            // Rendering
            ImGui::Render();
            glViewport(0, 0, (int) io.DisplaySize.x, (int) io.DisplaySize.y);
            glClearColor(clear_color.x, clear_color.y, clear_color.z, clear_color.w);
            glClear(GL_COLOR_BUFFER_BIT);
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
            SDL_GL_SwapWindow(window);

            if (applyChanges) {
                if (inputGeom == Arbitrary) {
                    if (inputMeshFileName.compare(inputMeshFileName.length() - 4, inputMeshFileName.length(), ".vtk") == 0) {
                        if (mesh.init_from_file(inputMeshFileName)>0)
                            init();
                        else
                            _ERROR_MESSAGE("Input mesh can't be loaded !");
                    } else
                        _ERROR_MESSAGE("Input mesh in not a vtk file !");
                // } else if (inputType == 1) {
                //     grid.init(domainSize[0],domainSize[1],domainSize[2],domainDx[0],domainDx[1],domainDx[2],mesh);
                //     init();
                }  else if (inputGeom == Box) {
                    mesh.init_from_multi_layer_box(boxDims.L,boxDims.W,
                        std::vector<double>(boxDims.H, boxDims.H + nLayers),
                        std::vector<double>(boxDims.spacing, boxDims.spacing + nLayers));
                    init();
                }  else if (inputGeom == Disk) {
                    mesh.init_from_multi_layer_disk(diskDims.R,
                        std::vector<double>(diskDims.H, diskDims.H + nLayers),
                        std::vector<double>(diskDims.spacing, diskDims.spacing + nLayers));
                    init();
                }  else if (inputGeom == Tube) {
                    std::vector<double> radii; radii.push_back(tubeDims.Ri);
                    for (int i=0; i<nLayers; i++) {radii.push_back(radii.back() + tubeDims.H[i]);}
                    mesh.init_from_multi_layer_tube(tubeDims.L ,radii, std::vector<double>(tubeDims.spacing, tubeDims.spacing + nLayers));
                    init();
                }  else if (inputGeom == Cone) {
                    std::vector<double> radii; radii.push_back(coneDims.Ri);
                    for (int i=0; i<nLayers; i++) {radii.push_back(radii.back() + coneDims.H[i]/std::cos(coneDims.apexAng * 0.5 * M_PI / 180.0));}
                    mesh.init_from_multi_layer_cone(coneDims.L,coneDims.apexAng * M_PI / 180.0, radii, std::vector<double>(coneDims.spacing, coneDims.spacing + nLayers));
                    init();
                }  else if (inputGeom == Sphere) {
                    std::vector<double> radii; radii.push_back(sphereDims.Ri);
                    for (int i=0; i<nLayers; i++) {radii.push_back(radii.back() + sphereDims.H[i]);}
                    mesh.init_from_multi_layer_sphere(radii, std::vector<double>(sphereDims.spacing, sphereDims.spacing + nLayers));
                    init();
                }  else if (inputGeom == Torus) {
                    std::vector<double> radii; radii.push_back(torusDims.Ri);
                    for (int i=0; i<nLayers; i++) {radii.push_back(radii.back() + torusDims.H[i]);}
                    mesh.init_from_multi_layer_torus(torusDims.R, radii, std::vector<double>(torusDims.spacing, torusDims.spacing + nLayers));
                    init();
                }

                layerFaceColors.resize(mesh.ntri, 4);

                modelChanged = false;
                initSimFlag = true;
                setNewMeshToViewer = true;
                renderNewData = true;
            }

            if (initSimFlag) {
                initSimFlag = false;
                run = false;
                runSim.notify_all();

                init();
                clockTime = 0.0;
                globalTime = 0.0;
                std::cout<<"time factor reset to 1.0"<<std::endl;
                timeFactor = 1.0;
                colorChanged = true;
                bcChanged = true;
                paramChanged = true;
                simIter = 0;
                iterCounter = 0;
                FPS = 0.0f;
            }

            if (reloadMeshFile) {
                reloadMeshFile = false;
                run = false;
                if (inputGeom == Arbitrary) {
                    if (inputMeshFileName.compare(inputMeshFileName.length() - 4, inputMeshFileName.length(), ".vtk") == 0) {
                        if (mesh.init_from_file(inputMeshFileName)>0)
                            init();
                        else
                            _ERROR_MESSAGE("Input mesh can't be loaded !");
                    } else
                        _ERROR_MESSAGE("Input mesh in not a vtk file !");
                } else {
                    init();
                }
                clockTime = 0.0;
                std::cout<<"time factor reset to 1.0"<<std::endl;
                timeFactor = 1.0;
                colorChanged = true;
                bcChanged = true;
                paramChanged = true;
                simIter = 0;
                iterCounter = 0;
                FPS = 0.0f;
            }

            if (defaultParamFlag) {
                gP2.init();
                for (int i = 0; i < MAX_NLAYERS; i++)
                    gP.isRigidLayer[i] = false;
                gP.grRateGlobal = Tensor(0.0);
                gP.damping = 0.0;
                gP.useMeshDef_E = false;
                gP.useMeshDef_nu = false;
                gP.useMeshDef_viscosity = false;
                gP.useMeshDef_plasticity = false;
                gP.useMeshDef_grRate1_Ref = false;
                gP.useMeshDef_grRate2_Ref = false;
                gP.useMeshDef_grRate3_Ref = false;
                gP.useMeshDef_k1 = false;
                gP.useMeshDef_k2 = false;
                gP.useMeshDef_fiber1_Ref = false;
                gP.useMeshDef_fiber2_Ref = false;
                gP.useMeshDef_fiber3_Ref = false;
                gP.useMeshDef_fiber4_Ref = false;
                gP.useMeshDef_actin_Ref = false;
                gP.bcTypeMinAxis0 = 0;
                gP.bcTypeMaxAxis0 = 0;
                gP.bcTypeMinAxis1 = 0;
                gP.bcTypeMaxAxis1 = 0;
                gP.bcTypeMinAxis2 = 0;
                gP.bcTypeMaxAxis2 = 0;


                defaultParamFlag = false;
            }

            if (compileFlag) {
                data.isRigid.assign(mesh.nver, 0);
                cudaDeviceSynchronize();
                _LAUNCH(mesh.ntet, 256, mark_rigid_nodes) (dataPtr, mesh.ntet);
                cudaDeviceSynchronize();
                if (compile()) {
                    paramChanged = false;
                }
                compileFlag = false;
                resetSimFlag = true;
            }

            if (resetSimFlag) {
                resetSimFlag = false;
                run = false;
                init(false);
                clockTime = 0.0;
                std::cout<<"time factor reset to 1.0"<<std::endl;
                timeFactor = 1.0;
                colorChanged = true;
                bcChanged = true;
                simIter = 0;
                iterCounter = 0;
                videoCounter = 0;
                FPS = 0.0f;
            }

            if (colorChanged) {
                setNewMeshToViewer = true;
                renderNewData = true;
                colorChanged = false;
            }
            applyChanges = false;

            if (bcChanged) {
                bcChanged = false;
                _LAUNCH(mesh.nver, 256, compute_bids) (dataPtr, bcTol*spacing, mesh.nver);
                cudaDeviceSynchronize();

                ScalarArrayEigen distance = getSignedDist(mesh.pos,mesh.pos,mesh.tri);
                ScalarArrayDev distanceDev(mesh.nver,0.0);
                thrust::copy(reinterpret_cast<Float*>(distance.data()), reinterpret_cast<Float*>(distance.data())+mesh.nver, distanceDev.begin());
                _LAUNCH(mesh.ntet, 256, compute_normal_from_dist) (dataPtr, thrust::raw_pointer_cast(distanceDev.data()), mesh.ntet);
                cudaDeviceSynchronize();
                _LAUNCH(mesh.ntet, 256, compute_orthogonal_basis) (dataPtr, mesh.ntet);
                cudaDeviceSynchronize();
                renderNewData = true;
            }
        }
        // Cleanup
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplSDL2_Shutdown();
        ImGui::DestroyContext();

        SDL_GL_DeleteContext(gl_context);
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 0;
    };

    std::thread *draw_mesh_thread, *draw_menu_thread, *nogui_thread, *fem_iterate_thread;
    if (runMode==0) {
        draw_mesh_thread = new std::thread(draw_mesh);
        std::this_thread::sleep_for(std::chrono::seconds(1));
        draw_menu_thread = new std::thread(draw_menu);
    } else {
        nogui_thread = new std::thread(nogui);
    }
    fem_iterate_thread = new std::thread (fem_iterate);

    if (runMode==0) {
        draw_mesh_thread->join();
        draw_menu_thread->join();
    } else {
        nogui_thread->join();
    }
    fem_iterate_thread->join();

    if (runMode==0) {
        delete draw_mesh_thread;
        delete draw_menu_thread;
    } else {
        delete nogui_thread;
    }
    delete fem_iterate_thread;
    delete dataPtr;

    return 0;
}
