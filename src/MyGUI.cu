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

#include <immintrin.h>
#include "thrust/logical.h"
#include "thrust/extrema.h"
#include <thrust/iterator/constant_iterator.h>
#include <thrust/sort.h>

#include "MyGUI.h"
#include "Octree.h"
#include "TextIO.h"
#include "igl/signed_distance.h"


#include <iostream>
#include <stdexcept>

#include <cuda.h>
#include <nvrtc.h>


#include <cuda_runtime.h>
#include <string>
#include <unistd.h>

std::string getExecutablePath() {
    std::vector<char> buf(1024);
    ssize_t len = readlink("/proc/self/exe", buf.data(), buf.size());
    if (len == -1) return "";
    return std::string(buf.data(), len);
}

std::string detect_nvrtc_arch(int device_id = 0) {
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, device_id) != cudaSuccess) {
        throw std::runtime_error("Failed to query CUDA device properties");
    }

    int major = prop.major;
    int minor = prop.minor;

    // Clamp to architectures supported by your NVRTC version if needed
    // Example: CUDA 12.x supports up to compute_90
    if (major > 9) major = 9;

    std::string result = std::string("--gpu-architecture=compute_") + std::to_string(major) + std::to_string(minor);
    // std::cout<< result << std::endl;
    return result;
}

#define CUDA_CHECK(err) if (err != CUDA_SUCCESS) { \
std::cerr << "CUDA error: " << err << std::endl; exit(1); }

#define NVRTC_CHECK(err) if (err != NVRTC_SUCCESS) { \
throw std::runtime_error("NVRTC error: " + std::string(nvrtcGetErrorString(err))); }

// Utility to read a whole file into a string
std::string loadTextSource(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

// Compile CUDA source string into PTX at runtime
std::string compileKernel(const std::string& kernelCode, const std::string& kernelName) {
    nvrtcProgram prog;
    try {
        NVRTC_CHECK(nvrtcCreateProgram(&prog, kernelCode.c_str(), kernelName.c_str(), 0, NULL, NULL));
    }
    catch (const std::runtime_error& e) {
        std::cout << e.what() << std::endl;
        return "";
    }
    std::string arch_opt = detect_nvrtc_arch();
    std::cout << "Detected architecture: " << arch_opt << std::endl;
    const char *opts[] = {"--include-path=" CUDA_INCLUDE_PATH, "--disable-warnings", arch_opt.c_str()};
    nvrtcResult res = nvrtcCompileProgram(prog, 3, opts);

    // Print log if any
    size_t logSize;
    nvrtcGetProgramLogSize(prog, &logSize);
    if (logSize > 1) {
        std::string log(logSize, '\0');
        nvrtcGetProgramLog(prog, &log[0]);
        std::cout << log << std::endl;
    }
    try {
        NVRTC_CHECK(res);
    }
    catch (const std::runtime_error& e) {
        std::cout << e.what() << std::endl;
        return "";
    }

    // Extract PTX
    size_t ptxSize;
    try {
        NVRTC_CHECK(nvrtcGetPTXSize(prog, &ptxSize));
    }
    catch (const std::runtime_error& e) {
        std::cout << e.what() << std::endl;
        return "";
    }

    std::string ptx(ptxSize, '\0');
    try {
        NVRTC_CHECK(nvrtcGetPTX(prog, &ptx[0]));
    }
    catch (const std::runtime_error& e) {
        std::cout << e.what() << std::endl;
        return "";
    }

    nvrtcDestroyProgram(&prog);

    return ptx;
}

// Load PTX into a CUDA module and get a function handle
CUfunction loadKernel(const std::string& ptx, CUmodule& module, const std::string& funcName) {
    if (ptx.empty())
        return nullptr;
    CUDA_CHECK(cuModuleLoadDataEx(&module, ptx.c_str(), 0, 0, 0));
    CUfunction func;
    CUDA_CHECK(cuModuleGetFunction(&func, module, funcName.c_str()));
    return func;
}

namespace LagSol {

    bool applyChanges = false;
    bool reloadMeshFile = false;
    bool shutDown = false;
    bool run = false;
    bool basisChanged = true;
    bool modelChanged = true;
    bool paramChanged = false;
    bool colorChanged = false;
    bool bcChanged = false;
    bool simDiverged = false;
    bool initSimFlag = false;
    bool compileFlag = false;
    bool defaultParamFlag = false;
    bool resetSimFlag = false;
    bool resetViewFlag = false;
    bool renderNewData = true;
    bool setNewMeshToViewer = true;
    bool firstTimeSetMeshToViewer = true;
    double FPS = INFINITY;
    double clockTime = 0.0;
    int iterCounter = 0;
    int videoCounter = 0;
    float enDiff[1000], kinEnergy = INFINITY;
    int averageInterval = 1000;
    Float dt;
    Float maxDt = 1.0f;
    Float spacing;
    Float charL;
    Float charLRef;
    Vector maxPos = 0;
    Vector minPos = 0;
    Float minR = 0;
    Float maxR = 0;
    Float bcTol = 1.0e-3;


    BasicGeometry inputGeom = Box;
    int nLayers = 1;
    int nFibers = 0;
    Float globalTime = 0;
    std::string inputMeshFileName = "";
    std::string outputFileName = "./../output.vtk";
    BoxDims boxDims;
    DiskDims diskDims;
    TubeDims tubeDims;
    ConeDims coneDims;
    SphereDims sphereDims;
    TorusDims torusDims;

    Float saveTime = 100000.0;
    int frames = 0;
    bool saveVTK = false;
    bool savePLY = false;
    bool saveMP4 = false;
    bool showBasis = true;
    bool showMeshQ = true;
    int numBins = 50;
    bool showEdges = false;
    bool showOrtho = false;
    int searchIter = 400;
    size_t maxIter = 200000;
    Float kinEnergyTol = 1e-3;
    Float timeThreshold = 10000000.0;

    size_t simIter = 0;
    bool crinkleClip = false;
    bool inverseClip = false;
    int meshVisState = 0;
    int selectedLayer = 0;
    int sliceDir = 0;
    float sliceMag[3] = {0.5, 0.5, 0.5};
    int visibleData = 0;
    bool showRank1Tensors[8] = {false, false, false, false, false, false, false, false};
    float scaleRank1Tensors[8] = {500.0, 500.0, 500.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    float timeFactor = 1.0;
    double minTimeFactor = 1e-6;
    int sUpdate = 50;
    float fontScale = 1.0;
    float pert = 0.1;
    int gWWidth, gWHeight;
    Eigen::Vector4f bkgColor(0.45, 0.45, 0.45, 1.0);
    Eigen::Array4Xf layersColorEigen(4, MAX_NLAYERS);
    Eigen::Array4Xf rank1TensorColors(4, 8);
    Eigen::Quaternionf trackball_angle = Eigen::Quaternionf::Identity();
    Eigen::Quaternionf current_trackball_angle = Eigen::Quaternionf::Identity();

    float contactFact = 500.0;
    float contactFactVel = 0.0;
    float repThickness = 0.3;

    Parameters gP;


    Eigen::ArrayX3d triCenter;
    Eigen::MatrixXd sliceVertex;
    Eigen::MatrixXi sliceFace;
    Eigen::VectorXd sliceData;
    Eigen::MatrixXd Normals;
    Eigen::MatrixX4d layerFaceColors;

    Eigen::ArrayX3d triCenter2;
    Eigen::MatrixXd sliceVertex2;
    Eigen::MatrixXi sliceFace2;
    Eigen::VectorXd sliceData2;
    Eigen::MatrixXd Normals2;
    Eigen::MatrixX4d layerFaceColors2;

    Eigen::Matrix<double, 256, 3> rgb;
    GLuint colorMapId = 0;

    Mesh mesh;
    DeviceData data;
    DeviceDataPtrManaged *dataPtr;

    thrust::device_vector <Vector> oldPos;
    thrust::device_vector <Vector> oldPosRef;
    thrust::device_vector <Vector> oldVel;
    thrust::device_vector <Tensor> oldStress;
    thrust::device_vector<Tensor> oldFp;
    thrust::device_vector<Tensor> oldFg;
    Float oldGlobalTime;
    int oldFrame = 0;

    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> R(3840, 2160);
    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> G(3840, 2160);
    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> B(3840, 2160);
    Eigen::Matrix<unsigned char, Eigen::Dynamic, Eigen::Dynamic> A(3840, 2160);

    void recover_from_snapshot() {
        data.pos = oldPos;
        data.posRef = oldPosRef;
        data.vel = oldVel;
        data.stress = oldStress;
        data.Fg = oldFg;
        data.Fp = oldFp;
        globalTime = oldGlobalTime;
        frames = oldFrame;
    }

    void save_to_snapshot() {
        oldPos = data.pos;
        oldPosRef = data.posRef;
        oldVel = data.vel;
        oldStress = data.stress;
        oldFg = data.Fg;
        oldFp = data.Fp;
        oldGlobalTime = globalTime;
        oldFrame = frames;
    }

    void get_domain_limits(const VectorArrayDev &pos) {
        ScalarArrayDev xArray(pos.size());
        ScalarArrayDev yArray(pos.size());
        ScalarArrayDev zArray(pos.size());


        _LAUNCH(pos.size(), 256, copy_pos_xyz) (thrust::raw_pointer_cast(pos.data()),
                thrust::raw_pointer_cast(xArray.data()),
                thrust::raw_pointer_cast(yArray.data()),
                thrust::raw_pointer_cast(zArray.data()),
                pos.size());
        cudaDeviceSynchronize();
        auto mX = thrust::minmax_element(xArray.begin(), xArray.end());
        auto mY = thrust::minmax_element(yArray.begin(), yArray.end());
        auto mZ = thrust::minmax_element(zArray.begin(), zArray.end());

        minPos[0] = *mX.first;
        minPos[1] = *mY.first;
        minPos[2] = *mZ.first;
        maxPos[0] = *mX.second;
        maxPos[1] = *mY.second;
        maxPos[2] = *mZ.second;

        if (gP.grCoordType == CylindricalZ) {
            ScalarArrayDev rArray(pos.size());
            thrust::transform(pos.begin(), pos.end(), rArray.begin(), [=] __device__(const Vector& v) {return sqrt(v[0]*v[0] + v[1]*v[1]);});
            auto rMinMax = thrust::minmax_element(rArray.begin(), rArray.end());
            minR = 0.0;
            maxR = *rMinMax.second;
        }

        if (gP.grCoordType == CylindricalY) {
            ScalarArrayDev rArray(pos.size());
            thrust::transform(pos.begin(), pos.end(), rArray.begin(), [=] __device__(const Vector& v) {return sqrt(v[0]*v[0] + v[2]*v[2]);});
            auto rMinMax = thrust::minmax_element(rArray.begin(), rArray.end());
            minR = *rMinMax.first;
            maxR = *rMinMax.second;
        }

        if (gP.grCoordType == ConeAdapted) {
            ScalarArrayDev rArray(pos.size());
            thrust::transform(pos.begin(), pos.end(), rArray.begin(), [=] __device__(const Vector& v) {return sqrt(v[0]*v[0] + v[2]*v[2]);});
            auto rMinMax = thrust::minmax_element(rArray.begin(), rArray.end());
            minR = *rMinMax.first;
            maxR = *rMinMax.second;
        }

        if (gP.grCoordType == Spherical) {
            ScalarArrayDev rArray(pos.size());
            thrust::transform(pos.begin(), pos.end(), rArray.begin(), [=] __device__(const Vector& v) {return sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);});
            auto rMinMax = thrust::minmax_element(rArray.begin(), rArray.end());
            minR = *rMinMax.first;
            maxR = *rMinMax.second;
        }

        if (gP.grCoordType == Toroidal) {
            ScalarArrayDev rArray(pos.size());
            Float RTorus = gP.RTorus;
            thrust::transform(pos.begin(), pos.end(), rArray.begin(), [=] __device__(const Vector& v) {return sqrt((sqrt(v[0]*v[0] + v[1]*v[1]) - RTorus)*(sqrt(v[0]*v[0] + v[1]*v[1]) - RTorus)+v[2]*v[2]);});
            auto rMinMax = thrust::minmax_element(rArray.begin(), rArray.end());
            minR = *rMinMax.first;
            maxR = *rMinMax.second;
        }
        charL = (maxPos - minPos).norm1();

    }

    template<class T1, class T2>
    bool copy_eigen_to_device(T1 &meshData, T2& deviceData) {
        typedef typename T2::value_type T;
        if (meshData.rows() == deviceData.size() && meshData.cols() == sizeof(T)/sizeof(Float)) {
            auto tmp = T2(std::vector<T>(reinterpret_cast<T*>(meshData.data()),reinterpret_cast<T*>(meshData.data())+meshData.rows()));
            deviceData.assign(tmp.begin(), tmp.end());
        } else {
            return false;
        }
        return true;
    }

    void generate_nvrtc_kernel_source(const std::string& pathToKernel, const std::string& inputName, const std::string& outputName, int nLayers) {
        auto kernelContent = getFile(pathToKernel + "/" + inputName);

        findAndReplaceToEndOfLine(kernelContent, "/* default */#include \"../src/Typedefs.h\"",  "#include \"" + pathToKernel +"../../src/Typedefs.h\"");
        findAndReplaceToEndOfLine(kernelContent, "/* default */#include \"../src/Primitives.h\"",  "#include \"" + pathToKernel +"../../src/Primitives.h\"");
        findAndReplaceToEndOfLine(kernelContent, "/* default */#include \"../src/DeviceDataPtr.h\"",  "#include \"" + pathToKernel +"../../src/DeviceDataPtr.h\"");
        findAndReplaceToEndOfLine(kernelContent, "/* default */#include \"../src/SVD3Cuda.h\"",  "#include \"" + pathToKernel +"../../src/SVD3Cuda.h\"");

        findAndReplaceToEndOfLine(kernelContent, "/* default */bool isRigid", std::string("bool isRigid = ")  + getLayerExpressionsBoolean(gP.isRigidLayer, nLayers) + std::string(";"));

        if (gP.useMeshDef_E)
            findAndReplaceToEndOfLine(kernelContent, "/* default */Float E", std::string("Float E = data->E[i];"));
        else
            findAndReplaceToEndOfLine(kernelContent, "/* default */Float E", std::string("Float E = ")  + getLayerExpressions(gP.E,   nLayers) + std::string(";"));

        if (gP.useMeshDef_nu)
            findAndReplaceToEndOfLine(kernelContent, "/* default */Float nu", std::string("Float nu = data->nu[i];"));
        else
            findAndReplaceToEndOfLine(kernelContent, "/* default */Float nu", std::string("Float nu = ") + getLayerExpressions(gP.nu, nLayers) + std::string(";"));

        if (gP.useMeshDef_viscosity)
            findAndReplaceToEndOfLine(kernelContent, "/* default */Float visc", std::string("Float visc = data->visc[i];"));
        else
            findAndReplaceToEndOfLine(kernelContent, "/* default */Float visc", std::string("Float visc = ") + getLayerExpressions(gP.visc, nLayers) + std::string(";"));

        if (gP.useMeshDef_plasticity)
            findAndReplaceToEndOfLine(kernelContent, "/* default */Float plasticity", std::string("Float plasticity = data->plasticity[i];"));
        else
            findAndReplaceToEndOfLine(kernelContent, "/* default */Float plasticity", std::string("Float plasticity = ") + getLayerExpressions(gP.plasticity, nLayers) + std::string(";"));

        if (gP.useMeshDef_grRate1_Ref)
             findAndReplaceToEndOfLine(kernelContent, "/* default */Float grRate1_Ref", std::string("Float grRate1_Ref = data->grRate1_Ref[i];"));
        else findAndReplaceToEndOfLine(kernelContent, "/* default */Float grRate1_Ref", std::string("Float grRate1_Ref = ") + getLayerExpressions(gP.grRate1_Ref, nLayers) + std::string(";"));
        if (gP.useMeshDef_grRate2_Ref)
             findAndReplaceToEndOfLine(kernelContent, "/* default */Float grRate2_Ref", std::string("Float grRate2_Ref = data->grRate2_Ref[i];"));
        else findAndReplaceToEndOfLine(kernelContent, "/* default */Float grRate2_Ref", std::string("Float grRate2_Ref = ") + getLayerExpressions(gP.grRate2_Ref, nLayers) + std::string(";"));
        if (gP.useMeshDef_grRate3_Ref)
             findAndReplaceToEndOfLine(kernelContent, "/* default */Float grRate3_Ref", std::string("Float grRate3_Ref = data->grRate3_Ref[i];"));
        else findAndReplaceToEndOfLine(kernelContent, "/* default */Float grRate3_Ref", std::string("Float grRate3_Ref = ") + getLayerExpressions(gP.grRate3_Ref, nLayers) + std::string(";"));

        if (gP.useMeshDef_k1)
             findAndReplaceToEndOfLine(kernelContent, "/* default */Float k1", std::string("Float k1 = data->k1[i];"));
        else findAndReplaceToEndOfLine(kernelContent, "/* default */Float k1", std::string("Float k1 = ") + getLayerExpressions(gP.k1, nLayers) + std::string(";"));

        if (gP.useMeshDef_k2)
             findAndReplaceToEndOfLine(kernelContent, "/* default */Float k2", std::string("Float k2 = data->k2[i];"));
        else findAndReplaceToEndOfLine(kernelContent, "/* default */Float k2", std::string("Float k2 = ") + getLayerExpressions(gP.k2, nLayers) + std::string(";"));

        if (gP.useMeshDef_fiber1_Ref)
             findAndReplaceToEndOfLine(kernelContent, "/* default */Tensor fiber1_Ref", std::string("Tensor fiber1_Ref = data->fiber1_Ref[i];"));
        else findAndReplaceToEndOfLine(kernelContent, "/* default */Tensor fiber1_Ref", std::string("Tensor fiber1_Ref = ") + getLayerExpressions(gP.fiber1_Ref, nLayers) + std::string(";"));
        if (gP.useMeshDef_fiber2_Ref)
             findAndReplaceToEndOfLine(kernelContent, "/* default */Tensor fiber2_Ref", std::string("Tensor fiber2_Ref = data->fiber2_Ref[i];"));
        else findAndReplaceToEndOfLine(kernelContent, "/* default */Tensor fiber2_Ref", std::string("Tensor fiber2_Ref = ") + getLayerExpressions(gP.fiber2_Ref, nLayers) + std::string(";"));
        if (gP.useMeshDef_fiber3_Ref)
             findAndReplaceToEndOfLine(kernelContent, "/* default */Tensor fiber3_Ref", std::string("Tensor fiber3_Ref = data->fiber3_Ref[i];"));
        else findAndReplaceToEndOfLine(kernelContent, "/* default */Tensor fiber3_Ref", std::string("Tensor fiber3_Ref = ") + getLayerExpressions(gP.fiber3_Ref, nLayers) + std::string(";"));
        if (gP.useMeshDef_fiber4_Ref)
             findAndReplaceToEndOfLine(kernelContent, "/* default */Tensor fiber4_Ref", std::string("Tensor fiber4_Ref = data->fiber4_Ref[i];"));
        else findAndReplaceToEndOfLine(kernelContent, "/* default */Tensor fiber4_Ref", std::string("Tensor fiber4_Ref = ") + getLayerExpressions(gP.fiber4_Ref, nLayers) + std::string(";"));
        if (gP.useMeshDef_actin_Ref)
             findAndReplaceToEndOfLine(kernelContent, "/* default */Tensor actin_Ref", std::string("Tensor actin_Ref = data->actin_Ref[i];"));
        else findAndReplaceToEndOfLine(kernelContent, "/* default */Tensor actin_Ref", std::string("Tensor actin_Ref = ") + getLayerExpressions(gP.actin_Ref, nLayers) + std::string(";"));

        findAndReplaceToEndOfLine(kernelContent, "/* default */Float xMin","Float xMin = "+std::to_string(gP.posRefMin[0])+std::string(";"));
        findAndReplaceToEndOfLine(kernelContent, "/* default */Float xMax","Float xMax = "+std::to_string(gP.posRefMax[0])+std::string(";"));
        findAndReplaceToEndOfLine(kernelContent, "/* default */Float yMin","Float yMin = "+std::to_string(gP.posRefMin[1])+std::string(";"));
        findAndReplaceToEndOfLine(kernelContent, "/* default */Float yMax","Float yMax = "+std::to_string(gP.posRefMax[1])+std::string(";"));
        findAndReplaceToEndOfLine(kernelContent, "/* default */Float zMin","Float zMin = "+std::to_string(gP.posRefMin[2])+std::string(";"));
        findAndReplaceToEndOfLine(kernelContent, "/* default */Float zMax","Float zMax = "+std::to_string(gP.posRefMax[2])+std::string(";"));

        findAndReplaceToEndOfLine(kernelContent, "/* default */int bcTypeMinAxis0","int bcTypeMinAxis0 = "+std::to_string(gP.bcTypeMinAxis0)+std::string(";"));
        findAndReplaceToEndOfLine(kernelContent, "/* default */int bcTypeMaxAxis0","int bcTypeMaxAxis0 = "+std::to_string(gP.bcTypeMaxAxis0)+std::string(";"));
        findAndReplaceToEndOfLine(kernelContent, "/* default */int bcTypeMinAxis1","int bcTypeMinAxis1 = "+std::to_string(gP.bcTypeMinAxis1)+std::string(";"));
        findAndReplaceToEndOfLine(kernelContent, "/* default */int bcTypeMaxAxis1","int bcTypeMaxAxis1 = "+std::to_string(gP.bcTypeMaxAxis1)+std::string(";"));
        findAndReplaceToEndOfLine(kernelContent, "/* default */int bcTypeMinAxis2","int bcTypeMinAxis2 = "+std::to_string(gP.bcTypeMinAxis2)+std::string(";"));
        findAndReplaceToEndOfLine(kernelContent, "/* default */int bcTypeMaxAxis2","int bcTypeMaxAxis2 = "+std::to_string(gP.bcTypeMaxAxis2)+std::string(";"));

        if (gP.grCoordType == NormalTangent || gP.grCoordType == Cartesian) {
            findAndReplaceToEndOfLine(kernelContent,
                "/* default *///unsigned int bcState","unsigned int bcState = "
                                        "\n\t\t\t(bcTypeMinAxis0==1) * xMinState * (ax0Constraint | ax1Constraint | ax2Constraint) |"
                                        "\n\t\t\t(bcTypeMaxAxis0==1) * xMaxState * (ax0Constraint | ax1Constraint | ax2Constraint) |"
                                        "\n\t\t\t(bcTypeMinAxis0==2) * xMinState * ax0Constraint |"
                                        "\n\t\t\t(bcTypeMaxAxis0==2) * xMaxState * ax0Constraint |"
                                        "\n\t\t\t(bcTypeMinAxis1==1) * yMinState * (ax0Constraint | ax1Constraint | ax2Constraint) |"
                                        "\n\t\t\t(bcTypeMaxAxis1==1) * yMaxState * (ax0Constraint | ax1Constraint | ax2Constraint) |"
                                        "\n\t\t\t(bcTypeMinAxis1==2) * yMinState * ax1Constraint |"
                                        "\n\t\t\t(bcTypeMaxAxis1==2) * yMaxState * ax1Constraint |"
                                        "\n\t\t\t(bcTypeMinAxis2==1) * zMinState * (ax0Constraint | ax1Constraint | ax2Constraint) |"
                                        "\n\t\t\t(bcTypeMaxAxis2==1) * zMaxState * (ax0Constraint | ax1Constraint | ax2Constraint) |"
                                        "\n\t\t\t(bcTypeMinAxis2==2) * zMinState * ax2Constraint |"
                                        "\n\t\t\t(bcTypeMaxAxis2==2) * zMaxState * ax2Constraint;");

            findAndReplaceToEndOfLine(kernelContent,
                "/* default */Tensor proj","Tensor proj = "
                                        "\n\t\t\t xxt * Float((bcState.x & 1u) > 0) +"
                                        "\n\t\t\t yyt * Float((bcState.x & 2u) > 0) +"
                                        "\n\t\t\t zzt * Float((bcState.x & 4u) > 0);");
        }
        if (gP.grCoordType == NormalTangent) {
            findAndReplaceToEndOfLine(kernelContent, "/* default *///Vector n","Vector n = data->normalTetra[i]/data->normalTetra[i].mag();");
            findAndReplaceToEndOfLine(kernelContent, "/* default *///Vector t1","Vector t1 = (Xa-Xb).cross(n);");
            findAndReplaceToEndOfLine(kernelContent, "/* default *///Vector t2","Vector t2 = (Xa-Xc).cross(n);");
            findAndReplaceToEndOfLine(kernelContent, "/* default *///Vector t","Vector t = (t1.mag() > t2.mag()) ? t1/t1.mag() : t2/t2.mag();");
            findAndReplaceToEndOfLine(kernelContent, "/* default *///Vector b","Vector b = n.cross(t)/n.cross(t).mag();");

            findAndReplaceToEndOfLine(kernelContent, "/* default */data->R[i][0]","data->R[i][0] = n[0];");
            findAndReplaceToEndOfLine(kernelContent, "/* default */data->R[i][1]","data->R[i][1] = t[0];");
            findAndReplaceToEndOfLine(kernelContent, "/* default */data->R[i][2]","data->R[i][2] = b[0];");
            findAndReplaceToEndOfLine(kernelContent, "/* default */data->R[i][3]","data->R[i][3] = n[1];");
            findAndReplaceToEndOfLine(kernelContent, "/* default */data->R[i][4]","data->R[i][4] = t[1];");
            findAndReplaceToEndOfLine(kernelContent, "/* default */data->R[i][5]","data->R[i][5] = b[1];");
            findAndReplaceToEndOfLine(kernelContent, "/* default */data->R[i][6]","data->R[i][6] = n[2];");
            findAndReplaceToEndOfLine(kernelContent, "/* default */data->R[i][7]","data->R[i][7] = t[2];");
            findAndReplaceToEndOfLine(kernelContent, "/* default */data->R[i][8]","data->R[i][8] = b[2];");
        }

        if (gP.grCoordType == CylindricalZ) {
            findAndReplaceToEndOfLine(kernelContent, "/* default *///Float R","Float R = sqrt(X*X + Y*Y);");
            findAndReplaceToEndOfLine(kernelContent, "/* default *///Float Theta","Float Theta = atan2(Y, X);");

            findAndReplaceToEndOfLine(kernelContent, "/* default */data->R[i][0]","data->R[i][0] = cos(Theta);");
            findAndReplaceToEndOfLine(kernelContent, "/* default */data->R[i][1]","data->R[i][1] =-sin(Theta);");
            findAndReplaceToEndOfLine(kernelContent, "/* default */data->R[i][2]","data->R[i][2] = 0.0;");
            findAndReplaceToEndOfLine(kernelContent, "/* default */data->R[i][3]","data->R[i][3] = sin(Theta);");
            findAndReplaceToEndOfLine(kernelContent, "/* default */data->R[i][4]","data->R[i][4] = cos(Theta);");
            findAndReplaceToEndOfLine(kernelContent, "/* default */data->R[i][5]","data->R[i][5] = 0.0;");
            findAndReplaceToEndOfLine(kernelContent, "/* default */data->R[i][6]","data->R[i][6] = 0.0;");
            findAndReplaceToEndOfLine(kernelContent, "/* default */data->R[i][7]","data->R[i][7] = 0.0;");
            findAndReplaceToEndOfLine(kernelContent, "/* default */data->R[i][8]","data->R[i][8] = 1.0;");

            findAndReplaceToEndOfLine(kernelContent, "/* default *///Float rMin","Float rMin = "+std::to_string(gP.rRefMin)+std::string(";"));
            findAndReplaceToEndOfLine(kernelContent, "/* default *///Float rMax","Float rMax = "+std::to_string(gP.rRefMax)+std::string(";"));
            findAndReplaceToEndOfLine(kernelContent, "/* default *///const unsigned int rMinState","const unsigned int rMinState = fabs(R - rMin) < tol;");
            findAndReplaceToEndOfLine(kernelContent, "/* default *///const unsigned int rMaxState","const unsigned int rMaxState = fabs(R - rMax) < tol;");
            findAndReplaceToEndOfLine(kernelContent,
                "/* default *///unsigned int bcState","unsigned int bcState = "
                                        "\n\t\t\t(bcTypeMinAxis0==1) * rMinState * (ax0Constraint | ax1Constraint | ax2Constraint) |"
                                        "\n\t\t\t(bcTypeMaxAxis0==1) * rMaxState * (ax0Constraint | ax1Constraint | ax2Constraint) |"
                                        "\n\t\t\t(bcTypeMinAxis0==2) * rMinState * ax0Constraint |"
                                        "\n\t\t\t(bcTypeMaxAxis0==2) * rMaxState * ax0Constraint |"
                                        "\n\t\t\t(bcTypeMinAxis1==1) * zMinState * (ax0Constraint | ax1Constraint | ax2Constraint) |"
                                        "\n\t\t\t(bcTypeMaxAxis1==1) * zMaxState * (ax0Constraint | ax1Constraint | ax2Constraint) |"
                                        "\n\t\t\t(bcTypeMinAxis1==2) * zMinState * ax1Constraint |"
                                        "\n\t\t\t(bcTypeMaxAxis1==2) * zMaxState * ax1Constraint;");

            findAndReplaceToEndOfLine(kernelContent, "/* default *///Vector r","Vector r = Vector(x, y, 0.0).safe_normal();");
            findAndReplaceToEndOfLine(kernelContent, "/* default *///Tensor rrt","Tensor rrt(r);");
            findAndReplaceToEndOfLine(kernelContent, "/* default *///Tensor ppt","Tensor ppt = Tensor::eye() - rrt - zzt;");
            findAndReplaceToEndOfLine(kernelContent,
                "/* default */Tensor proj","Tensor proj = "
                                        "\n\t\t\t rrt * Float((bcState.x & 1u) > 0) +"
                                        "\n\t\t\t zzt * Float((bcState.x & 2u) > 0) +"
                                        "\n\t\t\t ppt * Float((bcState.x & 4u) > 0);");
        }

        if (gP.grCoordType == CylindricalY) {
            findAndReplaceToEndOfLine(kernelContent, "/* default *///Float R","Float R = sqrt(X*X + Z*Z);");
            findAndReplaceToEndOfLine(kernelContent, "/* default *///Float Theta","Float Theta = atan2(Z, X);");

            findAndReplaceToEndOfLine(kernelContent, "/* default */data->R[i][0]","data->R[i][0] = cos(Theta);");
            findAndReplaceToEndOfLine(kernelContent, "/* default */data->R[i][1]","data->R[i][1] = 0.0;");
            findAndReplaceToEndOfLine(kernelContent, "/* default */data->R[i][2]","data->R[i][2] =-sin(Theta);");
            findAndReplaceToEndOfLine(kernelContent, "/* default */data->R[i][3]","data->R[i][3] = 0.0;");
            findAndReplaceToEndOfLine(kernelContent, "/* default */data->R[i][4]","data->R[i][4] = 1.0;");
            findAndReplaceToEndOfLine(kernelContent, "/* default */data->R[i][5]","data->R[i][5] = 0.0;");
            findAndReplaceToEndOfLine(kernelContent, "/* default */data->R[i][6]","data->R[i][6] = sin(Theta);");
            findAndReplaceToEndOfLine(kernelContent, "/* default */data->R[i][7]","data->R[i][7] = 0.0;");
            findAndReplaceToEndOfLine(kernelContent, "/* default */data->R[i][8]","data->R[i][8] = cos(Theta);");


            findAndReplaceToEndOfLine(kernelContent, "/* default *///Float rMin","Float rMin = "+std::to_string(gP.rRefMin)+std::string(";"));
            findAndReplaceToEndOfLine(kernelContent, "/* default *///Float rMax","Float rMax = "+std::to_string(gP.rRefMax)+std::string(";"));
            findAndReplaceToEndOfLine(kernelContent, "/* default *///const unsigned int rMinState","const unsigned int rMinState = fabs(R - rMin) < tol;");
            findAndReplaceToEndOfLine(kernelContent, "/* default *///const unsigned int rMaxState","const unsigned int rMaxState = fabs(R - rMax) < tol;");
            findAndReplaceToEndOfLine(kernelContent,
                "/* default *///unsigned int bcState","unsigned int bcState = "
                                        "\n\t\t\t(bcTypeMinAxis0==1) * rMinState * (ax0Constraint | ax1Constraint | ax2Constraint) |"
                                        "\n\t\t\t(bcTypeMaxAxis0==1) * rMaxState * (ax0Constraint | ax1Constraint | ax2Constraint) |"
                                        "\n\t\t\t(bcTypeMinAxis0==2) * rMinState * ax0Constraint |"
                                        "\n\t\t\t(bcTypeMaxAxis0==2) * rMaxState * ax0Constraint |"
                                        "\n\t\t\t(bcTypeMinAxis1==1) * yMinState * (ax0Constraint | ax1Constraint | ax2Constraint) |"
                                        "\n\t\t\t(bcTypeMaxAxis1==1) * yMaxState * (ax0Constraint | ax1Constraint | ax2Constraint) |"
                                        "\n\t\t\t(bcTypeMinAxis1==2) * yMinState * ax1Constraint |"
                                        "\n\t\t\t(bcTypeMaxAxis1==2) * yMaxState * ax1Constraint;");

            findAndReplaceToEndOfLine(kernelContent, "/* default *///Vector r","Vector r = Vector(x, 0.0, z).safe_normal();");
            findAndReplaceToEndOfLine(kernelContent, "/* default *///Tensor rrt","Tensor rrt(r);");
            findAndReplaceToEndOfLine(kernelContent, "/* default *///Tensor ppt","Tensor ppt = Tensor::eye() - rrt - yyt;");
            findAndReplaceToEndOfLine(kernelContent,
                "/* default */Tensor proj","Tensor proj = "
                                        "\n\t\t\t rrt * Float((bcState.x & 1u) > 0) +"
                                        "\n\t\t\t yyt * Float((bcState.x & 2u) > 0) +"
                                        "\n\t\t\t ppt * Float((bcState.x & 4u) > 0);");

        }

        if (gP.grCoordType == ConeAdapted) {
            findAndReplaceToEndOfLine(kernelContent, "/* default *///Float R","Float R = sqrt(X*X + Z*Z);");
            findAndReplaceToEndOfLine(kernelContent, "/* default *///Float Theta","Float Theta = atan2(Z, X);");
            findAndReplaceToEndOfLine(kernelContent, "/* default *///Float hapex","Float hapex = " + std::to_string(gP.apex*0.5) + std::string(";"));

            findAndReplaceToEndOfLine(kernelContent, "/* default */data->R[i][0]","data->R[i][0] = cos(hapex) * cos(Theta);");
            findAndReplaceToEndOfLine(kernelContent, "/* default */data->R[i][1]","data->R[i][1] =-sin(hapex) * cos(Theta);");
            findAndReplaceToEndOfLine(kernelContent, "/* default */data->R[i][2]","data->R[i][2] =-sin(Theta);");
            findAndReplaceToEndOfLine(kernelContent, "/* default */data->R[i][3]","data->R[i][3] = sin(hapex);");
            findAndReplaceToEndOfLine(kernelContent, "/* default */data->R[i][4]","data->R[i][4] = cos(hapex);");
            findAndReplaceToEndOfLine(kernelContent, "/* default */data->R[i][5]","data->R[i][5] = 0.0;");
            findAndReplaceToEndOfLine(kernelContent, "/* default */data->R[i][6]","data->R[i][6] = cos(hapex) * sin(Theta);");
            findAndReplaceToEndOfLine(kernelContent, "/* default */data->R[i][7]","data->R[i][7] =-sin(hapex) * sin(Theta);");
            findAndReplaceToEndOfLine(kernelContent, "/* default */data->R[i][8]","data->R[i][8] = cos(Theta);");

            findAndReplaceToEndOfLine(kernelContent, "/* default *///Float rMin","Float rMin = max("+std::to_string(gP.rRefMin)+std::string(" - Y * tan(hapex),Float(0.0));"));
            findAndReplaceToEndOfLine(kernelContent, "/* default *///Float rMax","Float rMax = max("+std::to_string(gP.rRefMax)+std::string(" - Y * tan(hapex),Float(0.0));"));
            findAndReplaceToEndOfLine(kernelContent, "/* default *///const unsigned int rMinState","const unsigned int rMinState = fabs(R - rMin) < tol;");
            findAndReplaceToEndOfLine(kernelContent, "/* default *///const unsigned int rMaxState","const unsigned int rMaxState = fabs(R - rMax) < tol;");
            findAndReplaceToEndOfLine(kernelContent,
                "/* default *///unsigned int bcState","unsigned int bcState = "
                                        "\n\t\t\t(bcTypeMinAxis0==1) * rMinState * (ax0Constraint | ax1Constraint | ax2Constraint) |"
                                        "\n\t\t\t(bcTypeMaxAxis0==1) * rMaxState * (ax0Constraint | ax1Constraint | ax2Constraint) |"
                                        "\n\t\t\t(bcTypeMinAxis0==2) * rMinState * ax0Constraint |"
                                        "\n\t\t\t(bcTypeMaxAxis0==2) * rMaxState * ax0Constraint |"
                                        "\n\t\t\t(bcTypeMinAxis1==1) * yMinState * (ax0Constraint | ax1Constraint | ax2Constraint) |"
                                        "\n\t\t\t(bcTypeMaxAxis1==1) * yMaxState * (ax0Constraint | ax1Constraint | ax2Constraint) |"
                                        "\n\t\t\t(bcTypeMinAxis1==2) * yMinState * ax1Constraint |"
                                        "\n\t\t\t(bcTypeMaxAxis1==2) * yMaxState * ax1Constraint;");

            findAndReplaceToEndOfLine(kernelContent, "/* default *///Float theta","Float theta = atan2(z, x);");
            findAndReplaceToEndOfLine(kernelContent, "/* default *///Tensor nnt","Tensor nnt(cos(hapex) * cos(theta), sin(hapex), cos(hapex) * sin(theta));");
            findAndReplaceToEndOfLine(kernelContent, "/* default *///Tensor aat","Tensor aat(-sin(hapex) * cos(theta), cos(hapex), -sin(hapex) * sin(theta)); if ((bcState.x & 1u) == 0 && (bcState.x & 2u) > 0 && (bcState.x & 4u) == 0) { aat = yyt; };");
            findAndReplaceToEndOfLine(kernelContent, "/* default *///Tensor ppt","Tensor ppt = Tensor::eye() - nnt - aat;");
            findAndReplaceToEndOfLine(kernelContent,
                "/* default */Tensor proj","Tensor proj = "
                                        "\n\t\t\t nnt * Float((bcState.x & 1u) > 0) +"
                                        "\n\t\t\t aat * Float((bcState.x & 2u) > 0) +"
                                        "\n\t\t\t ppt * Float((bcState.x & 4u) > 0);");
        }

        if (gP.grCoordType == Spherical) {
            findAndReplaceToEndOfLine(kernelContent, "/* default *///Float R","Float R = sqrt(X*X + Y*Y + Z*Z);");
            findAndReplaceToEndOfLine(kernelContent, "/* default *///Float Theta","Float Theta = atan2(sqrt(X*X+Y*Y), Z);");
            findAndReplaceToEndOfLine(kernelContent, "/* default *///Float Phi","Float Phi = atan2(Y, X);");

            findAndReplaceToEndOfLine(kernelContent, "/* default */data->R[i][0]","data->R[i][0] = sin(Theta)*cos(Phi);");
            findAndReplaceToEndOfLine(kernelContent, "/* default */data->R[i][1]","data->R[i][1] = cos(Theta)*cos(Phi);");
            findAndReplaceToEndOfLine(kernelContent, "/* default */data->R[i][2]","data->R[i][2] =-sin(Phi);");
            findAndReplaceToEndOfLine(kernelContent, "/* default */data->R[i][3]","data->R[i][3] = sin(Theta)*sin(Phi);");
            findAndReplaceToEndOfLine(kernelContent, "/* default */data->R[i][4]","data->R[i][4] = cos(Theta)*sin(Phi);");
            findAndReplaceToEndOfLine(kernelContent, "/* default */data->R[i][5]","data->R[i][5] = cos(Phi);");
            findAndReplaceToEndOfLine(kernelContent, "/* default */data->R[i][6]","data->R[i][6] = cos(Theta);");
            findAndReplaceToEndOfLine(kernelContent, "/* default */data->R[i][7]","data->R[i][7] =-sin(Theta);");
            findAndReplaceToEndOfLine(kernelContent, "/* default */data->R[i][8]","data->R[i][8] = 0.0;");

            findAndReplaceToEndOfLine(kernelContent, "/* default *///Float rMin","Float rMin = "+std::to_string(gP.rRefMin)+std::string(";"));
            findAndReplaceToEndOfLine(kernelContent, "/* default *///Float rMax","Float rMax = "+std::to_string(gP.rRefMax)+std::string(";"));
            findAndReplaceToEndOfLine(kernelContent, "/* default *///const unsigned int rMinState","const unsigned int rMinState = fabs(R - rMin) < tol;");
            findAndReplaceToEndOfLine(kernelContent, "/* default *///const unsigned int rMaxState","const unsigned int rMaxState = fabs(R - rMax) < tol;");
            findAndReplaceToEndOfLine(kernelContent,
                "/* default *///unsigned int bcState","unsigned int bcState = "
                                        "\n\t\t\t(bcTypeMinAxis0==1) * rMinState * (ax0Constraint | ax1Constraint | ax2Constraint) |"
                                        "\n\t\t\t(bcTypeMaxAxis0==1) * rMaxState * (ax0Constraint | ax1Constraint | ax2Constraint) |"
                                        "\n\t\t\t(bcTypeMinAxis0==2) * rMinState * ax0Constraint |"
                                        "\n\t\t\t(bcTypeMaxAxis0==2) * rMaxState * ax0Constraint;");

            findAndReplaceToEndOfLine(kernelContent, "/* default *///Vector r","Vector r = Vector(x, y, z).safe_normal();");
            findAndReplaceToEndOfLine(kernelContent, "/* default *///Tensor rrt","Tensor rrt(r);");
            findAndReplaceToEndOfLine(kernelContent, "/* default *///Float phi","Float phi = atan2(y, x);");
            findAndReplaceToEndOfLine(kernelContent, "/* default *///Tensor ppt","Tensor ppt(-sin(phi),cos(phi),0.0);");
            findAndReplaceToEndOfLine(kernelContent,
                "/* default */Tensor proj","Tensor proj = "
                                        "\n\t\t\t rrt * Float((bcState.x & 1u) > 0) +"
                                        "\n\t\t\t ppt * Float((bcState.x & 2u) > 0) +"
                                        "\n\t\t\t (Tensor::eye() - rrt - ppt) * Float((bcState.x & 4u) > 0);");
        }

        if (gP.grCoordType == Toroidal) {
            findAndReplaceToEndOfLine(kernelContent, "/* default *///Float R","Float R = sqrt((sqrt(X*X+Y*Y)- Float("+std::to_string(gP.RTorus)+")) * (sqrt(X*X+Y*Y)-Float("+std::to_string(gP.RTorus)+")) + Z*Z);");
            findAndReplaceToEndOfLine(kernelContent, "/* default *///Float Theta","Float Theta = atan2(sqrt(X*X+Y*Y) - Float("+std::to_string(gP.RTorus)+"), Z);");
            findAndReplaceToEndOfLine(kernelContent, "/* default *///Float Phi",std::string("Float Phi = atan2(Y, X);"));

            findAndReplaceToEndOfLine(kernelContent, "/* default */data->R[i][0]","data->R[i][0] = sin(Theta)*cos(Phi);");
            findAndReplaceToEndOfLine(kernelContent, "/* default */data->R[i][1]","data->R[i][1] = cos(Theta)*cos(Phi);");
            findAndReplaceToEndOfLine(kernelContent, "/* default */data->R[i][2]","data->R[i][2] =-sin(Phi);");
            findAndReplaceToEndOfLine(kernelContent, "/* default */data->R[i][3]","data->R[i][3] = sin(Theta)*sin(Phi);");
            findAndReplaceToEndOfLine(kernelContent, "/* default */data->R[i][4]","data->R[i][4] = cos(Theta)*sin(Phi);");
            findAndReplaceToEndOfLine(kernelContent, "/* default */data->R[i][5]","data->R[i][5] = cos(Phi);");
            findAndReplaceToEndOfLine(kernelContent, "/* default */data->R[i][6]","data->R[i][6] = cos(Theta);");
            findAndReplaceToEndOfLine(kernelContent, "/* default */data->R[i][7]","data->R[i][7] =-sin(Theta);");
            findAndReplaceToEndOfLine(kernelContent, "/* default */data->R[i][8]","data->R[i][8] = 0.0;");

            findAndReplaceToEndOfLine(kernelContent, "/* default *///Float rMin","Float rMin = "+std::to_string(gP.rRefMin)+std::string(";"));
            findAndReplaceToEndOfLine(kernelContent, "/* default *///Float rMax","Float rMax = "+std::to_string(gP.rRefMax)+std::string(";"));
            findAndReplaceToEndOfLine(kernelContent, "/* default *///const unsigned int rMinState","const unsigned int rMinState = fabs(R - rMin) < tol;");
            findAndReplaceToEndOfLine(kernelContent, "/* default *///const unsigned int rMaxState","const unsigned int rMaxState = fabs(R - rMax) < tol;");
            findAndReplaceToEndOfLine(kernelContent,
                "/* default *///unsigned int bcState","unsigned int bcState = "
                                        "\n\t\t\t(bcTypeMinAxis0==1) * rMinState * (ax0Constraint | ax1Constraint | ax2Constraint) |"
                                        "\n\t\t\t(bcTypeMaxAxis0==1) * rMaxState * (ax0Constraint | ax1Constraint | ax2Constraint) |"
                                        "\n\t\t\t(bcTypeMinAxis0==2) * rMinState * ax0Constraint |"
                                        "\n\t\t\t(bcTypeMaxAxis0==2) * rMaxState * ax0Constraint;");

            //Vector pos = data->pos[i];
            //Float phi = atan2(pos[1],pos[0]);
            //Float theta = atan2(sqrt(pos[0]*pos[0]+pos[1]*pos[1]) - gP.RTorus, pos[2]);
            //Vector r = Vector(cos(phi)*sin(theta), sin(phi)*sin(theta), cos(theta));
            //Tensor rrt(r);
            //Tensor ppt(-sin(phi),cos(phi),0.0);
            //proj = rrt * Float((data->bcState[i].x & 1u) > 0) +
            //       ppt * Float((data->bcState[i].x & 2u) > 0) +
            //       (Tensor::eye() - rrt - ppt) * Float((data->bcState[i].x & 4u) > 0);

            findAndReplaceToEndOfLine(kernelContent, "/* default *///Float phi","Float phi = atan2(y, x);");
            findAndReplaceToEndOfLine(kernelContent, "/* default *///Float theta","Float theta = atan2(sqrt(x*x+y*y) - Float("+std::to_string(gP.RTorus)+"), z);");
            findAndReplaceToEndOfLine(kernelContent, "/* default *///Vector r","Vector r(cos(phi)*sin(theta), sin(phi)*sin(theta), cos(theta));");
            findAndReplaceToEndOfLine(kernelContent, "/* default *///Tensor rrt","Tensor rrt(r);");
            findAndReplaceToEndOfLine(kernelContent, "/* default *///Tensor ppt","Tensor ppt(-sin(phi),cos(phi),0.0);");
            findAndReplaceToEndOfLine(kernelContent,
                "/* default */Tensor proj","Tensor proj = "
                                        "\n\t\t\t rrt * Float((bcState.x & 1u) > 0) +"
                                        "\n\t\t\t ppt * Float((bcState.x & 2u) > 0) +"
                                        "\n\t\t\t (Tensor::eye() - rrt - ppt) * Float((bcState.x & 4u) > 0);");
        }

        saveFile(pathToKernel + "/" + outputName,kernelContent);
    }

    bool compile() {

        std::string execPath = getExecutablePath();
        std::string execDir  = execPath.substr(0, execPath.find_last_of("/\\"));
        std::cout<<"executable location "<<execDir<<std::endl;


        // Compile and load nvrtc kernels
        int kernelCount = 0;
        std::string pathToKernel = execDir+"/"+"../src/nvrtc_kernels/";

        generate_nvrtc_kernel_source(pathToKernel,stringReplace(compute_force_nvrtc.name, "_nvrtc", "_default")+".cu",compute_force_nvrtc.name+".cu", mesh.nlay);
        generate_nvrtc_kernel_source(pathToKernel,stringReplace(compute_orthogonal_basis_nvrtc.name, "_nvrtc", "_default")+".cu",compute_orthogonal_basis_nvrtc.name+".cu", mesh.nlay);
        generate_nvrtc_kernel_source(pathToKernel,stringReplace(compute_bids_nvrtc.name, "_nvrtc", "_default")+".cu",compute_bids_nvrtc.name+".cu", mesh.nlay);
        generate_nvrtc_kernel_source(pathToKernel,stringReplace(enforceBC_nvrtc.name, "_nvrtc", "_default")+".cu",enforceBC_nvrtc.name+".cu", mesh.nlay);
        generate_nvrtc_kernel_source(pathToKernel,stringReplace(compute_critical_timestep_nvrtc.name, "_nvrtc", "_default")+".cu",compute_critical_timestep_nvrtc.name+".cu", mesh.nlay);
        generate_nvrtc_kernel_source(pathToKernel,stringReplace(compute_for_vis_nvrtc.name, "_nvrtc", "_default")+".cu",compute_for_vis_nvrtc.name+".cu", mesh.nlay);

        _LOAD_NVRTC(compute_force_nvrtc,             kernelCount++, pathToKernel);
        _LOAD_NVRTC(compute_orthogonal_basis_nvrtc,  kernelCount++, pathToKernel);
        _LOAD_NVRTC(compute_bids_nvrtc,              kernelCount++, pathToKernel);
        _LOAD_NVRTC(enforceBC_nvrtc,                 kernelCount++, pathToKernel);
        _LOAD_NVRTC(compute_critical_timestep_nvrtc, kernelCount++, pathToKernel);
        _LOAD_NVRTC(compute_for_vis_nvrtc,           kernelCount++, pathToKernel);

        return compute_force_nvrtc.kernel != nullptr &&
            compute_orthogonal_basis_nvrtc.kernel != nullptr &&
            compute_bids_nvrtc.kernel != nullptr &&
            enforceBC_nvrtc.kernel != nullptr &&
            compute_for_vis_nvrtc.kernel != nullptr &&
            compute_critical_timestep_nvrtc.kernel != nullptr;
    }
    void init(bool fromMesh) {
        if (fromMesh) {
            data.init(mesh);
            dataPtr = new DeviceDataPtrManaged;
            data.set(*dataPtr);
        }

        data.vel.assign(mesh.nver, Vector(0.0,0.0,0.0));
        data.Fg.assign(mesh.ntet, Tensor(1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0));
        data.Fp.assign(mesh.ntet, Tensor(1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0));
        data.stress.assign(mesh.ntet, Tensor(0.0));
        globalTime = 0.0f;
        frames = 0;

        copy_eigen_to_device(mesh.pos, data.pos);
        if (mesh.posRef.rows() == mesh.nver)
            copy_eigen_to_device(mesh.posRef, data.posRef);
        else
            copy_eigen_to_device(mesh.pos, data.posRef);
        copy_eigen_to_device(mesh.vel, data.vel);


        copy_eigen_to_device(mesh.Fg, data.Fg);
        copy_eigen_to_device(mesh.Fp, data.Fp);

        save_to_snapshot();
        nLayers = mesh.nlay;
        globalTime = 0.0f;
//        growthRateFactor = 1.0f;
        frames = 0;

        get_domain_limits(data.pos);
        gP.posRefMin = minPos;
        gP.posRefMax = maxPos;
        if (gP.grCoordType != CoordinateSystem::ConeAdapted) {
            gP.rRefMin = minR;
            gP.rRefMax = maxR;
        }
        charLRef = charL;

        ///////////////////////////////////////////
        ScalarArrayEigen distance = getSignedDist(mesh.pos,mesh.pos,mesh.tri);
        ScalarArrayDev distanceDev(mesh.nver,0.0);
        thrust::copy(reinterpret_cast<Float*>(distance.data()), reinterpret_cast<Float*>(distance.data())+mesh.nver, distanceDev.begin());
        _LAUNCH(mesh.ntet, 256, compute_normal_from_dist) (dataPtr, thrust::raw_pointer_cast(distanceDev.data()), mesh.ntet);
        cudaDeviceSynchronize();
        _LAUNCH_NVRTC(mesh.ntet, 256, compute_orthogonal_basis_nvrtc.kernel, {&dataPtr, &mesh.ntet});
        // _LAUNCH(mesh.ntet, 256, compute_orthogonal_basis) (dataPtr, mesh.ntet);
        cudaDeviceSynchronize();
        ///////////////////////////////////////////

        data.vol.assign(mesh.nver, 0.0f);
        cudaDeviceSynchronize();
        // _LAUNCH(mesh.nver, 256, compute_bids) (dataPtr, bcTol*spacing, mesh.nver);
        Float tempFloat = bcTol*spacing;
        _LAUNCH_NVRTC(mesh.nver, 256, compute_bids_nvrtc.kernel, {&dataPtr, &tempFloat, &mesh.nver});
        // _LUNCH(mesh.ntet, 256, compute_mechanical_parameters) (dataPtr, minPosRef, maxPosRef, mesh.ntet);
        _LAUNCH(mesh.ntet, 256, compute_volume) (dataPtr, mesh.ntet);
        cudaDeviceSynchronize();

        ScalarArrayDev tempArray = data.tetVol;
        thrust::sort(tempArray.begin(), tempArray.end());
        spacing = std::pow(tempArray[static_cast<int>(0.01 * tempArray.size())],Float(1.0/3.0));
        Float minVol = tempArray.front();
        Float maxVol = tempArray.back();

        std::cout << "average spacing = "<< spacing << ",  min / max volume = " << minVol << " / " << maxVol << std::endl;
        cudaDeviceSynchronize();

        data.force.assign(mesh.nver,Vector(0.0f));
        _LAUNCH(mesh.nnbd, 128, read_boundary_nodes) (dataPtr, spacing, mesh.nnbd);
        _LAUNCH(mesh.ntri, 128, read_boundary_faces) (dataPtr, spacing, mesh.ntri);
        octreeSearch(data, minPos - Vector(spacing * 10.0), maxPos + Vector(spacing * 10.0));
        kinEnergy = 0;
        for (int i=0; i<averageInterval; i++)
            enDiff[i] = INFINITY;
    };


    bool octreeSearch(DeviceData& data, const Vector& _minPos, const Vector& _maxPos) {

        Encoder32 encoder(_minPos, _maxPos);

        thrust::device_vector<UIntM> mKeyN(data.boundaryNodePos.size(),0);
        thrust::device_vector<UIntM> mKeyF(data.boundaryFacePos.size(),0);
        thrust::transform(data.boundaryNodePos.begin(), data.boundaryNodePos.end(), mKeyN.begin(), encoder);
        thrust::transform(data.boundaryFacePos.begin(), data.boundaryFacePos.end(), mKeyF.begin(), encoder);

        thrust::device_vector<int> seqIdN(data.boundaryNodePos.size());
        thrust::device_vector<int> seqIdF(data.boundaryFacePos.size());
        thrust::sequence(seqIdN.begin(),seqIdN.end());
        thrust::sequence(seqIdF.begin(),seqIdF.end());
        thrust::sort_by_key(mKeyN.begin(),mKeyN.end(), seqIdN.begin());
        thrust::sort_by_key(mKeyF.begin(),mKeyF.end(), seqIdF.begin());

        _REMAP(seqIdN,data.boundaryNodePos);
        _REMAP(seqIdN,data.boundaryNodeR);
        _REMAP(seqIdN,data.boundaryNodeIds);

        _REMAP(seqIdF,data.boundaryFacePos);
        _REMAP(seqIdF,data.boundaryFaceR);
        _REMAP(seqIdF,data.tri);

        thrust::device_vector<Int> bNgbIds;    // This array contains the index of the branches neighbors. Branches are recognized as neighbor if their bounding volume intersects.
        thrust::device_vector<Int> bNgbSize;   // This array contains the the number of neighbors for each branch.
        thrust::device_vector<Int> bNgbOffset; // This array contains the exclusive summation of the neighbors number for each branch.


        Octree treeF(data.boundaryFacePos,data.boundaryFaceR,mKeyF,1000,8);
        Octree treeN(data.boundaryNodePos,data.boundaryNodeR,mKeyN,1000,8);
        findBranchsNgb(treeN, treeF,bNgbIds,bNgbSize,bNgbOffset);

       int nBranches = treeN.bCode.size();
       int num = data.boundaryNodePos.size();
        size_t nP1Max = (*thrust::max_element(treeN.bPNum.begin(), treeN.bPNum.end()));
        size_t nP2Max = (*thrust::max_element(treeF.bPNum.begin(), treeF.bPNum.end())) * (*thrust::max_element(bNgbSize.begin(), bNgbSize.end()));
        nP2Max = std::max(nP2Max, (nP1Max*MAX_NGB));

        Float allocMemMB = (4.0*(nP1Max*nP2Max+3.0*nP1Max+nP2Max) + 8.0*(nP1Max+nP2Max))*1e-6*1.25;
        if (allocMemMB > 24000.0){
            return false;
        }

        thrust::device_vector<Int> localNgbIds(nP1Max*nP2Max,-1);
        thrust::device_vector<Int> particles1(nP1Max);
        thrust::device_vector<Int> particles2(nP2Max);

        thrust::device_vector<Int> localNgbSize(nP1Max);
        thrust::device_vector<Int> localNgbOffset(nP1Max);

        thrust::device_vector<Vector> posP1(nP1Max);
        thrust::device_vector<Vector> posP2(nP2Max);
        thrust::device_vector<Float>  hP1(nP1Max);
        thrust::device_vector<Float>  hP2(nP2Max);

        thrust::host_vector<Int> hBNgbSize(bNgbSize);
        thrust::host_vector<Int> hBNgbIds(bNgbIds);
        thrust::host_vector<Int> hBNgbOffset(bNgbOffset);
//        thrust::host_vector<Int> hBCode(treeN.bCode);
        thrust::host_vector<Int> hBPOffsetN(treeN.bPOffset);
        thrust::host_vector<Int> hBPNumN(treeN.bPNum);
        thrust::host_vector<Int> hBPOffsetF(treeF.bPOffset);
        thrust::host_vector<Int> hBPNumF(treeF.bPNum);

        data.ngb_list.resize(num * MAX_NGB);
        data.ngb_size.resize(num);
        data.ngb_offset.resize(num);

        size_t ngbListSize = 0;

        Int temp1 = 0;
        Int temp2 = 0;

        for (Int branch = 0; branch < nBranches; branch++) {
            Int offset = hBPOffsetN[branch];
            Int nP1 = hBPNumN[branch];
            Int bNgbSize = hBNgbSize[branch];
            Int bNgbOffset = hBNgbOffset[branch];

            thrust::sequence(particles1.begin(),particles1.begin() + nP1, offset);

            Int nP2 = 0;

            for (Int i = 0; i < bNgbSize; i++) {
                Int bId = hBNgbIds[bNgbOffset + i];
                Int offset2 = hBPOffsetF[bId];
                thrust::sequence(particles2.begin() + nP2, particles2.begin() + nP2 + hBPNumF[bId], offset2);
                nP2 += hBPNumF[bId];
            }

            if (nP2==0)
                continue;

            thrust::fill_n(localNgbSize.begin(), nP1, 0);
            thrust::fill_n(localNgbIds.begin(), nP1 * MAX_NGB, num+1);

            thrust::gather(particles1.begin(), particles1.begin() + nP1, data.boundaryNodePos.begin(), posP1.begin());
            thrust::gather(particles2.begin(), particles2.begin() + nP2, data.boundaryFacePos.begin(), posP2.begin());
            thrust::gather(particles1.begin(), particles1.begin() + nP1, data.boundaryNodeR.begin(), hP1.begin());
            thrust::gather(particles2.begin(), particles2.begin() + nP2, data.boundaryFaceR.begin(), hP2.begin());

            _LAUNCH(nP2,64,distCheckKernel)(thrust::raw_pointer_cast(posP1.data()),
                    thrust::raw_pointer_cast(posP2.data()),
                    thrust::raw_pointer_cast(hP1.data()),
                    thrust::raw_pointer_cast(hP2.data()),
                    nP1, nP2, MAX_NGB,
                    thrust::raw_pointer_cast(particles2.data()),
                    thrust::raw_pointer_cast(localNgbSize.data()),
                    thrust::raw_pointer_cast(localNgbIds.data()));
            cudaDeviceSynchronize();

            thrust::exclusive_scan(localNgbSize.begin(),localNgbSize.end(), localNgbOffset.begin());
            size_t sz = localNgbOffset[nP1-1] + localNgbSize[nP1-1];
            ngbListSize += sz;

            _LAUNCH(nP1,64,copyToNgbList)(thrust::raw_pointer_cast(localNgbSize.data()),
                    thrust::raw_pointer_cast(localNgbOffset.data()),
                    thrust::raw_pointer_cast(localNgbIds.data()),
                    thrust::raw_pointer_cast(particles1.data()),
                    nP1,MAX_NGB,
                    thrust::raw_pointer_cast(&data.ngb_list[ngbListSize - sz]));
            cudaDeviceSynchronize();

            thrust::copy(localNgbSize.begin(), localNgbSize.begin() + nP1, data.ngb_size.begin() + temp1);

            temp1+=nP1;
            temp2+=nP2;
        }

        data.ngb_list.resize(ngbListSize);
        thrust::exclusive_scan(data.ngb_size.begin(),data.ngb_size.end(), data.ngb_offset.begin());

        if (temp1 != num)
            _ERROR_MESSAGE("Search error "<<temp1<<" != "<<num);
        int maxNgbTemp = *thrust::max_element(data.ngb_size.begin(),data.ngb_size.end());
        if (maxNgbTemp > MAX_NGB)
            _ERROR_MESSAGE("ngbSize, "<<maxNgbTemp<<", exceeds "<<MAX_NGB);

        cudaDeviceSynchronize();
        return true;
   }


    void simulate() {
        cudaDeviceSynchronize();
        _LAUNCH(mesh.nnbd, 128, read_boundary_nodes) (dataPtr, spacing, mesh.nnbd);
        if (simIter % searchIter == 0) {
            _LAUNCH(mesh.ntri, 128, read_boundary_faces) (dataPtr, spacing, mesh.ntri);
            get_domain_limits(data.pos);
            simDiverged = !octreeSearch(data, minPos - Vector(spacing * 10.0f), maxPos + Vector(spacing * 10.0f));
        }

        if (simDiverged)
            return;

        data.vol.assign(mesh.nver, 0.0f);
        _LAUNCH(mesh.ntet, 256, compute_volume)        (dataPtr, mesh.ntet);

        cudaDeviceSynchronize();

        data.vGradNode.assign(mesh.nver, Tensor(0.0));
        _LAUNCH(mesh.ntet, 256, compute_vgrad_node) (dataPtr, mesh.ntet);
        cudaDeviceSynchronize();


        if (simIter % 50 == 0) {
            _LAUNCH_NVRTC(mesh.ntet, 32, compute_critical_timestep_nvrtc.kernel, {&dataPtr, &globalTime, &mesh.ntet});
            cudaDeviceSynchronize();
            maxDt = *thrust::min_element(data.dt.begin(), data.dt.end());
        }
        dt = maxDt * timeFactor;

        data.force.assign(mesh.nver, Vector(0.0));
        cudaDeviceSynchronize();
        _LAUNCH_NVRTC(mesh.ntet, 32, compute_force_nvrtc.kernel, {&dataPtr, &dt, &globalTime, &mesh.ntet});
        cudaDeviceSynchronize();
        _LAUNCH(mesh.nnbd, 64, compute_contact_force) (dataPtr, spacing, contactFact, contactFactVel, repThickness, mesh.nnbd);
        cudaDeviceSynchronize();
        _LAUNCH(mesh.nver, 256, update_vel) (dataPtr, dt, mesh.nver);
        cudaDeviceSynchronize();
        _LAUNCH_NVRTC(mesh.nver, 256, enforceBC_nvrtc.kernel, {&dataPtr, &mesh.nver});
        // _LAUNCH(mesh.nver,256, enforceBC) (dataPtr, mesh.nver);
        cudaDeviceSynchronize();
        if (!gP.isAnchored()) {
            _LAUNCH(mesh.nver, 256, enforce_anchored) (dataPtr, dt, mesh.nver);
            cudaDeviceSynchronize();
        }
        _LAUNCH(mesh.nver, 256, update_pos) (dataPtr, dt, mesh.nver);
        cudaDeviceSynchronize();

        cudaDeviceSynchronize();
        _LAUNCH(mesh.nver, 256, compute_kenergy) (dataPtr, mesh.nver);
        cudaDeviceSynchronize();
        globalTime += dt;
    }
}
