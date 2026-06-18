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

#include "Typedefs.h"
#include "TextIO.h"
#include "MyGUI.h"
#include "Kernels.h"
#include <iosfwd>
#include <fstream>

std::string getFile(const std::string& filename) {
    std::ifstream filein(filename);
    std::string contents;
    for (char ch; filein.get(ch); contents.push_back(ch)) {}
    return contents;
}

std::string stringReplace(const std::string& input, const std::string& old_str, const std::string& new_str) {
    // This searches the file for the first occurence of the old_str string.
    std::string output = input;
    auto pos = input.find(old_str);
    while (pos != std::string::npos) {
        output.replace(pos, old_str.length(), new_str);
        // Continue searching from here.
        pos = output.find(old_str, pos);
    }
    return output;
}

void findAndReplace(std::string& file_contents, const std::string& old_str, const std::string& new_str) {
    // This searches the file for the first occurence of the old_str string.
    auto pos = file_contents.find(old_str);
    while (pos != std::string::npos) {
        file_contents.replace(pos, old_str.length(), new_str);
        // Continue searching from here.
        pos = file_contents.find(old_str, pos);
    }
}

void findAndReplaceToEndOfLine(std::string& file_contents, const std::string& old_str, const std::string& new_str) {
    size_t pos = file_contents.find(old_str);
    while (pos != std::string::npos) {
        // Find end of line
        size_t end = file_contents.find('\n', pos);
        if (end == std::string::npos) {
            // No newline found → replace until end of file
            end = file_contents.size();
        }
        // Replace from the match to end of line
        file_contents.replace(pos, end - pos, new_str);
        // Continue searching after the replacement
        pos = file_contents.find(old_str, pos + new_str.length());
    }
}

std::string trim(const std::string& str,
                 const std::string& whitespace = " \t")
{
    const auto strBegin = str.find_first_not_of(whitespace);
    if (strBegin == std::string::npos)
        return ""; // no content

    const auto strEnd = str.find_last_not_of(whitespace);
    const auto strRange = strEnd - strBegin + 1;

    return str.substr(strBegin, strRange);
}


void saveFile(const std::string& filename, const std::string& contents) {
    std::ofstream fileout(filename);
    fileout << contents;
}

std::string getLayerExpressions(const std::string expressions[], int nLayers) {
    std::string tempStr="";
    for (int l=0;l<nLayers;l++) {
        tempStr.append(std::string("(") + std::string(expressions[l]) + std::string(") * Float(layer==") + std::to_string(l) + std::string(")"));
        if (l<nLayers-1)
            tempStr.append(" + ");
    }
    return tempStr;
}

std::string getLayerExpressionsBoolean(const bool values[], int nLayers) {
    std::string tempStr="";
    for (int l=0;l<nLayers;l++) {
        tempStr.append(std::string("(") + std::to_string(values[l]) + std::string(") && (layer==") + std::to_string(l) + std::string(")"));
        if (l<nLayers-1)
            tempStr.append(" || ");
    }
    return tempStr;
}

using namespace LagSol;

std::string getNameOfGeom(const BasicGeometry& geom) {
    std::string result;
    if (geom == Box)
        result = "Box";
    else if (geom == Sphere)
        result = "Sphere";
    else if (geom == Tube)
        result = "Tube";
    else if (geom == Cone)
        result = "Cone";
    else if (geom == Disk)
        result = "Disk";
    else if (geom == Torus)
        result = "Torus";
    else if (geom == Arbitrary)
        result = "Arbitrary";
    else
        result = "Unknown";
    return result;
}

LagSol::CoordinateSystem grCoordType;

std::string getNameOfCoordSystem(const CoordinateSystem& coord) {
    std::string result;
    if (coord == Cartesian)
        result = "Cartesian";
    else if (coord == Spherical)
        result = "Spherical";
    else if (coord == CylindricalZ)
        result = "CylindricalZ";
    else if (coord == CylindricalY)
        result = "CylindricalY";
    else if (coord == ConeAdapted)
        result = "ConeAdapted";
    else if (coord == CylindricalZ)
        result = "CylindricalZ";
    else if (coord == Toroidal)
        result = "Toroidal";
    else if (coord == NormalTangent)
        result = "NormalTangent";
    else
        result = "Unknown";
    return result;
}

CoordinateSystem getCoordSystemFromName(const std::string& coord) {
    CoordinateSystem result;
    if (coord == "Cartesian")
        result = Cartesian;
    else if (coord == "Spherical")
        result = Spherical;
    else if (coord == "CylindricalZ")
        result = CylindricalZ;
    else if (coord == "CylindricalY")
        result = CylindricalY;
    else if (coord == "ConeAdapted")
        result = ConeAdapted;
    else if (coord == "CylindricalZ")
        result = CylindricalZ;
    else if (coord == "Toroidal")
        result = Toroidal;
    else if (coord == "NormalTangent")
        result = NormalTangent;
    else {
        _ERROR_MESSAGE(coord +" is not a valid geometry!");
        exit(-1);
    }
    return result;
}

BasicGeometry getGeomFromName(const std::string& geom) {
    BasicGeometry result;
    if (geom == "Box")
        result = Box;
    else if (geom == "Sphere")
        result = Sphere;
    else if (geom == "Tube")
        result = Tube;
    else if (geom == "Cone")
        result = Cone;
    else if (geom == "Disk")
        result = Disk;
    else if (geom == "Torus")
        result = Torus;
    else if (geom == "Arbitrary")
        result = Arbitrary;
    else {
        _ERROR_MESSAGE(geom +" is not a valid geometry!");
        exit(-1);
    }
    return result;
}

void readSetting(std::string fileName){
    for (int i=0;i<8;i++)
        showRank1Tensors[i]=false;
    gP.init();
    std::string delimiter(" ");
    std::string lineStr;
    std::ifstream infile;
    infile.open(fileName);
    std::getline(infile,lineStr);
    int i=0;
    while(!infile.eof()) {
        std::stringstream ss(lineStr);
        std::string token, tempStr;
        int tempInt;
        ss >> token;

        if (token == std::string("inputGeom"))   {ss >> tempStr; inputGeom = getGeomFromName(tempStr);}
        if (token == std::string("coordSystem")) {ss >> tempStr; gP.grCoordType = getCoordSystemFromName(tempStr);}

        if (inputGeom == Arbitrary) {
            if (token == std::string("inputMeshFileName")) ss >> inputMeshFileName;
            if (token == std::string("nLayers")) ss >> nLayers;
        } else if (inputGeom == Sphere) {
            if (token == std::string("nLayers")) ss >> sphereDims.nLayers;
            if (token == std::string("Ri")) ss >> sphereDims.Ri;
            if (token == std::string("H")) for (int j=0; j<sphereDims.nLayers; j++) ss >> sphereDims.H[j];
            if (token == std::string("spacing")) for (int j=0; j<sphereDims.nLayers; j++) ss >> sphereDims.spacing[j];
        } else if (inputGeom == Tube) {
            if (token == std::string("nLayers")) ss >> tubeDims.nLayers;
            if (token == std::string("L")) ss >> tubeDims.L;
            if (token == std::string("Ri")) ss >> tubeDims.Ri;
            if (token == std::string("H")) for (int j=0; j<tubeDims.nLayers; j++) ss >> tubeDims.H[j];
            if (token == std::string("spacing")) for (int j=0; j<tubeDims.nLayers; j++) ss >> tubeDims.spacing[j];
        } else if (inputGeom == Cone) {
            if (token == std::string("nLayers")) ss >> coneDims.nLayers;
            if (token == std::string("L")) ss >> coneDims.L;
            if (token == std::string("Ri")) ss >> coneDims.Ri;
            if (token == std::string("apexAng")) ss >> coneDims.apexAng;
            if (token == std::string("H")) for (int j=0; j<coneDims.nLayers; j++) ss >> coneDims.H[j];
            if (token == std::string("spacing")) for (int j=0; j<coneDims.nLayers; j++) ss >> coneDims.spacing[j];
        } else if (inputGeom == Disk) {
            if (token == std::string("nLayers")) ss >> diskDims.nLayers;
            if (token == std::string("R")) ss >> diskDims.R;
            if (token == std::string("H")) for (int j=0; j<diskDims.nLayers; j++) ss >> diskDims.H[j];
            if (token == std::string("spacing")) for (int j=0; j<diskDims.nLayers; j++) ss >> diskDims.spacing[j];
        } else if (inputGeom == Torus) {
            if (token == std::string("nLayers")) ss >> torusDims.nLayers;
            if (token == std::string("R")) ss >> torusDims.R;
            if (token == std::string("Ri")) ss >> torusDims.Ri;
            if (token == std::string("H")) for (int j=0; j<torusDims.nLayers; j++) ss >> torusDims.H[j];
            if (token == std::string("spacing")) for (int j=0; j<torusDims.nLayers; j++) ss >> torusDims.spacing[j];
        } else if (inputGeom == Box) {
            if (token == std::string("nLayers")) ss >> boxDims.nLayers;
            if (token == std::string("L")) ss >> boxDims.L;
            if (token == std::string("W")) ss >> boxDims.W;
            if (token == std::string("H")) for (int j=0; j<boxDims.nLayers; j++) ss >> boxDims.H[j];
            if (token == std::string("spacing")) for (int j=0; j<boxDims.nLayers; j++) ss >> boxDims.spacing[j];
        }

        if (inputGeom == Arbitrary) {
            if (token == std::string("useMeshDef_E")) ss >> gP.useMeshDef_E;
            if (token == std::string("useMeshDef_nu")) ss >> gP.useMeshDef_nu;
            if (token == std::string("useMeshDef_viscosity")) ss >> gP.useMeshDef_viscosity;
            if (token == std::string("useMeshDef_plasticity")) ss >> gP.useMeshDef_plasticity;
            if (token == std::string("useMeshDef_grRate1")) ss >> gP.useMeshDef_grRate1_Ref;
            if (token == std::string("useMeshDef_grRate2")) ss >> gP.useMeshDef_grRate2_Ref;
            if (token == std::string("useMeshDef_grRate3")) ss >> gP.useMeshDef_grRate3_Ref;
            if (token == std::string("useMeshDef_k1")) ss >> gP.useMeshDef_k1;
            if (token == std::string("useMeshDef_k2")) ss >> gP.useMeshDef_k2;
            if (token == std::string("useMeshDef_fiber1_Ref")) ss >> gP.useMeshDef_fiber1_Ref;
            if (token == std::string("useMeshDef_fiber2_Ref")) ss >> gP.useMeshDef_fiber2_Ref;
            if (token == std::string("useMeshDef_fiber3_Ref")) ss >> gP.useMeshDef_fiber3_Ref;
            if (token == std::string("useMeshDef_fiber4_Ref")) ss >> gP.useMeshDef_fiber4_Ref;
            if (token == std::string("useMeshDef_actin_Ref")) ss >> gP.useMeshDef_actin_Ref;
        }

        if (token == std::string("nFibers")) ss >> nFibers;
        if (token == std::string("globalTime")) ss >> globalTime;
        if (token == std::string("outputFileName")) ss >> outputFileName;
        if (token == std::string("maxIter"))  ss >> maxIter;
        if (token == std::string("timeThreshold"))  ss >> timeThreshold;
        if (token == std::string("saveTime"))  ss >> saveTime;
        if (token == std::string("frames"))  ss >> frames;
        if (token == std::string("timeFactor"))  ss >> timeFactor;
        if (token == std::string("minTimeFactor"))  ss >> minTimeFactor;
        if (token == std::string("saveVTK")) ss >> saveVTK;
        if (token == std::string("savePLY")) ss >> savePLY;
        if (token == std::string("savePNG")) ss >> saveMP4;
        if (token == std::string("showBasis")) ss >> showBasis;
        if (token == std::string("showEdges")) ss >> showEdges;
        if (token == std::string("showOrtho")) ss >> showOrtho;
        if (token == std::string("showFilamentFibers")) for (int j=0; j<boxDims.nLayers; j++) ss >> showRank1Tensors[j];
        if (token == std::string("scaleFilamentFibers")) for (int j=0; j<boxDims.nLayers; j++) ss >> scaleRank1Tensors[j];
        if (token == std::string("searchIter")) ss >> searchIter;
        if (token == std::string("kinEnergyTol")) ss >> kinEnergyTol;
        if (token == std::string("simIter")) ss >> simIter;
        if (token == std::string("meshVisState")) ss >> meshVisState;
        if (token == std::string("selectedLayer")) ss >> selectedLayer;
        if (token == std::string("selectedLayer")) ss >> selectedLayer;
        if (token == std::string("sliceDir")) ss >> sliceDir;
        if (token == std::string("sliceMag")) ss >> sliceMag[0] >> sliceMag[1] >> sliceMag[2];
        if (token == std::string("visibleData")) ss >> visibleData;
        if (token == std::string("sUpdate")) ss >> sUpdate;
        if (token == std::string("pert")) ss >> pert;
        if (token == std::string("bkgColor")) ss >> bkgColor[0] >> bkgColor[1] >> bkgColor[2] >> bkgColor[3];
        // if (token == std::string("layerColors"))    ss >> layersColorEigen(0,i) >> layersColorEigen(1,i) >> layersColorEigen(2,i) >> layersColorEigen(3,i);
        // Eigen::Array4Xf layersColorEigen(4, MAX_NLAYERS);
        if (token == std::string("trackball_angle")) ss >> trackball_angle.x() >> trackball_angle.y()
                                                          >> trackball_angle.z() >> trackball_angle.w();
        if (token == std::string("contactFact")) ss >> contactFact;
        if (token == std::string("contactFactVel")) ss >> contactFactVel;
        if (token == std::string("repThickness")) ss >> repThickness;
        if (token == std::string("grRateGlobal"))     ss >> gP.grRateGlobal[0] >> gP.grRateGlobal[1] >> gP.grRateGlobal[2] >>
                                                              gP.grRateGlobal[3] >> gP.grRateGlobal[4] >> gP.grRateGlobal[5] >>
                                                              gP.grRateGlobal[6] >> gP.grRateGlobal[7] >> gP.grRateGlobal[8];
        if (token == std::string("damping")) ss >> gP.damping;
        if (token == std::string("bcTypes")) ss >> gP.bcTypeMinAxis0 >> gP.bcTypeMaxAxis0 >>
                          gP.bcTypeMinAxis1 >> gP.bcTypeMaxAxis1 >>
                          gP.bcTypeMinAxis2 >> gP.bcTypeMaxAxis2;

        if (token == std::string("Layer"))      ss >> i;

        if (token == std::string("isRigid"))    ss >> gP.isRigidLayer[i];
        if (token == std::string("nu"))        {std::getline(ss, gP.nu[i]); gP.nu[i] = trim(gP.nu[i], "= \t");};
        if (token == std::string("E"))         {std::getline(ss, gP.E[i]); gP.E[i] = trim(gP.E[i], "= \t");};
        if (token == std::string("visc"))      {std::getline(ss, gP.visc[i]); gP.visc[i] = trim(gP.visc[i], "= \t");};
        if (token == std::string("plasticity"))    {std::getline(ss, gP.plasticity[i]); gP.plasticity[i] = trim(gP.plasticity[i], "= \t");};
        if (token == std::string("grRate1_Ref"))   {std::getline(ss, gP.grRate1_Ref[i]); gP.grRate1_Ref[i] = trim(gP.grRate1_Ref[i], "= \t");};
        if (token == std::string("grRate2_Ref"))   {std::getline(ss, gP.grRate2_Ref[i]); gP.grRate2_Ref[i] = trim(gP.grRate2_Ref[i], "= \t");};
        if (token == std::string("grRate3_Ref"))   {std::getline(ss, gP.grRate3_Ref[i]); gP.grRate3_Ref[i] = trim(gP.grRate3_Ref[i], "= \t");};

        if (token == std::string("k1"))          {std::getline(ss, gP.k1[i]); gP.k1[i] = trim(gP.k1[i], "= \t");};
        if (token == std::string("k2"))          {std::getline(ss, gP.k2[i]); gP.k2[i] = trim(gP.k2[i], "= \t");};
        if (token == std::string("fiber1_Ref"))  {std::getline(ss, gP.fiber1_Ref[i]); gP.fiber1_Ref[i] = trim(gP.fiber1_Ref[i], "= \t");};
        if (token == std::string("fiber2_Ref"))  {std::getline(ss, gP.fiber2_Ref[i]); gP.fiber2_Ref[i] = trim(gP.fiber2_Ref[i], "= \t");};
        if (token == std::string("fiber3_Ref"))  {std::getline(ss, gP.fiber3_Ref[i]); gP.fiber3_Ref[i] = trim(gP.fiber3_Ref[i], "= \t");};
        if (token == std::string("fiber4_Ref"))  {std::getline(ss, gP.fiber4_Ref[i]); gP.fiber4_Ref[i] = trim(gP.fiber4_Ref[i], "= \t");};
        if (token == std::string("actin_Ref"))   {std::getline(ss, gP.actin_Ref[i]); gP.actin_Ref[i] = trim(gP.actin_Ref[i], "= \t");};

        std::getline(infile,lineStr);
    }
    infile.close();
}

void writeSetting(std::string fileName) {
    std::ofstream ofile;
    ofile.open(fileName);

    ofile << "inputGeom " << getNameOfGeom(inputGeom) << std::endl;
    ofile << "coordSystem " << getNameOfCoordSystem(gP.grCoordType) << std::endl;
    if (inputGeom == Arbitrary) {
        ofile << "inputMeshFileName "<< inputMeshFileName << std::endl;
        ofile << "nLayers "   << nLayers << std::endl;
    } else if (inputGeom == Sphere) {
        ofile << "nLayers "   << sphereDims.nLayers << std::endl;
        ofile << "Ri "<< sphereDims.Ri << std::endl;
        ofile << "H "; for (int i=0; i<nLayers; i++) ofile << sphereDims.H[i] << " "; ofile << std::endl;
        ofile << "spacing "; for (int i=0; i<nLayers; i++) ofile << sphereDims.spacing[i] << " "; ofile << std::endl;
    } else if (inputGeom == Tube) {
        ofile << "nLayers "   << tubeDims.nLayers << std::endl;
        ofile << "L "<< tubeDims.L << std::endl;
        ofile << "Ri "<< tubeDims.Ri << std::endl;
        ofile << "H "; for (int i=0; i<nLayers; i++) ofile << tubeDims.H[i] << " "; ofile << std::endl;
        ofile << "spacing "; for (int i=0; i<nLayers; i++) ofile << tubeDims.spacing[i] << " "; ofile << std::endl;
    } else if (inputGeom == Cone) {
        ofile << "nLayers "   << coneDims.nLayers << std::endl;
        ofile << "L "<< coneDims.L << std::endl;
        ofile << "Ri " << coneDims.Ri << std::endl;
        ofile << "apexAng " << coneDims.apexAng << std::endl;
        ofile << "H "; for (int i=0; i<nLayers; i++) ofile << coneDims.H[i] << " "; ofile << std::endl;
        ofile << "spacing "; for (int i=0; i<nLayers; i++) ofile << coneDims.spacing[i] << " ";ofile << std::endl;
    } else if (inputGeom == Disk) {
        ofile << "nLayers "   << diskDims.nLayers << std::endl;
        ofile << "R" << diskDims.R << std::endl;
        ofile << "H "; for (int i=0; i<nLayers; i++) ofile << diskDims.H[i] << " "; ofile << std::endl;
        ofile << "spacing "; for (int i=0; i<nLayers; i++) ofile << diskDims.spacing[i] << " ";ofile << std::endl;
    } else if (inputGeom == Torus) {
        ofile << "nLayers "   << torusDims.nLayers << std::endl;
        ofile << "R" << torusDims.R << std::endl;
        ofile << "Ri" << torusDims.Ri << std::endl;
        ofile << "H "; for (int i=0; i<nLayers; i++) ofile << torusDims.H[i] << " "; ofile << std::endl;
        ofile << "spacing "; for (int i=0; i<nLayers; i++) ofile << torusDims.spacing[i] << " "; ofile << std::endl;
    } else if (inputGeom == Box) {
        ofile << "nLayers "   << boxDims.nLayers << std::endl;
        ofile << "L "<< boxDims.L << std::endl;
        ofile << "W "<< boxDims.W << std::endl;
        ofile << "H "; for (int i=0; i<nLayers; i++) ofile << boxDims.H[i] << " "; ofile << std::endl;
        ofile << "spacing "; for (int i=0; i<nLayers; i++) ofile << boxDims.spacing[i] << " "; ofile << std::endl;
    }

    if (inputGeom == Arbitrary) {
        ofile << "useMeshDef_E "   << gP.useMeshDef_E << std::endl;
        ofile << "useMeshDef_nu "   << gP.useMeshDef_nu << std::endl;
        ofile << "useMeshDef_viscosity "   << gP.useMeshDef_viscosity << std::endl;
        ofile << "useMeshDef_plasticity "   << gP.useMeshDef_plasticity << std::endl;
        ofile << "useMeshDef_grRate1_Ref "   << gP.useMeshDef_grRate1_Ref << std::endl;
        ofile << "useMeshDef_grRate2_Ref "   << gP.useMeshDef_grRate2_Ref << std::endl;
        ofile << "useMeshDef_grRate3_Ref "   << gP.useMeshDef_grRate3_Ref << std::endl;
        ofile << "useMeshDef_k1 "   << gP.useMeshDef_k1 << std::endl;
        ofile << "useMeshDef_k2 "   << gP.useMeshDef_k2 << std::endl;
        ofile << "useMeshDef_fiber1_Ref "   << gP.useMeshDef_fiber1_Ref << std::endl;
        ofile << "useMeshDef_fiber2_Ref "   << gP.useMeshDef_fiber2_Ref << std::endl;
        ofile << "useMeshDef_fiber3_Ref "   << gP.useMeshDef_fiber3_Ref << std::endl;
        ofile << "useMeshDef_fiber4_Ref "   << gP.useMeshDef_fiber4_Ref << std::endl;
        ofile << "useMeshDef_actin_Ref "   << gP.useMeshDef_actin_Ref << std::endl;
    }

    ofile << "nFibers "   << nFibers << std::endl;
    ofile << "globalTime "<< globalTime << std::endl;
    ofile << "outputFileName "<< outputFileName << std::endl;
    ofile << "maxIter "<< maxIter << std::endl;
    ofile << "timeThreshold "<< timeThreshold << std::endl;
    ofile << "saveTime "<< saveTime << std::endl;
    ofile << "frames "<< frames << std::endl;
    ofile << "timeFactor "<< timeFactor << std::endl;
    ofile << "minTimeFactor "<< minTimeFactor << std::endl;
    ofile << "saveVTK " <<saveVTK << std::endl;
    ofile << "savePLY " <<savePLY << std::endl;
    ofile << "savePNG " <<saveMP4 << std::endl;
    ofile << "showBasis "<<showBasis << std::endl;
    ofile << "showEdges "<< showEdges << std::endl;
    ofile << "showOrtho "<< showOrtho << std::endl;
    ofile << "showFilamentFibers "; for (int i=0; i<8; i++) ofile << showRank1Tensors[i] << " "; ofile << std::endl;
    ofile << "scaleFilamentFibers "; for (int i=0; i<8; i++) ofile << scaleRank1Tensors[i] << " "; ofile << std::endl;
    ofile << "searchIter "<< searchIter << std::endl;
    ofile << "kinEnergyTol "<< kinEnergyTol << std::endl;
    ofile << "simIter "<< simIter << std::endl;
    ofile << "meshVisState "<< meshVisState << std::endl;
    ofile << "selectedLayer "<< selectedLayer << std::endl;
    ofile << "sliceDir "<< sliceDir << std::endl;
    ofile << "sliceMag "<< sliceMag[0] <<" "<<sliceMag[1]<<" "<<sliceMag[2]<< std::endl;
    ofile << "visibleData "<< visibleData << std::endl;
    ofile << "sUpdate "<< sUpdate << std::endl;
    ofile << "pert " << pert << std::endl;
    ofile << "bkgColor "<< bkgColor(0)<<" "<<bkgColor(1)<<" "<<bkgColor(2)<<" "<<bkgColor(3)<< std::endl;
    // Eigen::Array4Xf layersColorEigen(4, MAX_NLAYERS);
    ofile << "trackball_angle "<< current_trackball_angle.x()<<" "<<current_trackball_angle.y()<<" "<<current_trackball_angle.z()<<" "<<current_trackball_angle.w()<< std::endl;
    ofile << "contactFact "<< contactFact << std::endl;
    ofile << "contactFactVel "<< contactFactVel << std::endl;
    ofile << "repThickness "<< repThickness << std::endl;

    // ofile << "grRateGlobal "<< gP.grRateGlobal[0] << " " << gP.grRateGlobal[1]<< " " << gP.grRateGlobal[2] << " "
    //                             << gP.grRateGlobal[3] << " " << gP.grRateGlobal[4]<< " " << gP.grRateGlobal[5] << " "
    //                             << gP.grRateGlobal[6] << " " << gP.grRateGlobal[7]<< " " << gP.grRateGlobal[8] << std::endl;
    // ofile << "damping "<< gP.damping << std::endl;

    ofile << "bcTypes "<< gP.bcTypeMinAxis0 << " "<< gP.bcTypeMaxAxis0 << " " <<
                          gP.bcTypeMinAxis1 << " "<< gP.bcTypeMaxAxis1 << " " <<
                          gP.bcTypeMinAxis2 << " "<< gP.bcTypeMaxAxis2 << std::endl;

    for (int i=0; i<nLayers; i++) {
        ofile << "" << "Layer " << i << std::endl;

        ofile << "\t" << "isRigid "<< gP.isRigidLayer[i]<< std::endl;
        ofile << "\t" << "nu "<< gP.nu[i]<< std::endl;
        ofile << "\t" << "E "<< gP.E[i]<< std::endl;
        ofile << "\t" << "visc "<< gP.visc[i]<< std::endl;
        ofile << "\t" << "grRate1_Ref "<< gP.grRate1_Ref[i]<< std::endl;
        ofile << "\t" << "grRate2_Ref "<< gP.grRate2_Ref[i]<< std::endl;
        ofile << "\t" << "grRate3_Ref "<< gP.grRate3_Ref[i]<< std::endl;
        ofile << "\t" << "plasticity "<< gP.plasticity[i]<< std::endl;
        if (nFibers>0) {
            ofile << "\t" << "k1 "<< gP.k1[i]<< std::endl;
            ofile << "\t" << "k2 "<< gP.k2[i]<< std::endl;
            ofile << "\t" << "fiber1_Ref "<< gP.fiber1_Ref[i]<< std::endl;
            if (nFibers>1) ofile << "\t" << "fiber2_Ref "<< gP.fiber2_Ref[i]<< std::endl;
            if (nFibers>2) ofile << "\t" << "fiber3_Ref "<< gP.fiber3_Ref[i]<< std::endl;
            if (nFibers>3) ofile << "\t" << "fiber4_Ref "<< gP.fiber4_Ref[i]<< std::endl;
        }
        ofile << "\t" << "actin_Ref "<< gP.actin_Ref[i]<< std::endl;
    }
    ofile.close();
}
