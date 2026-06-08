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

#ifndef LAGRANGIANSOLID_CUDA_TEXTIO_H
#define LAGRANGIANSOLID_CUDA_TEXTIO_H
#include <string>

std::string getFile(const std::string& filename);
std::string stringReplace(const std::string& input, const std::string& old_str, const std::string& new_str);
void findAndReplace(std::string& file_contents, const std::string& old_str, const std::string& new_str);
void findAndReplaceToEndOfLine(std::string& file_contents, const std::string& old_str, const std::string& new_str);
std::string trim(const std::string& str, const std::string& whitespace);
void saveFile(const std::string& filename, const std::string& contents);
std::string getLayerExpressions(const std::string expressions[], int nLayers);
std::string getLayerExpressionsBoolean(const bool values[], int nLayers);

void readSetting(std::string fileName);
void writeSetting(std::string fileName);

#endif //LAGRANGIANSOLID_CUDA_TEXTIO_H