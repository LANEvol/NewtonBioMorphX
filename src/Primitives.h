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

#ifndef LAGRANGIANSOLID_CUDA_PRIMITIVES_H
#define LAGRANGIANSOLID_CUDA_PRIMITIVES_H
#include "Typedefs.h"

//namespace LagSol {
    template<class T, int N>
    struct PlainVector {
        T data[N];
        __host__ __device__ PlainVector() {};
        __host__ __device__ PlainVector(const T& _x) {
#pragma unroll
            for (int i = 0; i < N; i++)
                data[i] = _x;
        };
        __host__ __device__ PlainVector(const PlainVector& other) {
#pragma unroll
            for (int i = 0; i < N; i++)
                data[i] = other[i];
        };
        __host__ __device__ ~PlainVector() {};
        __host__ __device__ PlainVector operator+(const PlainVector<T,N>& other) const {
            PlainVector result;
#pragma unroll
            for (int i = 0; i < N; i++)
                result[i] = data[i] + other[i];
            return result;
        }
        __host__ __device__ PlainVector operator-(const PlainVector<T,N>& other) const {
            PlainVector result;
#pragma unroll
            for (int i = 0; i < N; i++)
                result[i] = data[i] - other[i];
            return result;
        }
        __host__ __device__ PlainVector operator*(const Float& aFloat) const {
            PlainVector result;
#pragma unroll
            for (int i = 0; i < N; i++)
                result[i] = data[i] * aFloat;
            return result;
        }
        __host__ __device__ PlainVector operator/(const Float& aFloat) const {
            PlainVector result;
#pragma unroll
            for (int i = 0; i < N; i++)
                result[i] = data[i] / aFloat;
            return result;
        }
        __host__ __device__ const T& operator[](int i) const  { return data[i]; }
        __host__ __device__ T& operator[](int i) { return data[i];}
    };

    struct Range : public PlainVector<Float,2> {
        __host__ __device__ Range() : PlainVector() {};
        __host__ __device__ Range(const Float& _x) : PlainVector(_x) {};
        __host__ __device__ Range(const Float& _x,const Float& _y) {data[0] = _x; data[1] = _y;};
        __host__ __device__ ~Range() {};
    };

    struct Vector : public PlainVector<Float,3> {
        __host__ __device__ Vector() : PlainVector() {};
        __host__ __device__ Vector(const Float& _x) : PlainVector(_x) {};
        __host__ __device__ Vector(const Float& _x,const Float& _y,const Float& _z) {data[0] = _x; data[1] = _y; data[2] = _z;};
        __host__ __device__ ~Vector() {};
        __host__ __device__ Vector operator+(const Vector& other) const { return Vector(data[0]+other[0],data[1]+other[1],data[2]+other[2]);}
        __host__ __device__ Vector operator-(const Vector& other) const  { return Vector(data[0]-other[0],data[1]-other[1],data[2]-other[2]);}
        __host__ __device__ Vector operator*(const Vector& other) const  { return Vector(data[0]*other[0],data[1]*other[1],data[2]*other[2]);}
        __host__ __device__ Vector operator*(const Float& other) const  { return Vector(data[0]*other,data[1]*other,data[2]*other);}
        __host__ __device__ Vector operator/(const Vector& other) const  { return Vector(data[0]/other[0],data[1]/other[1],data[2]/other[2]);}
        __host__ __device__ Vector operator/(const Float& other) const  { return Vector(data[0]/other,data[1]/other,data[2]/other);}
        __host__ __device__ Float dot(const Vector& other) const  { return (data[0]*other[0]+data[1]*other[1]+data[2]*other[2]);}
        __host__ __device__ Vector cross(const Vector& other) const  { return Vector(data[1]*other[2] - data[2]*other[1],data[2]*other[0]-data[0]*other[2],data[0]*other[1] - data[1]*other[0]);}
        __host__ __device__ Float norm1() const {return max(abs(data[0]),max(abs(data[1]),abs(data[2])));}
        __host__ __device__ Vector normal() const {return Vector(data[0],data[1],data[2])/(mag());}
        __host__ __device__ Vector safe_normal() const {return Vector(data[0],data[1],data[2])/(mag()+(mag2()==0)*1e-10);}
        __host__ __device__ Float mag() const {return sqrt(this->dot(*this));}
        __host__ __device__ Float mag2() const {return this->dot(*this);}
    };

    struct Tensor : public PlainVector<Float, 9> {
        __host__ __device__ Tensor() : PlainVector() {};
        __host__ __device__ Tensor(const Float& _x) : PlainVector(_x) {};
        __host__ __device__ Tensor(
            const Float& xx,const Float& xy,const Float& xz,
            const Float& yx,const Float& yy,const Float& yz,
            const Float& zx,const Float& zy,const Float& zz) {data[0] = xx; data[1] = xy; data[2] = xz; data[3] = yx; data[4] = yy; data[5] = yz; data[6] = zx; data[7] = zy; data[8] = zz;};
        __host__ __device__ Tensor(const Vector& a) : Tensor(a[0]*a[0], a[0]*a[1],a[0]*a[2],
                                                             a[1]*a[0], a[1]*a[1],a[1]*a[2],
                                                             a[2]*a[0], a[2]*a[1],a[2]*a[2]) {};
        __host__ __device__ Tensor(const Float& x,const Float& y,const Float& z) : Tensor(Vector(x,y,z)) {};

        __host__ __device__ ~Tensor() {};

        __host__ __device__ Vector col(int i) const { return Vector(data[0+i], data[3+i],data[6+i]);};
        __host__ __device__ Tensor operator+(const Tensor& other) const { return Tensor(data[0]+other[0],data[1]+other[1],data[2]+other[2],
                                                                                        data[3]+other[3],data[4]+other[4],data[5]+other[5],
                                                                                        data[6]+other[6],data[7]+other[7],data[8]+other[8]);}
        __host__ __device__ Tensor operator-(const Tensor& other) const { return Tensor(data[0]-other[0],data[1]-other[1],data[2]-other[2],
                                                                                        data[3]-other[3],data[4]-other[4],data[5]-other[5],
                                                                                        data[6]-other[6],data[7]-other[7],data[8]-other[8]);}
        __host__ __device__ Tensor operator*(const Float& other) const { return Tensor(data[0]*other,data[1]*other,data[2]*other,data[3]*other,data[4]*other,data[5]*other,data[6]*other,data[7]*other,data[8]*other);}
        __host__ __device__ Tensor operator/(const Float& other) const { return Tensor(data[0]/other,data[1]/other,data[2]/other,data[3]/other,data[4]/other,data[5]/other,data[6]/other,data[7]/other,data[8]/other);}
        __host__ __device__ Tensor dot(const Tensor& other) const {
            return Tensor(data[0]*other[0]+data[1]*other[3]+data[2]*other[6],
                          data[0]*other[1]+data[1]*other[4]+data[2]*other[7],
                          data[0]*other[2]+data[1]*other[5]+data[2]*other[8],
                          data[3]*other[0]+data[4]*other[3]+data[5]*other[6],
                          data[3]*other[1]+data[4]*other[4]+data[5]*other[7],
                          data[3]*other[2]+data[4]*other[5]+data[5]*other[8],
                          data[6]*other[0]+data[7]*other[3]+data[8]*other[6],
                          data[6]*other[1]+data[7]*other[4]+data[8]*other[7],
                          data[6]*other[2]+data[7]*other[5]+data[8]*other[8]);
        }

        __host__ __device__ static Tensor eye() {
            return Tensor(1,0,0,0,1,0,0,0,1);
        }

        __host__ __device__ static Tensor diag(const Vector& v) {
            return Tensor(v[0],0,0,0,v[1],0,0,0,v[2]);
        }

        __host__ __device__ Vector diag() const {
            return Vector(data[0],data[4],data[8]);
        }

        __host__ __device__ Vector dot(const Vector& other) const {
            return Vector(data[0]*other[0]+data[1]*other[1]+data[2]*other[2],
                          data[3]*other[0]+data[4]*other[1]+data[5]*other[2],
                          data[6]*other[0]+data[7]*other[1]+data[8]*other[2]);
        }

        __host__ __device__ Float det() const {
            return (data[0] * data[4] * data[8] - data[0] * data[5] * data[7] - data[1] * data[3] * data[8] + data[1] * data[5] * data[6] + data[2] * data[3] * data[7] - data[2] * data[4] * data[6]);
        }

        __host__ __device__ Tensor inv() const {
            return Tensor((data[4] * data[8] - data[5] * data[7]),(data[2] * data[7] - data[1] * data[8]),(data[1] * data[5] - data[2] * data[4]),
                          (data[5] * data[6] - data[3] * data[8]),(data[0] * data[8] - data[2] * data[6]),(data[2] * data[3] - data[0] * data[5]),
                          (data[3] * data[7] - data[4] * data[6]),(data[1] * data[6] - data[0] * data[7]),(data[0] * data[4] - data[1] * data[3]))/det();
        }
        __host__ __device__ Tensor inv(const Float& aDet) const {
            return Tensor((data[4] * data[8] - data[5] * data[7]),(data[2] * data[7] - data[1] * data[8]),(data[1] * data[5] - data[2] * data[4]),
                          (data[5] * data[6] - data[3] * data[8]),(data[0] * data[8] - data[2] * data[6]),(data[2] * data[3] - data[0] * data[5]),
                          (data[3] * data[7] - data[4] * data[6]),(data[1] * data[6] - data[0] * data[7]),(data[0] * data[4] - data[1] * data[3]))/aDet;
        }
        __host__ __device__ Tensor trans() const {
            return Tensor(data[0],data[3],data[6],data[1],data[4],data[7],data[2],data[5],data[8]);
        }
        __host__ __device__ Float trace() const {
            return (data[0]+data[4]+data[8]);
        }

        __host__ __device__ Tensor dev() const {
            Float tmp = (data[0]+data[4]+data[8])/Float(3.0);
            return Tensor(data[0]-tmp,data[3],data[6],data[1],data[4]-tmp,data[7],data[2],data[5],data[8]-tmp);
        }
        __host__ __device__ Float norm2() const {
            return sqrt(data[0]*data[0] + data[1]*data[1] + data[2]*data[2] + data[3]*data[3] + data[4]*data[4] + data[5]*data[5] + data[6]*data[6] + data[7]*data[7] + data[8]*data[8]);
        }
    };

    inline __host__ __device__ Tensor operator*(const Float& aFloat, const Tensor& aTensor) {
        return aTensor * aFloat;
    }

    inline __host__ __device__ Vector dot(const Vector& v, const Tensor& t) {
        return Vector(t[0] * v[0] + t[3] * v[1] + t[6] * v[2],
                      t[1] * v[0] + t[4] * v[1] + t[7] * v[2],
                      t[2] * v[0] + t[5] * v[1] + t[8] * v[2]);
    }

    inline __host__ __device__ Tensor tprod(const Vector& a, const Vector& b)  { return Tensor(a[0]*b[0], a[0]*b[1],a[0]*b[2],
                                                                                               a[1]*b[0], a[1]*b[1],a[1]*b[2],
                                                                                               a[2]*b[0], a[2]*b[1],a[2]*b[2]);
    }

    struct IntVector {
        int x,y,z;
        __host__ __device__ IntVector() : x(), y(), z() {};
        __host__ __device__ IntVector(int _x, int _y) : x(_x), y(_y), z(0) { };
        __host__ __device__ IntVector(int _x, int _y, int _z) : x(_x), y(_y), z(_z) { };
    };

//}

#endif //LAGRANGIANSOLID_CUDA_PRIMITIVES_H