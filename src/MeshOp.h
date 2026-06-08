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

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <vector>

// ----------------- Data for interpolation -----------------

struct NewVertex
{
    Eigen::Vector3d pos;          // position of new vertex
    std::vector<int> idx;         // original vertex indices
    std::vector<double> w;        // barycentric weights (sum to 1)
};

// ----------------- Segment-plane intersection -----------------

inline bool intersect_segment_plane(
    const Eigen::Vector3d& p0,
    const Eigen::Vector3d& p1,
    const Eigen::Vector3d& P,   // point on plane
    const Eigen::Vector3d& N,   // plane normal (assumed normalized)
    Eigen::Vector3d& out,
    double& t_out)
{
    double d0 = (p0 - P).dot(N);
    double d1 = (p1 - P).dot(N);
    double denom = d0 - d1;

    if (std::abs(denom) < 1e-12) {
        return false; // parallel or lies in plane
    }

    double t = d0 / (d0 - d1);  // param in [0,1] if crossing
    if (t < 0.0 || t > 1.0) {
        return false;
    }

    out = p0 + t * (p1 - p0);
    t_out = t;
    return true;
}

// ----------------- Helpers to create new vertices -----------------

inline int add_original_vertex(
    int vid,
    const Eigen::Vector3d& pos,
    std::vector<NewVertex>& new_vertices)
{
    NewVertex nv;
    nv.pos = pos;
    nv.idx = { vid };
    nv.w   = { 1.0 };
    new_vertices.push_back(nv);
    return (int)new_vertices.size() - 1;
}

inline int add_edge_vertex(
    int v0, int v1,
    double t,                      // p = (1-t)*V[v0] + t*V[v1]
    const Eigen::Vector3d& pos,
    std::vector<NewVertex>& new_vertices)
{
    NewVertex nv;
    nv.pos = pos;
    nv.idx = { v0, v1 };
    nv.w   = { 1.0 - t, t };
    new_vertices.push_back(nv);
    return (int)new_vertices.size() - 1;
}

// ----------------- Orientation enforcement -----------------

inline void enforce_orientation(
    int& i0, int& i1, int& i2,
    const std::vector<NewVertex>& new_vertices,
    const Eigen::Vector3d& n_orig)
{
    const Eigen::Vector3d& v0 = new_vertices[i0].pos;
    const Eigen::Vector3d& v1 = new_vertices[i1].pos;
    const Eigen::Vector3d& v2 = new_vertices[i2].pos;

    Eigen::Vector3d n_new = (v1 - v0).cross(v2 - v0);
    if (n_new.dot(n_orig) < 0.0) {
        std::swap(i1, i2);
    }
}

// ----------------- Clip a single triangle -----------------

inline void clip_triangle_against_plane(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    int f,                         // face index
    const Eigen::Vector3d& P,
    const Eigen::Vector3d& N,
    std::vector<NewVertex>& new_vertices,
    std::vector<Eigen::Vector3i>& new_faces,
    std::vector<int>& face_map)
{
    int ia = F(f,0);
    int ib = F(f,1);
    int ic = F(f,2);

    Eigen::Vector3d a = V.row(ia);
    Eigen::Vector3d b = V.row(ib);
    Eigen::Vector3d c = V.row(ic);

    Eigen::Vector3d v[3] = { a, b, c };
    int idx[3] = { ia, ib, ic };

    double s[3];
    for (int i = 0; i < 3; ++i)
        s[i] = (v[i] - P).dot(N);

    bool inside[3];
    int inside_count = 0;
    for (int i = 0; i < 3; ++i) {
        inside[i] = (s[i] >= 0.0);
        if (inside[i]) ++inside_count;
    }

    if (inside_count == 0) {
        return; // fully outside
    }

    // original (unnormalized) normal
    Eigen::Vector3d n_orig = (b - a).cross(c - a);

    if (inside_count == 3) {
        int i0 = add_original_vertex(ia, a, new_vertices);
        int i1 = add_original_vertex(ib, b, new_vertices);
        int i2 = add_original_vertex(ic, c, new_vertices);

        enforce_orientation(i0, i1, i2, new_vertices, n_orig);
        new_faces.emplace_back(i0, i1, i2);
        face_map.push_back(f);
        return;
    }

    std::vector<int> in_idx, out_idx;
    for (int i = 0; i < 3; ++i) {
        if (inside[i]) in_idx.push_back(i);
        else           out_idx.push_back(i);
    }

    if (inside_count == 1) {
        int i0 = in_idx[0];
        int i1 = out_idx[0];
        int i2 = out_idx[1];

        Eigen::Vector3d p0 = v[i0];
        Eigen::Vector3d p1 = v[i1];
        Eigen::Vector3d p2 = v[i2];

        Eigen::Vector3d i01, i02;
        double t01, t02;
        intersect_segment_plane(p0, p1, P, N, i01, t01);
        intersect_segment_plane(p0, p2, P, N, i02, t02);

        int id0 = add_original_vertex(idx[i0], p0, new_vertices);
        int id1 = add_edge_vertex(idx[i0], idx[i1], t01, i01, new_vertices);
        int id2 = add_edge_vertex(idx[i0], idx[i2], t02, i02, new_vertices);

        enforce_orientation(id0, id1, id2, new_vertices, n_orig);
        new_faces.emplace_back(id0, id1, id2);
        face_map.push_back(f);
    }
    else if (inside_count == 2) {
        int i0 = in_idx[0];
        int i1 = in_idx[1];
        int i2 = out_idx[0];

        Eigen::Vector3d p0 = v[i0];
        Eigen::Vector3d p1 = v[i1];
        Eigen::Vector3d p2 = v[i2];

        Eigen::Vector3d i20, i21;
        double t20, t21;
        intersect_segment_plane(p2, p0, P, N, i20, t20);
        intersect_segment_plane(p2, p1, P, N, i21, t21);

        int id0 = add_original_vertex(idx[i0], p0, new_vertices);
        int id1 = add_original_vertex(idx[i1], p1, new_vertices);
        int id21 = add_edge_vertex(idx[i2], idx[i1], t21, i21, new_vertices);
        int id20 = add_edge_vertex(idx[i2], idx[i0], t20, i20, new_vertices);

        // quad (p0, p1, i21, i20) -> two triangles

        // Triangle 1: (p0, p1, i21)
        {
            int a0 = id0, a1 = id1, a2 = id21;
            enforce_orientation(a0, a1, a2, new_vertices, n_orig);
            new_faces.emplace_back(a0, a1, a2);
            face_map.push_back(f);
        }

        // Triangle 2: (p0, i21, i20)
        {
            int b0 = id0, b1 = id21, b2 = id20;
            enforce_orientation(b0, b1, b2, new_vertices, n_orig);
            new_faces.emplace_back(b0, b1, b2);
            face_map.push_back(f);
        }
    }
}

// ----------------- Clip full mesh and build BC -----------------

inline void clip_mesh_against_plane_with_BC(
    const Eigen::MatrixXd& V,
    const Eigen::MatrixXi& F,
    const Eigen::Vector3d& P,
    const Eigen::Vector3d& N,
    Eigen::MatrixXd& Vout,
    Eigen::MatrixXi& Fout,
    Eigen::VectorXi& face_map,
    Eigen::SparseMatrix<double>& BC)   // #Vout x #V
{
    std::vector<NewVertex> new_vertices;
    std::vector<Eigen::Vector3i> new_faces;
    std::vector<int> face_map_tmp;

    for (int f = 0; f < F.rows(); ++f) {
        clip_triangle_against_plane(
            V, F, f, P, N,
            new_vertices, new_faces, face_map_tmp);
    }

    // Build Vout
    Vout.resize(new_vertices.size(), 3);
    for (int i = 0; i < (int)new_vertices.size(); ++i)
        Vout.row(i) = new_vertices[i].pos;

    // Build Fout
    Fout.resize(new_faces.size(), 3);
    for (int i = 0; i < (int)new_faces.size(); ++i)
        Fout.row(i) = new_faces[i];

    // Build face_map
    face_map.resize(face_map_tmp.size());
    for (int i = 0; i < (int)face_map_tmp.size(); ++i)
        face_map(i) = face_map_tmp[i];

    // Build interpolation matrix BC
    std::vector<Eigen::Triplet<double>> trips;
    trips.reserve(new_vertices.size() * 2); // most vertices are on edges

    for (int i = 0; i < (int)new_vertices.size(); ++i) {
        const auto& nv = new_vertices[i];
        for (int k = 0; k < (int)nv.idx.size(); ++k) {
            trips.emplace_back(i, nv.idx[k], nv.w[k]);
        }
    }

    BC.resize(new_vertices.size(), V.rows());
    BC.setFromTriplets(trips.begin(), trips.end());
}
