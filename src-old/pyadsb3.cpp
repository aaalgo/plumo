#ifdef _OPENMP
#include <omp.h>
#endif
#include <Python.h>
#include <numpy/ndarrayobject.h>
#include <limits>
#include <iostream>
#include <stdexcept>
#include <boost/assert.hpp>
#include <boost/python.hpp>
#include "mesh.h"

using namespace std;
namespace python = boost::python;

namespace {

    void shrink (PyArrayObject *array, float ratio) {
        Py_INCREF(array);
        if (array->nd != 2) throw runtime_error("not 2d array");
        if (array->descr->type_num != NPY_BOOL) throw runtime_error("not bool array");
        //auto rows = array->dimensions[0];
        //auto cols = array->dimensions[1];
        //cout << "XX " << rows << "x" << cols << endl;
        Py_DECREF(array);
    }

    void hull (PyObject *_array, PyObject *_dest) {
        PyArrayObject *array((PyArrayObject *)_array);
        PyArrayObject *dest((PyArrayObject *)_dest);
        if (array->nd != 3) throw runtime_error("not 3d array");
        if (dest->nd != 3) throw runtime_error("not 3d array");
        if (array->descr->type_num != NPY_UINT8) throw runtime_error("not uint8 array");
        if (dest->descr->type_num != NPY_UINT8) throw runtime_error("not uint8 array");
        static unsigned orders[][4] = {
            {1,2,0, 0},
            {2,0,1, 0},
            {0,1,2, 0}
        };
        for (unsigned no = 0; no < 3; ++no) {
            unsigned d1 = orders[no][0];
            unsigned d2 = orders[no][1];
            unsigned d3 = orders[no][2];
            unsigned fill0 = orders[no][3];
            auto Z = array->dimensions[d1];
            auto Y = array->dimensions[d2];
            auto X = array->dimensions[d3];
            auto from_sZ = array->strides[d1];
            auto from_sY = array->strides[d2];
            auto from_sX = array->strides[d3];
            auto to_sZ = dest->strides[d1];
            auto to_sY = dest->strides[d2];
            auto to_sX = dest->strides[d3];
            //cout << Z << ':' << from_sZ << ' ' << Y << ':' << from_sY << ' ' << X << ':' << from_sX << endl;
            auto from = reinterpret_cast<uint8_t const *>(array->data);
            auto to = reinterpret_cast<uint8_t *>(dest->data);
            for (auto z = 0; z < Z; ++z) {
                auto from_y = from;
                auto to_y = to;
                for (auto y = 0; y < Y; ++y) {
                    auto from_x = from_y;
                    auto to_x = to_y;

                    long int lb = 0;
                    for (long int x = 0, o1 = 0, o2 = 0;
                              x < X;
                              ++x, o1 += from_sX, o2 += to_sX) {  // forward
                        if (from_x[o1]) {
                            lb = o2;
                            to_x[o2] = 1;
                            break;
                        }
                    }
                    long int rb = (X-1) * to_sX;
                    for (long int x = 0, o1 = (X-1) * from_sX, o2 = (X-1) * to_sX;
                            x < X;
                            ++x, o1 -= from_sX, o2 -= to_sX) {  // backward
                        if (from_x[o1]) {
                            rb = o2;
                            to_x[o2] = 1;
                            break;
                        }
                    }
                    if (fill0) for (lb += to_sX; lb < rb; lb += to_sX) {
                        to_x[lb] = 0;
                    }
                    from_y += from_sY;
                    to_y += to_sY;
                }
                from += from_sZ;
                to += to_sZ;
            }
        }
    }

    void color_mesh (PyObject *_v, PyObject *_f, string const &path) {
        PyArrayObject *verts((PyArrayObject *)_v);
        PyArrayObject *faces((PyArrayObject *)_f);

        Mesh m;

        unsigned nv = verts->dimensions[0];
        unsigned vs = verts->strides[0];
        CHECK(verts->nd == 2);
        CHECK(verts->descr->type_num == NPY_FLOAT64);
        CHECK(verts->dimensions[1] == 3);
        Mesh::VertexIterator vi = vcg::tri::Allocator<Mesh>::AddVertices(m, nv);
        Mesh::VertexIterator v0 = vi;
        char const *pv = verts->data;
        for (unsigned i = 0; i < nv; ++i, pv += vs, ++vi) {
            double const *pp = reinterpret_cast<double const *>(pv);
            vi->P() = Mesh::CoordType(pp[2], pp[1], pp[0]);
        }

        unsigned nf = faces->dimensions[0];
        unsigned fs = faces->strides[0];
        CHECK(faces->nd == 2);
        CHECK(faces->descr->type_num == NPY_INT64);
        CHECK(faces->dimensions[1] == 3);
        Mesh::FaceIterator fi = vcg::tri::Allocator<::Mesh>::AddFaces(m, nf);
        char const *pf = faces->data;
        for (unsigned i = 0; i < nf; ++i, pf += fs, ++fi) {
            int64_t const *pp = reinterpret_cast<int64_t const *>(pf);
            fi->V(0) = &(*(v0 + pp[0]));
            fi->V(1) = &(*(v0 + pp[1]));
            fi->V(2) = &(*(v0 + pp[2]));
        }
        //vcg::tri::io::ExporterPLY<Mesh>::Save(m, path.c_str(), vcg::tri::io::SAVE_MASK);
        MeshModelParams params;
        MeshModel model(params);
        model.apply(m);
        vcg::tri::io::ExporterPLY<Mesh>::Save(m, path.c_str(), vcg::tri::io::SAVE_MASK );
    }

    void save_mesh (PyObject *_v, PyObject *_f, string const &path) {
        PyArrayObject *verts((PyArrayObject *)_v);
        PyArrayObject *faces((PyArrayObject *)_f);

        Mesh m;

        unsigned nv = verts->dimensions[0];
        unsigned vs = verts->strides[0];
        CHECK(verts->nd == 2);
        CHECK(verts->descr->type_num == NPY_FLOAT64);
        CHECK(verts->dimensions[1] == 3);
        Mesh::VertexIterator vi = vcg::tri::Allocator<Mesh>::AddVertices(m, nv);
        Mesh::VertexIterator v0 = vi;
        char const *pv = verts->data;
        for (unsigned i = 0; i < nv; ++i, pv += vs, ++vi) {
            double const *pp = reinterpret_cast<double const *>(pv);
            vi->P() = Mesh::CoordType(pp[2], pp[1], pp[0]);
        }

        unsigned nf = faces->dimensions[0];
        unsigned fs = faces->strides[0];
        CHECK(faces->nd == 2);
        CHECK(faces->descr->type_num == NPY_INT64);
        CHECK(faces->dimensions[1] == 3);
        Mesh::FaceIterator fi = vcg::tri::Allocator<::Mesh>::AddFaces(m, nf);
        char const *pf = faces->data;
        for (unsigned i = 0; i < nf; ++i, pf += fs, ++fi) {
            int64_t const *pp = reinterpret_cast<int64_t const *>(pf);
            fi->V(0) = &(*(v0 + pp[0]));
            fi->V(1) = &(*(v0 + pp[1]));
            fi->V(2) = &(*(v0 + pp[2]));
        }
        vcg::tri::io::ExporterPLY<Mesh>::Save(m, path.c_str(), vcg::tri::io::SAVE_MASK );
    }

    void decode3 (uint8_t from, float *to) {
        to[2] = from % 6;
        from /= 6;
        to[1] = from % 6;
        from /= 6;
        to[0] = from;
    }

    void decode_cell (float const *from, uint8_t *label, float *ft, vector<float> *cache) {
        if (from[0] == 0) {
            label[0] = 0;
            ft[0] = ft[1] = ft[2] = ft[3] = ft[4] = ft[5] = ft[6] = ft[7] = ft[8] = 0;
        }
        else {
            if (cache->empty()) {
                cache->resize(9);
                float *cp = &cache->at(0);
                decode3(uint8_t(from[0]), cp);
                cp += 3;
                decode3(uint8_t(from[1]), cp);
                cp += 3;
                decode3(uint8_t(from[2]), cp);
                //*cp = uint8_t(from[2]);
            }
            float const *cp = &cache->at(0);
            label[0] = 1;
            ft[0] = cp[0];
            ft[1] = cp[1];
            ft[2] = cp[2];
            ft[3] = cp[3];
            ft[4] = cp[4];
            ft[5] = cp[5];
            ft[6] = cp[6];
            ft[7] = cp[7];
            ft[8] = cp[8];
        }
    }

    void decode_cell_old (float const *from, uint8_t *label, float *ft, vector<float> *cache) {
        if (from[0] == 0) {
            label[0] = 0;
            ft[0] = ft[1] = ft[2] = ft[3] = ft[4] = ft[5] = ft[6] = 0;
        }
        else {
            if (cache->empty()) {
                cache->resize(7);
                float *cp = &cache->at(0);
                decode3(uint8_t(from[0]), cp);
                cp += 3;
                decode3(uint8_t(from[1]), cp);
                cp += 3;
                *cp = uint8_t(from[2]);
            }
            float const *cp = &cache->at(0);
            label[0] = 1;
            ft[0] = cp[0];
            ft[1] = cp[1];
            ft[2] = cp[2];
            ft[3] = cp[3];
            ft[4] = cp[4];
            ft[5] = cp[5];
            ft[6] = cp[6];
        }
    }

    template <typename T>
    T *walk (T *p, int stride) {
        return (T *)((char *)p + stride);
    }

    template <typename T>
    T const *walk (T const *p, int stride) {
        return (T *)((char const *)p + stride);
    }

    python::tuple decode_labels (PyObject *_array) {
        PyArrayObject *array((PyArrayObject *)_array);
        if (array->nd != 4) throw runtime_error("not 4d array");
        if (array->descr->type_num != NPY_FLOAT32) throw runtime_error("not float32 array");
        if (array->dimensions[0] != 1) throw runtime_error("not rgb image");
        if (array->dimensions[3] != 3) throw runtime_error("not rgb image");
        vector<npy_intp> label_dims{1,
                                    array->dimensions[1],
                                    array->dimensions[2],
                                    1};
        vector<npy_intp> ft_dims{1,
                                 array->dimensions[1],
                                 array->dimensions[2],
                                 9};
        PyArrayObject *label = (PyArrayObject*)PyArray_SimpleNew(label_dims.size(), &label_dims[0], NPY_UINT8);
        PyArrayObject *ft = (PyArrayObject *)PyArray_SimpleNew(ft_dims.size(), &ft_dims[0], NPY_FLOAT32);

        auto from_row = reinterpret_cast<float const *>(array->data);
        auto to_lb_row = reinterpret_cast<uint8_t *>(label->data);
        auto to_ft_row = reinterpret_cast<float *>(ft->data);
        vector<float> cache;
        for (unsigned y = 0; y < label_dims[1]; ++y) {
            auto from = from_row;
            auto to_lb = to_lb_row;
            auto to_ft = to_ft_row;
            for (unsigned x = 0; x < label_dims[2]; ++x) {
                decode_cell(from, to_lb, to_ft, &cache);
                from = walk(from, array->strides[2]);
                to_lb = walk(to_lb, label->strides[2]);
                to_ft = walk(to_ft, ft->strides[2]);
            }
            from_row = walk(from_row, array->strides[1]);
            to_lb_row = walk(to_lb_row, label->strides[1]);
            to_ft_row = walk(to_ft_row, ft->strides[1]);
        }
        PyArrayObject *cache_ft;
        {
            npy_intp dim[] = {cache.size()};
            cache_ft = (PyArrayObject *)PyArray_SimpleNew(1, dim, NPY_FLOAT32);
            std::copy(cache.begin(), cache.end(), (float *)cache_ft->data);
        }

        return python::make_tuple(python::object(boost::python::handle<>((PyObject*)label)),
                          python::object(boost::python::handle<>((PyObject*)ft)),
                          python::object(boost::python::handle<>((PyObject*)cache_ft))
                          );
    }

    python::tuple decode_labels_old (PyObject *_array) {
        PyArrayObject *array((PyArrayObject *)_array);
        if (array->nd != 4) throw runtime_error("not 4d array");
        if (array->descr->type_num != NPY_FLOAT32) throw runtime_error("not float32 array");
        if (array->dimensions[0] != 1) throw runtime_error("not rgb image");
        if (array->dimensions[3] != 3) throw runtime_error("not rgb image");
        vector<npy_intp> label_dims{1,
                                    array->dimensions[1],
                                    array->dimensions[2],
                                    1};
        vector<npy_intp> ft_dims{1,
                                 array->dimensions[1],
                                 array->dimensions[2],
                                 7};
        PyArrayObject *label = (PyArrayObject*)PyArray_SimpleNew(label_dims.size(), &label_dims[0], NPY_UINT8);
        PyArrayObject *ft = (PyArrayObject *)PyArray_SimpleNew(ft_dims.size(), &ft_dims[0], NPY_FLOAT32);

        auto from_row = reinterpret_cast<float const *>(array->data);
        auto to_lb_row = reinterpret_cast<uint8_t *>(label->data);
        auto to_ft_row = reinterpret_cast<float *>(ft->data);
        vector<float> cache;
        for (unsigned y = 0; y < label_dims[1]; ++y) {
            auto from = from_row;
            auto to_lb = to_lb_row;
            auto to_ft = to_ft_row;
            for (unsigned x = 0; x < label_dims[2]; ++x) {
                decode_cell_old(from, to_lb, to_ft, &cache);
                from = walk(from, array->strides[2]);
                to_lb = walk(to_lb, label->strides[2]);
                to_ft = walk(to_ft, ft->strides[2]);
            }
            from_row = walk(from_row, array->strides[1]);
            to_lb_row = walk(to_lb_row, label->strides[1]);
            to_ft_row = walk(to_ft_row, ft->strides[1]);
        }
        PyArrayObject *cache_ft;
        {
            npy_intp dim[] = {cache.size()};
            cache_ft = (PyArrayObject *)PyArray_SimpleNew(1, dim, NPY_FLOAT32);
            std::copy(cache.begin(), cache.end(), (float *)cache_ft->data);
        }

        return python::make_tuple(python::object(boost::python::handle<>((PyObject*)label)),
                          python::object(boost::python::handle<>((PyObject*)ft)),
                          python::object(boost::python::handle<>((PyObject*)cache_ft))
                          );
    }

    python::tuple norm3d (PyObject *_array) {
        PyArrayObject *array((PyArrayObject *)_array);
        if (array->nd != 3) throw runtime_error("not 3d array");
        if (array->descr->type_num != NPY_FLOAT32) throw runtime_error("not float32 array");
        auto Z = array->dimensions[0];
        auto Y = array->dimensions[1];
        auto X = array->dimensions[2];
        auto sZ = array->strides[0];
        auto sY = array->strides[1];
        auto sX = array->strides[2];
        //cout << Z << ':' << from_sZ << ' ' << Y << ':' << from_sY << ' ' << X << ':' << from_sX << endl;
        float sum_Z = 0, sum_Y = 0, sum_X = 0, sum = 0;
        auto from_z = reinterpret_cast<uint8_t const *>(array->data);
        for (auto z = 0; z < Z; ++z, from_z += sZ) {
            auto from_y = from_z;
            for (auto y = 0; y < Y; ++y, from_y += sY) {
                auto from_x = from_y;
                for (auto x = 0; x < X; ++x, from_x += sX) {
                    float v = *reinterpret_cast<float const *>(from_x);
                    sum += v;
                    sum_Z += v * z;
                    sum_Y += v * y;
                    sum_X += v * x;
                }
            }
        }
        float avg_Z = sum_Z / sum,
              avg_Y = sum_Y / sum,
              avg_X = sum_X / sum;
        float sum_ZZ = 0, sum_YY = 0, sum_XX = 0;
        float sum_ZY = 0, sum_ZX = 0, sum_YX = 0;
        from_z = reinterpret_cast<uint8_t const *>(array->data);
        float sum2 = 0;
        for (auto z = 0; z < Z; ++z, from_z += sZ) {
            auto from_y = from_z;
            float zz = z - avg_Z;
            for (auto y = 0; y < Y; ++y, from_y += sY) {
                auto from_x = from_y;
                float yy = y - avg_Y;
                for (auto x = 0; x < X; ++x, from_x += sX) {
                    float v = *reinterpret_cast<float const *>(from_x);
                    float xx = x - avg_X;
                    sum_ZZ += v * zz * zz;
                    sum_ZY += v * yy * zz;
                    sum_ZX += v * xx * zz;
                    sum_YY += v * yy * yy;
                    sum_YX += v * xx * yy;
                    sum_XX += v * xx * xx;
                    sum2 += v;
                }
            }
        }
        CHECK(sum == sum2);
        return python::make_tuple(avg_Z, avg_Y, avg_X,
                    sum_ZZ/sum, sum_ZY/sum, sum_ZX/sum,
                                sum_YY/sum, sum_YX/sum,
                                            sum_XX/sum);
    }
}

BOOST_PYTHON_MODULE(pyadsb3)
{
    import_array();
    python::numeric::array::set_module_and_type("numpy", "ndarray");
    python::def("shrink", ::shrink);
    python::def("hull", ::hull);
    python::def("norm3d", ::norm3d);
    python::def("save_mesh", ::save_mesh);
    python::def("color_mesh", ::color_mesh);
    python::def("decode_labels", ::decode_labels);
    python::def("decode_labels_old", ::decode_labels_old);
}


