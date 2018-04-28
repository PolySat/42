#include <Python.h>

#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL nasa42_ARRAY_API
#include <numpy/arrayobject.h>

#ifndef __UTILS_H__
#define __UTILS_H__

PyObject* pyarray_from_dblarray(npy_intp len, double arr[]);
PyObject* pymatrix_from_dblmatrix(npy_intp x, npy_intp y, double arr[][y]);
char pyarg_parse_frame(PyObject *args, PyObject *kwds);

#endif