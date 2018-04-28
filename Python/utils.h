#include <Python.h>

#ifndef __UTILS_H__
#define __UTILS_H__

PyObject* pyarray_from_dblarr(double *arr, unsigned int len);
char pyarg_parse_frame(PyObject *args, PyObject *kwds);

#endif