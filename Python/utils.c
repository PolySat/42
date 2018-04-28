#include <Python.h>

// numpy requires these macros to be used across multiple files.
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL nasa42_ARRAY_API
#include <numpy/arrayobject.h>

#include "utils.h"


PyObject* pyarray_from_dblarr(double *arr, unsigned int len)
{
   PyArrayObject *vec;
   npy_intp i, dims[] = {len};
   void *ptr;

   vec = (PyArrayObject *)PyArray_EMPTY(1, dims, NPY_DOUBLE, 0);
   if (!vec)
      return NULL;

   for (i = 0; i < len; i++) {
      ptr = PyArray_GETPTR1(vec, i);
      PyArray_SETITEM(vec, ptr, PyFloat_FromDouble(arr[i]));
   }

   return (PyObject *)vec;
}

char pyarg_parse_frame(PyObject *args, PyObject *kwds)
{
   int frame;

   static char *kwlist[] = {"frame", NULL};
   if (!PyArg_ParseTupleAndKeywords(args, kwds, "i", kwlist, &frame) ) {
      PyErr_SetString(PyExc_RuntimeError, "Function requires integer frame argument.");
      return 0; 
   }

   return (char)frame;
}