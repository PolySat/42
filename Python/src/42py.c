/*    This file is distributed with 42,                               */
/*    the (mostly harmless) spacecraft dynamics simulation            */
/*    created by Eric Stoneking of NASA Goddard Space Flight Center   */

/*    Copyright 2010 United States Government                         */
/*    as represented by the Administrator                             */
/*    of the National Aeronautics and Space Administration.           */

/*    No copyright is claimed in the United States                    */
/*    under Title 17, U.S. Code.                                      */

/*    All Other Rights Reserved.                                      */

#include <Python.h>

// numpy requires this macro to be used across multiple files.
#define PY_ARRAY_UNIQUE_SYMBOL nasa42_ARRAY_API
#include <numpy/arrayobject.h>

#include "42defines.h"
#include "spacecraft.h"
#include "simulation.h"

static PyObject *Py42Error = NULL;

static PyMethodDef nasa42_funcs[] = {
   {NULL, NULL, 0, NULL}
};

static struct PyModuleDef nasa42module = {
   PyModuleDef_HEAD_INIT,
   "nasa42",
   "Nasa42 Spaceraft Dynamics Simulator Python Bindings",
   -1,
   nasa42_funcs
};

PyMODINIT_FUNC
PyInit_nasa42(void)
{
   PyObject *m;
   Py42Error = PyErr_NewException("nasa42.Error", NULL, NULL);
   
   nasa42_SimulationType.tp_new = PyType_GenericNew;
   if(PyType_Ready(&nasa42_SimulationType) < 0)
      return NULL;
   nasa42_SpacecraftType.tp_new = PyType_GenericNew;
   if(PyType_Ready(&nasa42_SpacecraftType) < 0)
      return NULL;

   m = PyModule_Create(&nasa42module);
   if (m == NULL)
      return NULL;
   PyModule_AddObject(m, "Error", Py42Error);
   
   Py_INCREF(&nasa42_SimulationType);
   PyModule_AddObject(m, "Simulation", (PyObject *)&nasa42_SimulationType);
   Py_INCREF(&nasa42_SpacecraftType);
   PyModule_AddObject(m, "Spacecraft", (PyObject *)&nasa42_SpacecraftType);

   // Add module constants
   if (PyModule_AddIntConstant(m, "LVLH_FRAME", LVLH_FRAME) < 0)
      return NULL;
   if (PyModule_AddIntConstant(m, "BODY_FRAME", BODY_FRAME) < 0)
      return NULL;
   if (PyModule_AddIntConstant(m, "HELIOCENTRIC_FRAME", HELIOCENTRIC_FRAME) < 0)
      return NULL;
   if (PyModule_AddIntConstant(m, "WORLD_INERTIAL_FRAME", WORLD_INERTIAL_FRAME) < 0)
      return NULL;
   if (PyModule_AddIntConstant(m, "WORLD_ROTATING_FRAME", WORLD_ROTATING_FRAME) < 0)
      return NULL;

   // Import numpy
   import_array();

   return m;
}

