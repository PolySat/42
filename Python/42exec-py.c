/*    This file is distributed with 42,                               */
/*    the (mostly harmless) spacecraft dynamics simulation            */
/*    created by Eric Stoneking of NASA Goddard Space Flight Center   */

/*    Copyright 2010 United States Government                         */
/*    as represented by the Administrator                             */
/*    of the National Aeronautics and Space Administration.           */

/*    No copyright is claimed in the United States                    */
/*    under Title 17, U.S. Code.                                      */

/*    All Other Rights Reserved.                                      */


#include "42.h"
#include "42defines.h"
#include "42exec.h"
#include "Python.h"

static PyObject *Py42Error = NULL;

static PyObject *nasa42_set_mtb(PyObject *self, PyObject *args, PyObject *kwds)
{
   unsigned int def_id = 0;
   unsigned int *id, *sc_id = &def_id;
   double *mmt;
   
   static char *kwlist[] = {"mag_id", "mag_mmt", "spacecraft_id", NULL};

   if (!PyArg_ParseTupleAndKeywords(args, kwds, "Id|I", kwlist, &id, &mmt, &sc_id) ) {
      PyErr_SetString(PyExc_RuntimeError, "set_mtb parameters incorrect.");
      return NULL; 
   }
   
   if (*sc_id >= Nsc) {
      PyErr_Format(PyExc_ValueError, "Spacecraft ID %d does not exist.", sc_id);
      return NULL;
   }
   
   if (*id >= SC[*sc_id].Nmtb) {
      PyErr_Format(PyExc_ValueError, "Magnetorquer ID %d does not exist.", id);
      return NULL;
   }

   SC[*sc_id].FSW.Mmtbcmd[*id] = *mmt;
   Py_INCREF(Py_None);
   return Py_None;
}

static PyObject *nasa42_set_wheel(PyObject *self, PyObject *args, PyObject *kwds)
{
   unsigned int def_id = 0;
   unsigned int *id, *sc_id = &def_id;
   double *torque;
   
   static char *kwlist[] = {"whl_id", "whl_torque", "spacecraft_id", NULL};

   if (!PyArg_ParseTupleAndKeywords(args, kwds, "Id|I", kwlist, &id, &torque, &sc_id) ) {
      PyErr_SetString(PyExc_RuntimeError, "set_mtb parameters incorrect.");
      return NULL; 
   }
   
   if (*sc_id >= Nsc) {
      PyErr_Format(PyExc_ValueError, "Spacecraft ID %d does not exist.", sc_id);
      return NULL;
   }
   
   if (*id >= SC[*sc_id].Nw) {
      PyErr_Format(PyExc_ValueError, "Wheel ID %d does not exist.", id);
      return NULL;
   }

   SC[*sc_id].FSW.Twhlcmd[*id] = *torque;
   Py_INCREF(Py_None);
   return Py_None;
}

static PyObject *nasa42_propagate(PyObject *self, PyObject *args, PyObject *kwds)
{
   long Done = 0, stop_time;

   if (!SC) {
      PyErr_SetString(Py42Error, "Module not initialized");
      return NULL;
   }

   static char *kwlist[] = {"stop_time", NULL};

   if (!PyArg_ParseTupleAndKeywords(args, kwds, "l", kwlist, &stop_time) ) {
      PyErr_SetString(PyExc_RuntimeError, "Must provide propagation stop time.");
      return NULL; 
   }

   // Convert stop time to J2000
   stop_time += UNIX_EPOCH;
   if (stop_time <= AbsTime) {
      PyErr_SetString(PyExc_ValueError, "Stop time must be greated than current sim time.");
      return NULL;
   }

   STOPTIME = (double)(stop_time - AbsTime0);

   // Not going to do GUI for now. For GUI to work, will need to be running
   // in another thread, accessesing a shared mem location.

   /* Crunch numbers till done */
   while(!Done) {
      Done = SimStep();
   }

   Py_INCREF(Py_None);
   return Py_None;
}

static PyMethodDef nasa42_funcs[] = {
   {"propagate", nasa42_propagate, METH_VARARGS | METH_KEYWORDS, "Propagate satellite state to new time"},
   {"set_mtb", nasa42_set_mtb, METH_VARARGS | METH_KEYWORDS, "Set magnetorquer magnetic moment"},
   {"set_wheel", nasa42_set_wheel, METH_VARARGS | METH_KEYWORDS, "Set reaction wheel torque"},
   {NULL, NULL, 0, NULL}        /* Sentinel */
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
   PyObject* m;
   long Isc;
   char *argv[] = {"python-nasa42"};

   Py42Error = PyErr_NewException("nasa42.Error", NULL, NULL);

   m = PyModule_Create(&nasa42module);
   if (m == NULL)
      return NULL;
   PyModule_AddObject(m, "Error", Py42Error);

   // Initialize Nasa42 configuration
   InitSim(1, argv);
   for (Isc=0;Isc<Nsc;Isc++) {
      if (SC[Isc].Exists) {
         InitSpacecraft(&SC[Isc]);
         InitFSW(&SC[Isc]);
      }
   }
   
   return m;
}

