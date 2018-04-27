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
#include <numpy/arrayobject.h>


#include "42.h"
#include "42defines.h"
#include "42exec.h"

typedef struct Simulation {
   PyObject_HEAD

} nasa42_Simulation;

static PyObject *Py42Error = NULL;

static void
nasa42_Simulation_dealloc(nasa42_Simulation *self)
{
   self->ob_base.ob_type->tp_free((PyObject *)self);
}

static int
nasa42_Simulation_init(nasa42_Simulation *self, PyObject *args)
{
   long Isc;
   char *argv[] = {"python-nasa42"};

   // Initialize Nasa42 configuration
   InitSim(1, argv);
   for (Isc=0;Isc<Nsc;Isc++) {
      if (SC[Isc].Exists) {
         InitSpacecraft(&SC[Isc]);
         InitFSW(&SC[Isc]);
      }
   }

   return 0;
}

static PyObject*
nasa42_Simulation_set_mtb(PyObject *self, PyObject *args, PyObject *kwds)
{
   unsigned int id, sc_id = 0;
   double mmt;

   static char *kwlist[] = {"mag_id", "mag_mmt", "spacecraft_id", NULL};

   if (!PyArg_ParseTupleAndKeywords(args, kwds, "Id|I", kwlist, &id, &mmt, &sc_id) ) {
      PyErr_SetString(PyExc_RuntimeError, "set_mtb parameters incorrect.");
      return NULL; 
   }

   if (sc_id >= Nsc) {
      PyErr_Format(PyExc_ValueError, "Spacecraft ID %d does not exist.", sc_id);
      return NULL;
   }
   
   if (id >= SC[sc_id].Nmtb) {
      PyErr_Format(PyExc_ValueError, "Magnetorquer ID %d does not exist.", id);
      return NULL;
   }

   SC[sc_id].FSW.Mmtbcmd[id] = mmt;
   Py_INCREF(Py_None);
   return Py_None;
}

static PyObject*
nasa42_Simulation_set_wheel(PyObject *self, PyObject *args, PyObject *kwds)
{
   unsigned int id, sc_id = 0;
   double torque;
   
   static char *kwlist[] = {"whl_id", "whl_torque", "spacecraft_id", NULL};

   if (!PyArg_ParseTupleAndKeywords(args, kwds, "Id|I", kwlist, &id, &torque, &sc_id) ) {
      PyErr_SetString(PyExc_RuntimeError, "set_mtb parameters incorrect.");
      return NULL; 
   }
   
   if (sc_id >= Nsc) {
      PyErr_Format(PyExc_ValueError, "Spacecraft ID %d does not exist.", sc_id);
      return NULL;
   }
   
   if (id >= SC[sc_id].Nw) {
      PyErr_Format(PyExc_ValueError, "Wheel ID %d does not exist.", id);
      return NULL;
   }

   SC[sc_id].FSW.Twhlcmd[id] = torque;
   Py_INCREF(Py_None);
   return Py_None;
}

static PyObject*
nasa42_Simulation_propagate(PyObject *self, PyObject *args, PyObject *kwds)
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

static PyObject*
pyarray_from_doubles(double *arr, unsigned int len)
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

static PyObject*
nasa42_Simulation_position(PyObject *self, void *arg)
{
   return pyarray_from_doubles(SC[0].PosN, 3);
}

static PyMethodDef nasa42_Simulation_methods[] = {
   {"propagate", (PyCFunction)nasa42_Simulation_propagate, METH_VARARGS | METH_KEYWORDS, "Propagate satellite state to new time"},
   {"set_mtb", (PyCFunction)nasa42_Simulation_set_mtb, METH_VARARGS | METH_KEYWORDS, "Set magnetorquer magnetic moment"},
   {"set_wheel",(PyCFunction)nasa42_Simulation_set_wheel, METH_VARARGS | METH_KEYWORDS, "Set reaction wheel torque"},
   {NULL, NULL, 0, NULL}
};

static PyGetSetDef nasa42_Simulation_getset[] = {
   {"position", nasa42_Simulation_position, NULL, "Get position of satellite in ECI (km)", NULL},
   {NULL, NULL, NULL, NULL, NULL}
};

static PyTypeObject nasa42_SimulationType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "nasa42.Simulation",             /* tp_name */
    sizeof(nasa42_Simulation), /* tp_basicsize */
    0,                         /* tp_itemsize */
    (destructor)nasa42_Simulation_dealloc, /* tp_dealloc */
    0,                         /* tp_print */
    0,                         /* tp_getattr */
    0,                         /* tp_setattr */
    0,                         /* tp_reserved */
    0,                         /* tp_repr */
    0,                         /* tp_as_number */
    0,                         /* tp_as_sequence */
    0,                         /* tp_as_mapping */
    0,                         /* tp_hash  */
    0,                         /* tp_call */
    0,                         /* tp_str */
    0,                         /* tp_getattro */
    0,                         /* tp_setattro */
    0,                         /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,        /* tp_flags */
    "Nasa 42 Simulation Object",           /* tp_doc */
    0,		               /* tp_traverse */
    0,		               /* tp_clear */
    0,		               /* tp_richcompare */
    0,		               /* tp_weaklistoffset */
    0,		               /* tp_iter */
    0,		               /* tp_iternext */
    nasa42_Simulation_methods,             /* tp_methods */
    0,             /* tp_members */
    nasa42_Simulation_getset,  /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)nasa42_Simulation_init,      /* tp_init */
};

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
   PyObject* m;
   Py42Error = PyErr_NewException("nasa42.Error", NULL, NULL);
   
   nasa42_SimulationType.tp_new = PyType_GenericNew;
   if(PyType_Ready(&nasa42_SimulationType) < 0)
      return NULL;

   m = PyModule_Create(&nasa42module);
   if (m == NULL)
      return NULL;
   PyModule_AddObject(m, "Error", Py42Error);
   
   Py_INCREF(&nasa42_SimulationType);
   PyModule_AddObject(m, "Simulation", (PyObject *)&nasa42_SimulationType);

   // Import numpy
   import_array();

   return m;
}

