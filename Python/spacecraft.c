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

// numpy requires these macros to be used across multiple files.
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL nasa42_ARRAY_API
#include <numpy/arrayobject.h>

#include "spacecraft.h"
#include "42.h"
#include "42defines.h"
#include "utils.h"
#include "simulation.h"

static void
nasa42_Spacecraft_dealloc(nasa42_Spacecraft *self)
{
   self->ob_base.ob_type->tp_free((PyObject *)self);   
}

static int
nasa42_Spacecraft_init(nasa42_Spacecraft *self, PyObject *args, PyObject *kwds)
{
   struct PyObject *sim;
   unsigned int id;

   static char *kwlist[] = {"simulation", "id", NULL};
   if (!PyArg_ParseTupleAndKeywords(args, kwds, "OI", kwlist, &sim, &id) ) {
      PyErr_SetString(PyExc_RuntimeError, "Spcecraft parameters invalid.");
      return -1; 
   }

   if (!PyObject_TypeCheck(sim, &nasa42_SimulationType)) {
      PyErr_SetString(PyExc_TypeError, "Spacecraft requires Simulation object.");
      return -1;
   }

   if (id >= Nsc) {
      PyErr_SetString(PyExc_ValueError, "Spacecraft ID does not exist.");
      return -1;
   }

   self->id = id;
   self->sc = &SC[id];

   return 0;
}

static PyObject*
nasa42_Spacecraft_set_mtb(PyObject *self, PyObject *args, PyObject *kwds)
{
   nasa42_Spacecraft *sc = (nasa42_Spacecraft *)self;
   unsigned int id;
   double mmt;

   static char *kwlist[] = {"mag_id", "mag_mmt", NULL};

   if (!PyArg_ParseTupleAndKeywords(args, kwds, "Id", kwlist, &id, &mmt) ) {
      PyErr_SetString(PyExc_RuntimeError, "set_mtb parameters incorrect.");
      return NULL; 
   }

   if (id >= sc->sc->Nmtb) {
      PyErr_Format(PyExc_ValueError, "Magnetorquer ID %d does not exist.", id);
      return NULL;
   }

   sc->sc->FSW.Mmtbcmd[id] = mmt;
   Py_INCREF(Py_None);
   return Py_None;
}

static PyObject*
nasa42_Spacecraft_set_wheel(PyObject *self, PyObject *args, PyObject *kwds)
{
   nasa42_Spacecraft *sc = (nasa42_Spacecraft *)self;
   unsigned int id;
   double torque;
   
   static char *kwlist[] = {"whl_id", "whl_torque", NULL};

   if (!PyArg_ParseTupleAndKeywords(args, kwds, "Id", kwlist, &id, &torque) ) {
      PyErr_SetString(PyExc_RuntimeError, "set_mtb parameters incorrect.");
      return NULL; 
   }
   
   if (id >= sc->sc->Nw) {
      PyErr_Format(PyExc_ValueError, "Wheel ID %d does not exist.", id);
      return NULL;
   }

   sc->sc->FSW.Twhlcmd[id] = torque;
   Py_INCREF(Py_None);
   return Py_None;
}

static PyObject*
nasa42_Spacecraft_position(PyObject *self, PyObject *args, PyObject *kwds)
{
   nasa42_Spacecraft *sc = (nasa42_Spacecraft *)self;   
   double *vec = NULL;
   char frame;

   frame = pyarg_parse_frame(args, kwds);
   if (!frame)
      return NULL;

   switch (frame) {
      case HELIOCENTRIC_FRAME:
         vec = sc->sc->PosH;
         break;

      case WORLD_INERTIAL_FRAME:
         vec = sc->sc->PosN;
         break;

      case WORLD_ROTATING_FRAME:
         PyErr_SetString(PyExc_NotImplementedError,
            "Spacecraft does not support position in WORLD_ROTATING_FRAME");
         break;
      
      case LVLH_FRAME:
         PyErr_SetString(PyExc_NotImplementedError,
            "Spacecraft does not support position in LVLH_FRAME");
         break;
      
      case BODY_FRAME:
         PyErr_SetString(PyExc_NotImplementedError,
            "Spacecraft does not support position in BODY_FRAME");
         break;
      
      default:
         PyErr_Format(PyExc_ValueError, "Frame value %d not recognized.", frame);
         break;
   }

   if (!vec)
      return NULL;

   return pyarray_from_dblarr(vec, 3);
}

static PyMethodDef nasa42_Spacecraft_methods[] = {
   {"set_mtb", (PyCFunction)nasa42_Spacecraft_set_mtb, METH_VARARGS | METH_KEYWORDS, "Set magnetorquer magnetic moment"},
   {"set_wheel",(PyCFunction)nasa42_Spacecraft_set_wheel, METH_VARARGS | METH_KEYWORDS, "Set reaction wheel torque"},
   {"position", (PyCFunction)nasa42_Spacecraft_position, METH_VARARGS | METH_KEYWORDS, "Get position of satellite in ECI (m)"},   
   {NULL, NULL, 0, NULL}
};

PyTypeObject nasa42_SpacecraftType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "nasa42.Spacecraft",       /* tp_name */
    sizeof(nasa42_Spacecraft), /* tp_basicsize */
    0,                         /* tp_itemsize */
    (destructor)nasa42_Spacecraft_dealloc, /* tp_dealloc */
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
    "Nasa 42 Spacecraft Object",           /* tp_doc */
    0,		               /* tp_traverse */
    0,		               /* tp_clear */
    0,		               /* tp_richcompare */
    0,		               /* tp_weaklistoffset */
    0,		               /* tp_iter */
    0,		               /* tp_iternext */
    nasa42_Spacecraft_methods,             /* tp_methods */
    0,             /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)nasa42_Spacecraft_init,      /* tp_init */
};
