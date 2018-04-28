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
#include <structmember.h>
#include <numpy/arrayobject.h>
#include <stddef.h>


#include "simulation.h"
#include "42.h"
#include "42defines.h"
#include "42exec.h"
#include "spacecraft.h"


static void
nasa42_Simulation_dealloc(nasa42_Simulation *self)
{
   self->ob_base.ob_type->tp_free((PyObject *)self);
}

static int
nasa42_Simulation_init(nasa42_Simulation *self, PyObject *args)
{
   nasa42_Spacecraft *sc;
   PyObject *sc_args;
   long Isc;
   char *argv[] = {"python-nasa42"};
   unsigned int i;

   // Initialize Nasa42 configuration
   InitSim(1, argv);
   for (Isc=0;Isc<Nsc;Isc++) {
      if (SC[Isc].Exists) {
         InitSpacecraft(&SC[Isc]);
         InitFSW(&SC[Isc]);
      }
   }

   self->sc_list = PyList_New((Py_ssize_t)Nsc);
   if (!self->sc_list)
      return -1;
   
   for (i = 0; i < Nsc; i++) {
      sc = PyObject_New(nasa42_Spacecraft, &nasa42_SpacecraftType);
      if (!sc)
         return -1;
      Py_INCREF(sc);

      sc_args = Py_BuildValue("OI", self, i);
      if (sc->ob_base.ob_type->tp_init((PyObject *)sc, sc_args, NULL) < 0) {
         Py_DECREF(sc);
         return -1;
      }

      PyList_SET_ITEM(self->sc_list, i, (PyObject *)sc);
   }

   return 0;
}

static PyObject*
nasa42_Simulation_propagate(PyObject *self, PyObject *args, PyObject *kwds)
{
   long Done = 0, stop_time;

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

static PyMethodDef nasa42_Simulation_methods[] = {
   {"propagate", (PyCFunction)nasa42_Simulation_propagate, METH_VARARGS | METH_KEYWORDS, "Propagate satellite state to new time"},
   {NULL, NULL, 0, NULL}
};

static PyMemberDef nasa42_Simulation_members[] = {
   {"spacecraft", T_OBJECT, offsetof(nasa42_Simulation, sc_list), READONLY, "List of Spacecraft objects."},
   {NULL, 0, 0, 0, NULL}
};

PyTypeObject nasa42_SimulationType = {
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
    nasa42_Simulation_members,             /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)nasa42_Simulation_init,      /* tp_init */
};
