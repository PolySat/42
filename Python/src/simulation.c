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
#include <stddef.h>


#include "simulation.h"
#include "42.h"
#include "42defines.h"
#include "42exec.h"
#include "spacecraft.h"

extern int HandoffToGui(int argc, char **argv);
extern long (*SimStepCB)(void);

static PyObject *stepCB = NULL;
static PyObject *stepCB_sim = NULL;
static char *model_dir;

static void
nasa42_Simulation_dealloc(nasa42_Simulation *self)
{
   Py_DECREF(self->sc_list);
   self->ob_base.ob_type->tp_free((PyObject *)self);
}

static int
nasa42_Simulation_init(nasa42_Simulation *self, PyObject *args)
{
   Py_ssize_t size;
   nasa42_Spacecraft *sc;
   PyObject *sc_args = NULL, *directory = NULL, *nasa42_data = NULL;
   long Isc;
   char *argv[] = {"python-nasa42", NULL, NULL, NULL, NULL};
   char *data_dir;
   unsigned int i;
   int res, len;

   argv[1] = "InOut";
   argv[2] = "Model";

   nasa42_data = PyImport_ImportModule("nasa42_data");
   if (!nasa42_data)
      goto error;

   directory = PyObject_GetAttrString(nasa42_data, "directory");
   if (!directory)
      goto error;

   data_dir = PyUnicode_AsUTF8AndSize(directory, &size);
   if (!data_dir)
      goto error;

   len = strlen(data_dir);
   model_dir = malloc(len + strlen("/Model") + 1);
   if (!model_dir)
      goto error;
   strcpy(model_dir, data_dir);
   strcpy(model_dir + len, "/Model");
   argv[3] = model_dir;

   // Initialize Nasa42 configuration
   InitSim(4, argv);
   for (Isc=0;Isc<Nsc;Isc++) {
      if (SC[Isc].Exists) {
         InitSpacecraft(&SC[Isc], model_dir);
      }
   }

   self->sc_list = PyList_New((Py_ssize_t)Nsc);
   if (!self->sc_list)
      goto error;

   for (i = 0; i < Nsc; i++) {
      sc = PyObject_New(nasa42_Spacecraft, &nasa42_SpacecraftType);
      if (!sc)
         goto error;
      Py_INCREF(sc);

      sc_args = Py_BuildValue("OI", self, i);
      res = sc->ob_base.ob_type->tp_init((PyObject *)sc, sc_args, NULL);
      Py_DECREF(sc_args);
      if (res < 0) {
         Py_DECREF(sc);
         goto error;
      }

      PyList_SET_ITEM(self->sc_list, i, (PyObject *)sc);
   }

   /* Step simulation once so all values are properly initialized */
   SimTime -= DTSIM;
   SimStep(model_dir);
   return 0;

error:
   if (model_dir) free(model_dir);
   Py_XDECREF(nasa42_data);
   Py_XDECREF(directory);
   Py_XDECREF(self->sc_list);
   return -1;
}

static long nasa42_Simulation_SimStepCB(void)
{
   if (stepCB) {
      void *resObj;
      long res = 1;

      PyObject *arglist = Py_BuildValue("(O)", stepCB_sim);
      // PyObject_Print(stepCB, stdout, 0);
      resObj = PyEval_CallObject(stepCB, arglist);
      Py_DECREF(arglist);
      if (!resObj) {
         if (!PyErr_Occurred())
            PyErr_SetString(PyExc_ValueError, "Nothing returned by simulation callback.\n");
         PyErr_PrintEx(0);
      }
      if (resObj && PyLong_Check(resObj))
         res = PyLong_AsLong(resObj);
      else {
         PyErr_SetString(PyExc_TypeError, "Simulation callback return type is not long.\n");
         PyErr_PrintEx(0);
      }
      if (resObj)
         Py_DECREF(resObj);

      return res;
   }
   return SimStep(model_dir);
}

static PyObject*
nasa42_Simulation_startGUI(PyObject *self, PyObject *args, PyObject *kwds)
{
   char *argv[] = { "42", NULL, NULL, NULL, NULL }, modelPath[4096], shaderPath[4096], worldPath[4096];
   PyObject *temp, *nasa42_data, *directory;
   char *data_dir;
   Py_ssize_t size;

   static char *kwlist[] = {"step_callback", NULL};
   if (!PyArg_ParseTupleAndKeywords(args, kwds, "O", kwlist, &temp) ) {
      PyErr_SetString(PyExc_RuntimeError, "Must provide step method callback.");
      return NULL;
   }
   if (stepCB)
      Py_XDECREF(stepCB);

   if (temp && !PyCallable_Check(temp)) {
      PyErr_SetString(PyExc_RuntimeError, "Must provide a method to callback.");
      return NULL;
   }

   // Setup use callback
   stepCB = temp;
   if (stepCB)
      Py_XINCREF(stepCB);

   if (stepCB_sim)
      Py_XDECREF(stepCB_sim);

   stepCB_sim = self;
   Py_XINCREF(stepCB_sim);

   SimStepCB = &nasa42_Simulation_SimStepCB;

   nasa42_data = PyImport_ImportModule("nasa42_data");
   if (!nasa42_data)
      return NULL;
   directory = PyObject_GetAttrString(nasa42_data, "directory");
   if (!directory)
      return NULL;

   data_dir = PyUnicode_AsUTF8AndSize(directory, &size);
   if (!data_dir)
      return NULL;

   // Setup asset directory path
   sprintf(modelPath, "%s/%s", data_dir, "Model");
   sprintf(shaderPath, "%s/%s", data_dir, "Kit/Shaders");
   sprintf(worldPath, "%s/%s", data_dir, "World");
   argv[1] = modelPath;
   argv[2] = shaderPath;
   argv[3] = worldPath;

   HandoffToGui(4, argv);

   Py_XINCREF(Py_None);
   return Py_None;
}

static PyObject*
nasa42_Simulation_propagate(PyObject *self, PyObject *args, PyObject *kwds)
{
   double stop_time;

   static char *kwlist[] = {"stop_time", NULL};
   if (!PyArg_ParseTupleAndKeywords(args, kwds, "d", kwlist, &stop_time) ) {
      PyErr_SetString(PyExc_RuntimeError, "Must provide propagation stop time.");
      return NULL;
   }

   // Convert stop time to J2000
   stop_time += UNIX_EPOCH;
   if (stop_time <= AbsTime) {
      PyErr_SetString(PyExc_ValueError, "Stop time must be greater than current sim time.");
      return NULL;
   }

   /* Crunch numbers till done */
   while(stop_time > AbsTime)
      SimStep(model_dir);

   Py_INCREF(Py_None);
   return Py_None;
}

static PyObject*
nasa42_Simulation_SimStep(PyObject *self, PyObject *args, PyObject *kwds)
{
    long done;

    done = SimStep(model_dir);

    return PyLong_FromLong(done);
}

static PyObject*
nasa42_Simulation_epoch(PyObject *self, void *arg)
{
   return PyFloat_FromDouble(AbsTime - UNIX_EPOCH);
}

static PyMethodDef nasa42_Simulation_methods[] = {
   {"propagate", (PyCFunction)nasa42_Simulation_propagate, METH_VARARGS | METH_KEYWORDS, "Propagate satellite state to new time"},
   {"startGUI", (PyCFunction)nasa42_Simulation_startGUI, METH_VARARGS | METH_KEYWORDS, "Display the 42 GUI"},
   {"SimStep", (PyCFunction)nasa42_Simulation_SimStep, METH_VARARGS | METH_KEYWORDS, "Perform one simulation step"},
   {NULL, NULL, 0, NULL}
};

static PyGetSetDef nasa42_Simulation_getset[] = {
   {"epoch", nasa42_Simulation_epoch, NULL, "Current simulation epoch time.", NULL},
   {NULL, NULL, NULL, NULL, NULL}
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
    nasa42_Simulation_getset,              /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)nasa42_Simulation_init,      /* tp_init */
};
