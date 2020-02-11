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

PyDoc_STRVAR(nasa42_Spacecraft_set_mtb__doc__,
   "set_mtb(name, mag_mmt) -> None\n\n"
   "Set magnetorquer magnetic moment.");

static PyObject*
nasa42_Spacecraft_set_mtb(PyObject *self, PyObject *args, PyObject *kwds)
{
   nasa42_Spacecraft *sc = (nasa42_Spacecraft *)self;
   const char *dev_name;
   int i = 0;
   double mmt;

   static char *kwlist[] = {"name", "mag_mmt", NULL};

   if (!PyArg_ParseTupleAndKeywords(args, kwds, "sd", kwlist, &dev_name, &mmt) ) {
      PyErr_SetString(PyExc_RuntimeError, "set_mtb parameters incorrect.");
      return NULL;
   }

   // Find the MTB object by name
   for (; i < sc->sc->Nmtb && strcmp(sc->sc->MTB[i].name, dev_name); i++) {}

   if (i == sc->sc->Nmtb) {
      PyErr_Format(PyExc_ValueError, "Magnetorquer device name %s does not exist.", dev_name);
      return NULL;
   }

   sc->sc->AC.MTB[i].Mcmd = mmt;
   Py_INCREF(Py_None);
   return Py_None;
}

PyDoc_STRVAR(nasa42_Spacecraft_set_wheel__doc__,
   "set_wheel(name, torque) -> None\n\n"
   "Set reaciton wheel torque.");

static PyObject*
nasa42_Spacecraft_set_wheel(PyObject *self, PyObject *args, PyObject *kwds)
{
   nasa42_Spacecraft *sc = (nasa42_Spacecraft *)self;
   const char *dev_name;
   int i = 0;
   double torque;

   static char *kwlist[] = {"name", "torque", NULL};

   if (!PyArg_ParseTupleAndKeywords(args, kwds, "sd", kwlist, &dev_name, &torque) ) {
      PyErr_SetString(PyExc_RuntimeError, "set_mtb parameters incorrect.");
      return NULL;
   }

   // Find the MTB object by name
   for (; i < sc->sc->Nw && strcmp(sc->sc->Whl[i].name, dev_name); i++) {}

   if (i == sc->sc->Nw) {
      PyErr_Format(PyExc_ValueError, "Wheel device name %s does not exist.", dev_name);
      return NULL;
   }

   sc->sc->AC.Whl[i].Tcmd = torque;
   Py_INCREF(Py_None);
   return Py_None;
}

PyDoc_STRVAR(nasa42_Spacecraft_position__doc__,
   "position(frame) -> ndarray\n\n"
   "Position vector of spacecraft (m).");

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
            "No support for position in WORLD_ROTATING_FRAME");
         break;

      case LVLH_FRAME:
         PyErr_SetString(PyExc_NotImplementedError,
            "No support for position in LVLH_FRAME");
         break;

      case BODY_FRAME:
         PyErr_SetString(PyExc_NotImplementedError,
            "No support for position in BODY_FRAME");
         break;

      default:
         PyErr_Format(PyExc_ValueError, "Frame value %d not recognized.", frame);
         break;
   }

   if (!vec)
      return NULL;

   return pyarray_from_dblarray(3, vec);
}

PyDoc_STRVAR(nasa42_Spacecraft_velocity__doc__,
   "velocity(frame) -> ndarray\n\n"
   "Velocity vector of spacecraft (m/s).");

static PyObject*
nasa42_Spacecraft_velocity(PyObject *self, PyObject *args, PyObject *kwds)
{
   nasa42_Spacecraft *sc = (nasa42_Spacecraft *)self;
   double *vec = NULL;
   char frame;

   frame = pyarg_parse_frame(args, kwds);
   if (!frame)
      return NULL;

   switch (frame) {
      case HELIOCENTRIC_FRAME:
         vec = sc->sc->VelH;
         break;

      case WORLD_INERTIAL_FRAME:
         vec = sc->sc->VelN;
         break;

      case WORLD_ROTATING_FRAME:
         PyErr_SetString(PyExc_NotImplementedError,
            "No support for velocity in WORLD_ROTATING_FRAME");
         break;

      case LVLH_FRAME:
         PyErr_SetString(PyExc_NotImplementedError,
            "No support for velocity in LVLH_FRAME");
         break;

      case BODY_FRAME:
         PyErr_SetString(PyExc_NotImplementedError,
            "No support for velocity in BODY_FRAME");
         break;

      default:
         PyErr_Format(PyExc_ValueError, "Frame value %d not recognized.", frame);
         break;
   }

   if (!vec)
      return NULL;

   return pyarray_from_dblarray(3, vec);
}

PyDoc_STRVAR(nasa42_Spacecraft_mag_field__doc__,
   "mag_field(frame) -> ndarray\n\n"
   "Magnetic field vector for the current spacecraft position (T).");

static PyObject*
nasa42_Spacecraft_mag_field(PyObject *self, PyObject *args, PyObject *kwds)
{
   nasa42_Spacecraft *sc = (nasa42_Spacecraft *)self;
   double *vec = NULL;
   char frame;

   frame = pyarg_parse_frame(args, kwds);
   if (!frame)
      return NULL;

   switch (frame) {
      case HELIOCENTRIC_FRAME:
         PyErr_SetString(PyExc_NotImplementedError,
            "No support for magnetic field in HELIOCENTRIC_FRAME");
         break;

      case WORLD_INERTIAL_FRAME:
         vec = sc->sc->bvn;
         break;

      case WORLD_ROTATING_FRAME:
         PyErr_SetString(PyExc_NotImplementedError,
            "No support for magnetic field in WORLD_ROTATING_FRAME");
         break;

      case LVLH_FRAME:
         PyErr_SetString(PyExc_NotImplementedError,
            "No support for magnetic field in LVLH_FRAME");
         break;

      case BODY_FRAME:
         vec = sc->sc->bvb;
         break;

      default:
         PyErr_Format(PyExc_ValueError, "Frame value %d not recognized.", frame);
         break;
   }

   if (!vec)
      return NULL;

   return pyarray_from_dblarray(3, vec);
}

PyDoc_STRVAR(nasa42_Spacecraft_sun_vec__doc__,
   "sun_vec(frame) -> ndarray\n\n"
   "Sun-pointing unit vector for the current spacecraft position.");

static PyObject*
nasa42_Spacecraft_sun_vec(PyObject *self, PyObject *args, PyObject *kwds)
{
   nasa42_Spacecraft *sc = (nasa42_Spacecraft *)self;
   double *vec = NULL;
   char frame;

   frame = pyarg_parse_frame(args, kwds);
   if (!frame)
      return NULL;

   switch (frame) {
      case HELIOCENTRIC_FRAME:
         PyErr_SetString(PyExc_NotImplementedError,
            "No support for sun vector in HELIOCENTRIC_FRAME");
         break;

      case WORLD_INERTIAL_FRAME:
         vec = sc->sc->svn;
         break;

      case WORLD_ROTATING_FRAME:
         PyErr_SetString(PyExc_NotImplementedError,
            "No support for sun vector in WORLD_ROTATING_FRAME");
         break;

      case LVLH_FRAME:
         PyErr_SetString(PyExc_NotImplementedError,
            "No support for sun vector in LVLH_FRAME");
         break;

      case BODY_FRAME:
         vec = sc->sc->svb;
         break;

      default:
         PyErr_Format(PyExc_ValueError, "Frame value %d not recognized.", frame);
         break;
   }

   if (!vec)
      return NULL;

   return pyarray_from_dblarray(3, vec);
}

static PyObject*
nasa42_Spacecraft_quaternion(PyObject *self, void *arg)
{
   nasa42_Spacecraft *sc = (nasa42_Spacecraft *)self;
   return pyarray_from_dblarray(4, sc->sc->B[0].qn);
}

static PyObject*
nasa42_Spacecraft_ang_vel(PyObject *self, void *arg)
{
   nasa42_Spacecraft *sc = (nasa42_Spacecraft *)self;
   return pyarray_from_dblarray(3, sc->sc->B[0].wn);
}

static PyObject*
nasa42_Spacecraft_dcm_lvlh_inertial(PyObject *self, void *arg)
{
   nasa42_Spacecraft *sc = (nasa42_Spacecraft *)self;
   return pymatrix_from_dblmatrix(3, 3, sc->sc->CLN);
}

/**
 * Returns a Python dictionay of readings from all gyros on the spacecraft.
 * Each entry in the dictionary is formatted as follows:
 *    "Name" : [ x, y, z, Rate ]
 * Where  x, y, z Represent a unit vector in the direction of the gyro
 * and Rate is the reading from the gyro.
 */
static PyObject *
nasa42_Spacecraft_gyros(PyObject *self, void *arg)
{
   nasa42_Spacecraft *sc = (nasa42_Spacecraft *)self;
   PyObject *gyroDict = PyDict_New();
   if (!gyroDict)
      return NULL;

   for (size_t i = 0; i < sc->sc->AC.Ngyro; i++)
   {
      struct AcGyroType gyro = sc->sc->AC.Gyro[i];
      double gyroData[] = {gyro.Axis[0], gyro.Axis[1], gyro.Axis[2], gyro.Rate};
      PyDict_SetItemString(gyroDict, sc->sc->Gyro[i].name, pyarray_from_dblarray(4, gyroData));
   }
   return gyroDict;
}

/**
 * Returns a Python dictionay of all speeds of all wheels on the spacecraft.
 * Each entry in the dictionary is formatted as follows:
 *    "Name" : W
 * Where W is a float representing the speed of the wheel.
 */
static PyObject *
nasa42_Spacecraft_wheels(PyObject *self, void *arg)
{
   nasa42_Spacecraft *sc = (nasa42_Spacecraft *)self;
   PyObject *wheelDict = PyDict_New();
   if (!wheelDict)
      return NULL;

   for (size_t i = 0; i < sc->sc->AC.Nwhl; i++)
      PyDict_SetItemString(wheelDict, sc->sc->Whl[i].name, PyFloat_FromDouble(sc->sc->AC.Whl[i].w));

   return wheelDict;
}

static PyMethodDef nasa42_Spacecraft_methods[] = {
    {"set_mtb", (PyCFunction)nasa42_Spacecraft_set_mtb, METH_VARARGS | METH_KEYWORDS, nasa42_Spacecraft_set_mtb__doc__},
    {"set_wheel",(PyCFunction)nasa42_Spacecraft_set_wheel, METH_VARARGS | METH_KEYWORDS, nasa42_Spacecraft_set_wheel__doc__},
    {"position", (PyCFunction)nasa42_Spacecraft_position, METH_VARARGS | METH_KEYWORDS, nasa42_Spacecraft_position__doc__},
    {"velocity", (PyCFunction)nasa42_Spacecraft_velocity, METH_VARARGS | METH_KEYWORDS, nasa42_Spacecraft_velocity__doc__},
    {"mag_field", (PyCFunction)nasa42_Spacecraft_mag_field, METH_VARARGS | METH_KEYWORDS, nasa42_Spacecraft_mag_field__doc__},
    {"sun_vec", (PyCFunction)nasa42_Spacecraft_sun_vec, METH_VARARGS | METH_KEYWORDS, nasa42_Spacecraft_sun_vec__doc__},
    {NULL, NULL, 0, NULL}
    };

static PyGetSetDef nasa42_Spacecraft_getset[] = {
    {"quaternion", nasa42_Spacecraft_quaternion, NULL, "Satellite quaternion from ECI to Body.", NULL},
    {"ang_vel", nasa42_Spacecraft_ang_vel, NULL, "Angular velocity of Body in Inertial Frame (rads/s).", NULL},
    {"dcm_lvlh_inertial", nasa42_Spacecraft_dcm_lvlh_inertial, NULL, "Rotation matrix from inertial to lvlh frame.", NULL},

    {"gyros", nasa42_Spacecraft_gyros, NULL, "Dictonary of all gyros.", NULL},
    {"wheels", nasa42_Spacecraft_wheels, NULL, "Dictonary of all wheels.", NULL},

    {NULL, NULL, NULL, NULL, NULL}
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
    0,                         /* tp_members */
    nasa42_Spacecraft_getset,  /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)nasa42_Spacecraft_init,      /* tp_init */
};
