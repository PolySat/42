#include <Python.h>

#include "42types.h"

#ifndef __SIMULATION_H__
#define __SIMULATION_H__

typedef struct Simulation {
   PyObject_HEAD
   PyObject *sc_list;
} nasa42_Simulation;

extern PyTypeObject nasa42_SimulationType;

#endif