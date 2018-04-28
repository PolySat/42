#include <Python.h>

#include "42types.h"

#ifndef __SPACECRAFT_H__
#define __SPACECRAFT_H__

typedef struct Spacecraft {
   PyObject_HEAD
    unsigned int id;
    struct SCType *sc;
} nasa42_Spacecraft;

extern PyTypeObject nasa42_SpacecraftType;

#endif