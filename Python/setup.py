#!/usr/bin/python

from distutils.core import setup, Extension
setup(name='nasa42', version='1.0',  \
      ext_modules=[Extension('nasa42', libraries = ['nasa42'], sources = ['42exec-py.c'])])
