import numpy as np

from distutils.core import setup, Extension
setup(name='nasa42', version='1.0', include_dirs = [np.get_include()],   \
      ext_modules=[Extension('nasa42', libraries = ['nasa42'], \
      sources = ['42py.c', 'spacecraft.c', 'simulation.c', 'utils.c'])])
