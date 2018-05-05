import numpy as np
from distutils.core import setup, Extension

base = './Python/src/'

setup(name='nasa42',
      version='1.0',
      include_dirs = [np.get_include()],
      packages=['nasa42_data'],
      package_dir={'nasa42_data': base + 'nasa42_data'},
      package_data={
          'nasa42_data': ['Model/*.ppm', 'Model/*.obj', 'Model/*.mtl',
                          'Model/*.txt', 'World/*.ppm', 'Kit/Shaders/*.glsl',
                          'Model/*.raw']
      },
      ext_modules=[
          Extension('nasa42',
              libraries = ['glut', 'GLU', 'GL', 'm'],
              extra_objects = ['./libnasa42.a'],
              sources = [base + '42py.c', base + 'spacecraft.c',
                        base + 'simulation.c', base + 'utils.c']
          )
      ]
)
