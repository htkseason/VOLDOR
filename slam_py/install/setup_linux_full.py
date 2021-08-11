from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from glob import glob
import os
import numpy
import subprocess
import shutil

# Change this if you target different host/device
nvcc_machine_code = '' #'-m64 -arch=compute_61 -code=sm_61'

gpu_sources_cpp = ' '.join(glob('../../gpu-kernels/*.cpp'))
gpu_sources_cu = ' '.join(glob('../../gpu-kernels/*.cu'))

gpu_kernel_build_cmd = f'/usr/local/cuda/bin/nvcc {gpu_sources_cpp} {gpu_sources_cu} -lib -o libgpu-kernels.so -O3 {nvcc_machine_code}'
os.system(gpu_kernel_build_cmd)

opencv_libs = subprocess.check_output('pkg-config --libs opencv'.split())
opencv_libs = [name[2:] for name in str(opencv_libs, 'utf-8').split()][1:]

ext = Extension('pyvoldor_full',
    sources = ['pyvoldor_full.pyx'] + \
            [x for x in glob('../../voldor/*.cpp') if 'main.cpp' not in x] + \
            [x for x in glob('../../frame-alignment/*.cpp') if 'main.cpp' not in x] + \
            [x for x in glob('../../pose-graph/*.cpp') if 'main.cpp' not in x],
    language = 'c++',
    library_dirs = ['.', '/usr/local/lib', '/usr/local/cuda/lib64'],
    libraries = ['gpu-kernels', 'cudart', 'ceres', 'glog', \
            'amd','btf','camd','ccolamd','cholmod','colamd','cxsparse',\
            'graphblas','klu','ldl','rbio','spqr','umfpack', 'lapack', 'blas', 'gcc'] + opencv_libs,
    include_dirs = [numpy.get_include()]+['/usr/include/eigen3']
)

setup(
    name='pyvoldor_full',
    description='voldor visual odometry',
    author='Zhixiang Min',
    ext_modules=cythonize([ext])
)

