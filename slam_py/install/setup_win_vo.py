from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from glob import glob
import os
import numpy
import shutil

# Change these folders
opencv_include_dir = 'C:/opencv/build/include'
opencv_lib_dir = 'C:/opencv/build/x64/vc15/lib'
opencv_lib_name = 'opencv_world3414'

# Change this if you target different host/device
nvcc_machine_code = '-m64 -arch=compute_61 -code=sm_61' 

gpu_sources_cpp = ' '.join(glob('../../gpu-kernels/*.cpp'))
gpu_sources_cu = ' '.join(glob('../../gpu-kernels/*.cu'))

gpu_kernel_build_cmd = f'nvcc {gpu_sources_cpp} {gpu_sources_cu} -I {opencv_include_dir} -L {opencv_lib_dir} -l {opencv_lib_name} -o ./gpu-kernels.dll -O3 --shared -cudart static {nvcc_machine_code} -Xcompiler /wd4819'
os.system(gpu_kernel_build_cmd)

ext = Extension('pyvoldor_vo',
    sources = ['pyvoldor_vo.pyx'] + \
            [x for x in glob('../../voldor/*.cpp') if 'main.cpp' not in x],
    language = 'c++',
    library_dirs = ['./gpu-kernels.lib', opencv_lib_dir],
    libraries = ['gpu-kernels', opencv_lib_name],
    include_dirs = [numpy.get_include(), opencv_include_dir]
)

setup(
    name='pyvoldor_vo',
    description='voldor visual odometry',
    author='Zhixiang Min',
    ext_modules=cythonize([ext])
)

