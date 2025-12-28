"""
Setup script for crm_torch - PyTorch C++ extension for CRM physics.

This builds a native PyTorch extension that wraps the CRM physics engine
with OpenMP parallelization for batched forward/backward passes.
"""

from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension
import os
import sys

# Get paths relative to CRM_ML root
crm_ml_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_dir = os.path.join(crm_ml_root, 'src')
numerical_dir = os.path.join(src_dir, 'numerical')
third_party_dir = os.path.join(crm_ml_root, 'third_party')
autodiff_dir = os.path.join(third_party_dir, 'autodiff')

# CRM C++ source files to compile directly into extension
crm_sources = [
    os.path.join(src_dir, 'CRM_IVPSolver.cpp'),
    os.path.join(src_dir, 'CRM_BVPSolver.cpp'),
    os.path.join(src_dir, 'CRM_IVPJacobian.cpp'),
    os.path.join(src_dir, 'CRM_ForwardKinematics.cpp'),
    os.path.join(src_dir, 'CRM_CatheterClass.cpp'),
    os.path.join(src_dir, 'CRM_SupportFunctions.cpp'),
    os.path.join(numerical_dir, 'minpack.cpp'),
    os.path.join(numerical_dir, 'minpack_DYN_Defs.cpp'),
    os.path.join(src_dir, 'CoilDynamics_Defs.cpp'),
]

# Extension source files (relative to crm_torch/ directory)
extension_sources = [
    os.path.join(os.path.dirname(__file__), 'csrc', 'crm_torch_binding.cpp'),
    os.path.join(os.path.dirname(__file__), 'csrc', 'dynamics_op.cpp'),
]

# Combine all sources
all_sources = extension_sources + crm_sources

# Include directories
include_dirs = [
    src_dir,
    numerical_dir,
    autodiff_dir,
    third_party_dir,
    'csrc',  # For torch_utils.hpp
    '/usr/include/eigen3',  # Eigen3 system installation
]

# Compiler flags
extra_compile_args = [
    '-std=c++17',
    '-O3',
    '-fopenmp',  # Enable OpenMP for parallel batching
    '-Wall',
    '-Wno-sign-compare',
    '-Wno-unused-variable',
    '-Wno-unused-but-set-variable',
]

# Linker flags
# Get torch library path dynamically
try:
    import torch
    torch_lib_path = os.path.join(os.path.dirname(torch.__file__), 'lib')
except ImportError:
    torch_lib_path = None

extra_link_args = ['-fopenmp']  # Link OpenMP library
if torch_lib_path:
    extra_link_args.append(f'-Wl,-rpath,{torch_lib_path}')  # Add PyTorch lib to runtime path

# Platform-specific adjustments
if sys.platform == 'darwin':  # macOS
    # macOS may need libomp from Homebrew
    extra_compile_args.remove('-fopenmp')
    extra_link_args.remove('-fopenmp')
    # Try to add libomp if available
    if os.path.exists('/usr/local/opt/libomp'):
        extra_compile_args.append('-Xpreprocessor')
        extra_compile_args.append('-fopenmp')
        extra_link_args.append('-lomp')
        include_dirs.append('/usr/local/opt/libomp/include')
        extra_link_args.append('-L/usr/local/opt/libomp/lib')

# Define the extension
ext_modules = [
    CppExtension(
        name='_crm_torch_ext',
        sources=all_sources,
        include_dirs=include_dirs,
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    )
]

setup(
    name='crm_torch',
    version='0.1.0',
    author='CRM_ML Team',
    description='PyTorch C++ extension for CRM catheter physics simulation',
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtension
    },
    packages=['crm_torch'],  # The crm_torch package
    python_requires='>=3.8',
    install_requires=[
        'torch>=1.9.0',
        'numpy>=1.19.0',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Programming Language :: C++',
    ],
)
