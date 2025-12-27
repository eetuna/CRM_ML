"""
CRM Torch C++ Extension - Build Configuration
Option C: Native PyTorch Custom Operator

This extension wraps the existing CRM_ML C++ dynamics implementation
with PyTorch's autograd system for optimal performance.
"""

import os
import sys
from pathlib import Path
from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CppExtension

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
EXT_ROOT = Path(__file__).parent.absolute()

# Source files for the extension (relative to crm_torch_ext directory)
extension_sources = [
    str(EXT_ROOT / 'csrc' / 'crm_step_op.cpp'),
    str(EXT_ROOT / 'csrc' / 'bindings.cpp'),
]

# Include directories
include_dirs = [
    str(PROJECT_ROOT / 'src'),
    str(PROJECT_ROOT / 'src' / 'numerical'),
    str(PROJECT_ROOT / 'third_party' / 'autodiff'),
    '/usr/include/eigen3',  # System Eigen3
]

# Library directories
library_dirs = [
    str(PROJECT_ROOT / 'build'),
]

# Libraries to link against
libraries = [
    'CRMCPPLib',  # The existing CRM library (static lib)
]

# Compiler flags (match the main project build)
extra_compile_args = {
    'cxx': [
        '-std=c++17',
        '-O3',
        '-DNUM_ACT_SET=1',  # Match main build configuration
        '-DUSE_AUTODIFF',
    ],
}

# Define the extension module
ext_modules = [
    CppExtension(
        name='_crm_torch_ext',
        sources=extension_sources,
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_compile_args=extra_compile_args['cxx'],
        language='c++',
    )
]

setup(
    name='crm_torch_ext',
    version='0.1.0',
    author='CRM_ML Team',
    description='PyTorch C++ Extension for CRM Catheter Dynamics',
    long_description=open(PROJECT_ROOT / 'README.md').read() if (PROJECT_ROOT / 'README.md').exists() else '',
    long_description_content_type='text/markdown',
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtension
    },
    packages=['crm_torch_ext'],
    python_requires='>=3.8',
    install_requires=[
        'torch>=1.12.0',
        'numpy>=1.20.0',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Programming Language :: C++',
        'Topic :: Scientific/Engineering',
    ],
)
