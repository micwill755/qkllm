from setuptools import setup, find_packages, Extension
import sys

class get_pybind_include:
    def __str__(self):
        try:
            import pybind11
            return pybind11.get_include()
        except ImportError:
            return ""

# Try to build C++ extensions, but don't fail if pybind11 is missing
ext_modules = []
try:
    import pybind11
    ext_modules = [
        Extension(
            'mtrx.c_matmul',
            sources=[
                'mtrx/csrc/matmul.cpp',
                'mtrx/csrc/bindings.cpp',
            ],
            include_dirs=[
                get_pybind_include(),
                'mtrx/csrc',
            ],
            language='c++',
            extra_compile_args=['-std=c++14', '-O3'] if sys.platform != 'win32' else ['/std:c++14', '/O2'],
        ),
    ]
except ImportError:
    print("Warning: pybind11 not found. Installing without C++ backend.")
    print("To enable C++ backend: pip install pybind11 && pip install -e .")
    ext_modules = []

setup(
    name="mtrx",
    version="0.1.0",
    packages=find_packages(),
    ext_modules=ext_modules,
    install_requires=['pybind11>=2.6.0'] if ext_modules else [],
    python_requires=">=3.7",
)