#!/usr/bin/env python

"""The setup script."""

# needed for build the cpp extension
# from glob import glob
# import numpy as np
# from pybind11.setup_helpers import Pybind11Extension
from setuptools import find_packages, setup

with open("docs/readme.rst") as readme_file:
    readme = readme_file.read()

with open("docs/history.rst") as history_file:
    history = history_file.read()


setup_requirements = ["torch"]
install_requirements = [
    "matplotlib",
    "numpy",
    "omegaconf",
    "pretty-errors",
    "prettytable",
    "setuptools",
    "torch",
    "torch-geometric",
    "torch-sparse",
    "torch-scatter",
    "torch-spline-conv",
    "tqdm",
    "h5py",
    "comet-ml",
    "tensorboard",
    "toml",
    "tblib",
    "rich",
]

extras = {
    "test": [
        "tox",
        "coverage",
        "flake8",
        "isort",
        "black",
        "pytest",
        "pytest-runner",
    ],
    "code": ["pylint", "jedi", "rope"],
}

ext_modules = [
    # Pybind11Extension(
    #     "geomapper",
    #     sources=sorted(glob("fgsim/geo/*.cpp")),
    #     include_dirs=[
    #         np.get_include(),
    #         "fgsim/geo/libs/xtensor/include",
    #         "fgsim/geo/libs/xtensor-python/include",
    #         "fgsim/geo/libs/xtl/include",
    #     ],
    #     extra_compile_args=["-std=c99", "-Wno-error=vla"],
    # ),
]

setup(
    author="Anonymous",
    author_email="mova@users.noreply.github.com",
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python :: 3.8",
    ],
    description="Fast simulation of the HGCal using neural networks.",
    entry_points={
        "console_scripts": [
            "fgsim=fgsim.__main__.py:main",
        ],
    },
    install_requires=install_requirements,
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="fgsim",
    name="fgsim",
    packages=find_packages(
        include=["fgsim"], exclude=["fgsim/geo/libs/.*", "xtensor-python"]
    ),
    setup_requires=setup_requirements,
    ext_modules=ext_modules,
    extras_require=extras,
    url="https://github.com/mova/fgsim",
    version="0.1.0",
    zip_safe=False,
)
