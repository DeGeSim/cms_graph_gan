#!/usr/bin/env python
"""The setup script."""
from setuptools import find_packages, setup

with open("docs/readme.rst") as readme_file:
    readme = readme_file.read()

with open("docs/history.rst") as history_file:
    history = history_file.read()


setup_requirements = ["setuptools"]
install_requirements = [
    "torch",
    "torch-geometric",
    "torch-sparse",
    "torch-scatter",
    "torch-spline-conv",
    "matplotlib",
    "seaborn",
    "omegaconf",
    "pretty-errors",
    "prettytable",
    "typeguard",
    "tqdm",
    "rich",
    "h5py",
    "uproot",
    "awkward",
    "comet-ml",
    "tensorboard",
    "tblib",
]

extras = {
    "dev": ["jedi", "rope", "pylint", "ipython"],
}

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
    extras_require=extras,
    url="https://github.com/mova/fgsim",
    version="0.1.0",
    zip_safe=False,
)
