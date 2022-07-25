#!/usr/bin/env python
"""The setup script."""
from setuptools import find_packages, setup

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
    install_requires=[],
    include_package_data=True,
    keywords="fgsim",
    name="fgsim",
    packages=find_packages(
        include=["fgsim"],
    ),
    setup_requires=["setuptools"],
    url="https://github.com/mova/fgsim",
    version="0.1.0",
    zip_safe=False,
)
