#!/usr/bin/env python3

from setuptools import setup, find_packages

setup(
    name="orcinus",
    version='0.1.0',
    author="Vasiliy Sheredeko <piphon@gmail.com>",
    license="Copyright 2018-2019 (C) Vasiliy Sheredeko",
    description="Compiler for Python-like static typing language",
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'orcinus = orcinus.cli:main',
        ],
    },
    install_requires=[
        'llvmlite==0.28.0',
        'multidict==4.5.2',
        'colorlog==3.1.4',
        'pytest==4.3.0',
    ],
    extras_require={
        'mkdocs': [
            'mkdocs',
            'mkdocs-material',
            'markdown-checklist',
            'pygments',
        ]
    },
    include_package_data=True,
)
