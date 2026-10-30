#!/usr/bin/env python3
"""Setup script for ks3-vectors-embed CLI."""

import os
from setuptools import setup, find_packages

# Read version from ks3_vectors/__version__.py without importing the package
here = os.path.abspath(os.path.dirname(__file__))
version = {}
with open(os.path.join(here, "ks3_vectors", "__version__.py")) as f:
    exec(f.read(), version)

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ks3-vectors-embed-cli",
    version=version["__version__"],
    author="Kingsoft Cloud KS3",
    author_email="",
    maintainer="Kingsoft Cloud KS3",
    maintainer_email="",
    description="Standalone CLI for KS3 Vector operations with pluggable embeddings",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ks3sdklib/ks3-vectors-embed-cli",
    project_urls={
        "Bug Reports": "https://github.com/ks3sdklib/ks3-vectors-embed-cli/issues",
        "Source": "https://github.com/ks3sdklib/ks3-vectors-embed-cli",
        "Documentation": "https://github.com/ks3sdklib/ks3-vectors-embed-cli#readme",
        "Homepage": "https://github.com/ks3sdklib/ks3-vectors-embed-cli",
    },
    packages=find_packages(),
    keywords="ks3, vectors, embeddings, object-storage, cli, machine-learning, ai",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: System Administrators",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Systems Administration",
        "Topic :: Utilities",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Environment :: Console",
    ],
    python_requires=">=3.9",
    install_requires=[
        "ks3sdk>=1.19.0",
        "click>=8.0.0",
        "rich>=12.0.0",
    ],
    entry_points={
        "console_scripts": [
            "ks3-vectors-embed=ks3_vectors.cli:main",
        ],
    },
)
