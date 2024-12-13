"""
Created on Fri 2nc Feb 2024
@author: Thomas Alscher, NHMD
"""

from setuptools import setup, find_packages

setup(
    name='dirku',
    packages=find_packages(where="."),  # Look for packages in the current directory
    version='1.0.0',
    author='Thomas Alscher',
    author_email='',
    description='Deformable image registration framework',

    install_requires=[
        # List your package dependencies here
    ],

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
    ],
)