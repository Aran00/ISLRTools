from distutils.core import setup
from setuptools import find_packages


PACKAGE = "islrtools"
NAME = "islrtools"
DESCRIPTION = "Common files for ISLR code"
AUTHOR = "ryu"
AUTHOR_EMAIL = "ryu@microstrategy.com"
URL = ""
VERSION = __import__(PACKAGE).__version__

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    # license="BSD",
    url=URL,
    # packages=find_packages(exclude=["tests.*", "tests"]),
    packages=['islrtools'],
    package_dir={'islrtools': 'islrtools'},
    package_data={'islrtools': ['data/*.csv']},
    # py_modules = ['lrplot', 'tableplot'],
    # data_files=['islrtools/data'],
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "statsmodels",
        "sklearn"
    ],
    zip_safe=False
)

'''
package_data=find_package_data(
    PACKAGE,
    only_in_packages=False
),
'''