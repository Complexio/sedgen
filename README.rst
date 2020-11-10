======
SedGen
======

|travis| |cov| |docs|


Sediment generation model in Python.


Description
===========

This Python package covers a sediment generation model written as the product of 4 years of PhD research. 
It enables you to initialize a parent rock and let it weather mechanically and chemically to sediment.
For now, only granitoid rocks have been tested to work with this model but the rock suite may be expanded in the future.

Installation
============
1. Clone this repo onto your own computer whether via forking or copying.
2. Create a conda env or virtualenv with the following dependencies: 

    - numpy
    - numba
    - pandas
    - matplotlib
    - seaborn
    - fast-histogram
3. Start a cmd/powershell window in the location where you cloned the repo locally (or move into that path via cd).
4. Activate the recently created environment.
5. Run 'python setup.py' to statically install the sedgen package via pip or run 'python setup.py develop' to dynamically install it. The former option is best if you copied the repo instead of forked, since you'll probably want a frozen version of sedgen? The latter option is for when you forked the repo and want to easily receive updates locally to the sedgen package when you pull new commits from the sedgen repo (see https://www.dataschool.io/simple-guide-to-forks-in-github-and-git/ for more info on syncing forks locally).
6. Start a python interface and try to run 'import sedgen' (still in your dedicated environment of course). If no errors are shown, installation should be succesful!

Once the package is out of alfa stage, it will be packaged to pypi and.or conda for easier installation/updates.

Usage
=====

Initilization
-------------
SedGen has been developed in a object oriented and modular way.
Therefore, its usage is similar to the use of a scikit-learn class (e.g., PCA from sklearn.decomposition).
You start of by providing three fundamental properties to initiliaze a SedGen model along with some optional environmental properties.

The fundamental properties are:
    - Modal mineralogy, which characterizes the rock's composition by providing the proportions in which different minerals occur in the rock
    - Interfacial composition, which characterizes the rock's texture by providing the proportions of different crystal contacts within the rock
    - Crystal size probability distributions, which charaterize the sizes of the crystal present within the rock by providing the corresponding crystal size means and standard deviations per mineral class
    
The environmental properties are:
    - Parent rock volume, to declare how big of a volume of rock should be initialized at the start of the model. A good setting for this is between 0.01 and 1 m³ which translates to 10_000_000 - 1_000_000_000 mm³ to pass in as value.
    - Number of timesteps, to declare how many timesteps (iterations) of the weathering should go through. It should be noted that this in relative terms as no dating links have been made between SedGen model's outcomes and reality, for now.
    - Etc. (see docs for more optional parameters to pass)
    
Weathering
----------
Once a model has been initialized its '.weathering()' function can be called to start a sequence of weathering operations.

Note
====

This project has been set up using PyScaffold 3.2.2. For details and usage
information on PyScaffold see https://pyscaffold.org/.


.. |docs| image:: https://readthedocs.org/projects/sedgen/badge/?version=latest 
    :alt: Documentation Status
    :scale: 100%
    :target: https://sedgen.readthedocs.io/en/latest/?badge=latest

.. |cov| image:: https://codecov.io/gh/Complexio/sedgen/branch/master/graph/badge.svg
    :alt: Code Coverage
    :scale: 100%
    :target: https://codecov.io/gh/Complexio/sedgen

.. |travis| image:: https://travis-ci.org/Complexio/sedgen.svg?branch=master
    :alt: Travis CI
    :scale: 100%
    :target: https://travis-ci.org/Complexio/sedgen
