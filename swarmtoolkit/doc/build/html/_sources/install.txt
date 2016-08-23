Installation requirements
-------------------------

``swarmtoolkit`` requires::

- Python (>=3.2)
- Numpy (>=1.5)
- Scipy (>=0.14)
- matplotlib(>=1.5)
- basemap (>=1.0, from mpl_toolkits)
- spacepy (>=0.1.5)
- numexpr (>=2.4)
- ftputil (>=3.0)
- iminuit (>=1.0)

This *should* be all you need to do to get started with swarmtoolkit:

Install python, C-compile w/ python headers, which for ubuntu the following should suffice::

    apt-get install build-essential python3-dev


then download `miniconda <http://conda.pydata.org/miniconda.html>`_ and run the bash/exe installer.

install required packages::

    conda install numpy scipy matplotlib spacepy basemap numexpr pip ipython \
      ipython-notebook

    pip install ftputil iminuit

Then everything should be ready to be run. If you want to use ``swarmtoolkit``, either type in:: 

    python setup.py install 

in the root folder of swarmtoolkit, or, manually or add ``swarmtoolkit`` to your pythonpath in a ``.bash_profile`` or ``.bashrc`` file eg:: 

    export PYTHONPATH=$PYTHONPATH:/path/to/swarmtoolkit/directory

to use jupyter/ipython notebook just type ``ipython notebook`` in a terminal (optionally add a file path of a notebook file) and it should start up in a browser.
