import os
from setuptools import setup, find_packages

def get_version():
    with open(os.path.join('swarmtoolkit','__init__.py'),'r') as f:
        for line in f:
            if line.strip()[:11].lower()=='__version__':
                version='v'+line.split('=')[1].strip('" \n\t').strip("'")
                break
    return version

setup(
	name = "swarmtoolkit",
	version = get_version(),
	author = 'Mikael Toresen',
	author_email = 'mikael.toresen@esa.int',
	description = 'a toolbox intended to provide simple and quick access'+\
	              ' to the Swarm L1 and L2Cat2 products for Python3',
	
	packages=['swarmtoolkit'],
	install_requires = ['numpy>=1.5.0','spacepy>=0.1.5','scipy>=0.14',
	                    'matplotlib>=1.5','basemap>=1.0','numexpr>2.4',
	                    'ftputil>=3.0','iminuit>=1.0'],
  
	url = '',
	
	
)

