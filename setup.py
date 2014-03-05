try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
        'description': 'Analysis of gravity currents measured with PIV',
        'author': "Aaron O'Leary",
        'url': 'http://github.com/aaren/lab_turbulence',
        'download_url': 'http://github.com/aaren/lab_turbulence/download',
        'author_email': 'eeaol@leeds.ac.uk',
        'version': '0.1',
        'install_requires': ['nose'],
        'packages': ['gc_turbulence'],
        'scripts': [],
        'name': 'gc_turbulence',
        'entry_points': {'console_scripts': ['tt = gc_turbulence:cli']}
        }

setup(**config)
