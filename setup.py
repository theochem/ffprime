#!/usr/bin/env python
"""Setup & Install Script."""


from setuptools import setup, find_packages


setup(
    name="ffprime",
    version="0.0.0",
    description="Force Field derivation and evaluation.",
    author="QcDevs Team",
    package_dir={"ffprime": "ffprime"},
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "ffprime = ffprime.scripts.main:main",
        ],
    },
    install_requires=[
        "numpy",
        "scipy==1.16.2",
        "matplotlib",
        "horton",
        "qc-iodata @ git+https://github.com/theochem/iodata.git@main#egg=qc-iodata",
        "qc-grid @ git+https://github.com/theochem/grid.git@master#egg=qc-grid",
        "qc-gbasis @ git+https://github.com/theochem/gbasis.git@master#egg=qc-gbasis",
        "denspart@ git+https://github.com/theochem/denspart.git@main#egg=denspart",
        "qc-atomdb @ git+https://github.com/theochem/AtomDB.git@master#egg=qc-atomdb",
    ],
    dependency_links=[
        "git+https://github.com/theochem/iodata.git@main#egg=qc-iodata",
        "git+https://github.com/theochem/grid.git@main#egg=qc-grid",
        "git+https://github.com/theochem/gbasis.git@main#egg=qc-gbasis",
        "git+https://github.com/theochem/denspart.git@main#egg=denspart",
        "git+https://github.com/theochem/AtomDB.git@master#egg=qc-atomdb",
        
    ],
)
