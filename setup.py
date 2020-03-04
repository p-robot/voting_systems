#!/usr/bin/env python3

from setuptools import setup, find_packages
from distutils.core import setup

DESCRIPTION = "Voting systems for decision-making with multiple epidemiological model"
NAME = "voting_systems"
AUTHOR = "Will Probert"
AUTHOR_EMAIL = "william.probert@bdi.ox.ac.uk"
MAINTAINER = "Will Probert"
MAINTAINER_EMAIL = "william.probert@bdi.ox.ac.uk"
URL = "https://github.com/p-robot/voting_systems"
DOWNLOAD_URL = "https://github.com/p-robot/voting_systems"
LICENSE = "Apache License, Version 2.0"
VERSION = "1.0"

setup(  
    name = NAME,
    version = VERSION,
    description = DESCRIPTION,
    author = AUTHOR,
    author_email = AUTHOR_EMAIL,
    maintainer = MAINTAINER,
    maintainer_email = MAINTAINER_EMAIL,
    url = URL,
    download_url = DOWNLOAD_URL,
    license = LICENSE,
    packages=find_packages()) #["voting_systems"]) # , "voting_systems.data"

#package_data={"voting_systems": ["data/ebola_models.csv",]},
