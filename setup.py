from pkg_resources import parse_requirements
from setuptools import find_packages, setup
 
def load_requirements(file):
    """Parse requirements from file"""
    with open(file, "r") as reqs:
        return [str(req) for req in parse_requirements(reqs)]

setup(
    name='ltfs',
    packages=find_packages(),
    version='0.1.0',
    install_requires=load_requirements('./requirements.txt'),
)