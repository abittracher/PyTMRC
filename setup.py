# -*- coding: utf-8 -*-


from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='tram',
    version='0.1.0',
    description='Transition Manifold Package',
    long_description=readme,
    author='Andreas Bittracher',
    author_email='bittracher@mi.fu-berlin.de',
    url='',
    license=license,
    packages=find_packages(exclude=('demos', 'docs'))
)