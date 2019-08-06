# -*- coding: utf-8 -*-
"""
Installs:

    - ocrd-evaluate-segmentation
"""
import codecs

from setuptools import setup, find_packages

setup(
    name='ocrd_evaluate_segmentation',
    version='0.0.1',
    description='Segmentation evaluation',
    long_description=codecs.open('README.md', encoding='utf-8').read(),
    author='Konstantin Baierer, Kay-Michael Würzner, Robert Sachunsky',
    author_email='unixprog@gmail.com, wuerzner@gmail.com, sachunsky@informatik.uni-leipzig.de',
    url='https://github.com/OCR-D/ocrd_evaluate_segmentation',
    license='Apache License 2.0',
    packages=find_packages(exclude=('tests', 'docs')),
    install_requires=open('requirements.txt').read().split('\n'),
    package_data={
        '': ['*.json', '*.yml', '*.yaml'],
    },
    entry_points={
        'console_scripts': [
            'ocrd-evaluate-segmentation=ocrd_evaluate_segmentation.cli:ocrd_evaluate_segmentation',
        ]
    },
)
