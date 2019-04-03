from setuptools import setup
from codecs import open
from os import path
import re

package_name = 'causallift'

# Read version from __init__.py file
root_dir = path.abspath(path.dirname(__file__))
with open(path.join(root_dir, package_name, '__init__.py')) as f:
    init_text = f.read()
    version = re.search(r'__version__\s*=\s*[\'\"](.+?)[\'\"]', init_text).group(1)

setup(
    name=package_name,
    packages=[package_name],
    version=version,
    license='BSD 2-Clause',
    author='Yusuke Minami',
    author_email='me@minyus.github.com',
    url='https://github.com/Minyus/causallift',
	description='CausalLift: Python package for Uplift Modeling for A/B testing and observational data.',
	install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'scikit-learn',
        'xgboost'
    ],
	keywords='uplift lift causal propensity ipw observational',
	zip_safe=False,
    test_suite='tests',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Natural Language :: English',
        "License :: OSI Approved :: BSD License",
        'Programming Language :: Python :: 3.6',
        "Operating System :: OS Independent"
    ])
