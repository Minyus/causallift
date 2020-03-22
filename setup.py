# Copyright 2018-2019 QuantumBlack Visual Analytics Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
# NONINFRINGEMENT. IN NO EVENT WILL THE LICENSOR OR OTHER CONTRIBUTORS
# BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF, OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# The QuantumBlack Visual Analytics Limited ("QuantumBlack") name and logo
# (either separately or in combination, "QuantumBlack Trademarks") are
# trademarks of QuantumBlack. The License does not grant you any right or
# license to the QuantumBlack Trademarks. You may not use the QuantumBlack
# Trademarks or any confusingly similar mark as a trademark for your product,
#     or use the QuantumBlack Trademarks in any other manner that might cause
# confusion in the marketplace, including but not limited to in advertising,
# on websites, or on software.
#
# See the License for the specific language governing permissions and
# limitations under the License.

from setuptools import find_packages, setup

entry_point = "causallift = causallift.run:main"

# get the dependencies and installs
with open("requirements.txt", "r") as f:
    requires = [x.strip() for x in f if x.strip()]

with open("requirements_optional.txt", "r") as f:
    requires_optional = [x.strip() for x in f if x.strip()]

with open("requirements_docs.txt", "r") as f:
    requires_docs = [x.strip() for x in f if x.strip()]

with open("requirements_dev.txt", "r") as f:
    requires_dev = [x.strip() for x in f if x.strip()]

setup(
    name="causallift",
    version="1.0.1",  # Align with __version__ in __init__.py
    packages=find_packages(where="src", exclude=["tests"]),
    package_dir={"": "src"},
    entry_points={"console_scripts": [entry_point]},
    install_requires=requires,
    extras_require=dict(
        optional=requires_optional, docs=requires_docs, dev=requires_dev
    ),
    license="BSD 2-Clause",
    author="Yusuke Minami",
    author_email="me@minyus.github.com",
    url="https://github.com/Minyus/causallift",
    description="CausalLift: Python package for Uplift Modeling for A/B testing and observational data.",
    keywords="uplift lift causal propensity ipw observational",
    zip_safe=False,
    test_suite="tests",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Natural Language :: English",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Operating System :: OS Independent",
    ],
)
